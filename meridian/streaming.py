"""
Meridian - Streaming Support
Stream responses from LLM providers
"""

from typing import Generator, Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class StreamChunk:
    """A chunk of streamed response"""
    text: str
    token_count: int
    is_final: bool
    latency_ms: float  # Time since start


@dataclass
class StreamResult:
    """Final result after streaming completes"""
    output: str
    tokens_in: int
    tokens_out: int
    total_latency_ms: float
    time_to_first_token_ms: float
    chunks: int


def stream_openai(
    client,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    on_chunk: Optional[Callable[[StreamChunk], None]] = None
) -> Generator[StreamChunk, None, StreamResult]:
    """
    Stream from OpenAI-compatible API.
    
    Args:
        client: OpenAI client
        model: Model name
        prompt: Input prompt
        temperature: Temperature
        max_tokens: Max tokens
        on_chunk: Optional callback for each chunk
        
    Yields:
        StreamChunk for each token/chunk
        
    Returns:
        StreamResult with final stats
    """
    start = time.time()
    first_token_time = None
    full_output = ""
    chunk_count = 0
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            now = time.time()
            
            if first_token_time is None:
                first_token_time = now
            
            full_output += content
            chunk_count += 1
            
            stream_chunk = StreamChunk(
                text=content,
                token_count=1,  # Approximate
                is_final=False,
                latency_ms=(now - start) * 1000
            )
            
            if on_chunk:
                on_chunk(stream_chunk)
            
            yield stream_chunk
    
    end = time.time()
    
    # Final chunk
    final_chunk = StreamChunk(
        text="",
        token_count=0,
        is_final=True,
        latency_ms=(end - start) * 1000
    )
    yield final_chunk
    
    return StreamResult(
        output=full_output,
        tokens_in=len(prompt) // 4,
        tokens_out=len(full_output) // 4,
        total_latency_ms=(end - start) * 1000,
        time_to_first_token_ms=(first_token_time - start) * 1000 if first_token_time else 0,
        chunks=chunk_count
    )


def stream_deepseek(
    prompt: str,
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    max_tokens: int = 256,
    on_chunk: Optional[Callable[[StreamChunk], None]] = None
) -> Generator[StreamChunk, None, StreamResult]:
    """
    Stream from DeepSeek API.
    """
    import os
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    
    return stream_openai(
        client=client,
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        on_chunk=on_chunk
    )


def stream_generic(
    adapter,
    prompt: str,
    config = None,
    on_chunk: Optional[Callable[[StreamChunk], None]] = None
) -> StreamResult:
    """
    Non-streaming fallback that simulates streaming.
    Useful for adapters that don't support native streaming.
    """
    from core.model_adapters.base import GenerationConfig
    
    config = config or GenerationConfig()
    
    start = time.time()
    result = adapter.generate(prompt, config)
    end = time.time()
    
    # Simulate chunks from final result
    words = result.output.split()
    for i, word in enumerate(words):
        chunk = StreamChunk(
            text=word + " ",
            token_count=1,
            is_final=(i == len(words) - 1),
            latency_ms=(end - start) * 1000
        )
        if on_chunk:
            on_chunk(chunk)
    
    return StreamResult(
        output=result.output,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        total_latency_ms=(end - start) * 1000,
        time_to_first_token_ms=(end - start) * 1000,  # Same as total for non-streaming
        chunks=len(words)
    )


class StreamingAdapter:
    """Wrapper that adds streaming to any adapter"""
    
    def __init__(self, adapter):
        self.adapter = adapter
        self._supports_native = self._check_native_support()
    
    def _check_native_support(self) -> bool:
        """Check if adapter supports native streaming"""
        model_id = getattr(self.adapter, 'model_id', '')
        return any(p in model_id for p in ['openai', 'deepseek', 'anthropic'])
    
    def stream(
        self,
        prompt: str,
        config = None,
        on_chunk: Optional[Callable[[StreamChunk], None]] = None
    ) -> StreamResult:
        """
        Stream response from the adapter.
        
        Uses native streaming if supported, otherwise falls back to simulation.
        """
        model_id = getattr(self.adapter, 'model_id', '')
        
        if 'deepseek' in model_id:
            # Use native DeepSeek streaming
            result = None
            for chunk in stream_deepseek(prompt, on_chunk=on_chunk):
                if chunk.is_final:
                    pass  # Generator returns result
            # Fallback to non-streaming for now
            return stream_generic(self.adapter, prompt, config, on_chunk)
        
        # Fallback to simulated streaming
        return stream_generic(self.adapter, prompt, config, on_chunk)
    
    @property
    def supports_streaming(self) -> bool:
        return self._supports_native

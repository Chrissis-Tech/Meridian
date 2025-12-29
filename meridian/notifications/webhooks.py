"""
Meridian - Webhook Notifications
Send alerts to Slack, Discord, or custom URLs
"""

import os
import json
from typing import Optional, Literal
from dataclasses import dataclass
import requests
from datetime import datetime


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    url: str
    type: Literal["slack", "discord", "teams", "generic"] = "generic"
    enabled: bool = True
    events: list[str] = None  # ["run_complete", "regression", "threshold_breach"]
    
    def __post_init__(self):
        if self.events is None:
            self.events = ["run_complete", "regression", "threshold_breach"]


class WebhookNotifier:
    """Send notifications via webhooks"""
    
    def __init__(self, config: Optional[WebhookConfig] = None):
        self.config = config or self._load_from_env()
    
    def _load_from_env(self) -> Optional[WebhookConfig]:
        """Load webhook config from environment"""
        url = os.getenv("WEBHOOK_URL")
        if not url:
            return None
        
        return WebhookConfig(
            url=url,
            type=os.getenv("WEBHOOK_TYPE", "generic"),
            enabled=os.getenv("WEBHOOK_ENABLED", "true").lower() == "true"
        )
    
    def send(self, event: str, data: dict) -> bool:
        """
        Send a webhook notification.
        
        Args:
            event: Event type (run_complete, regression, threshold_breach)
            data: Event data
            
        Returns:
            True if sent successfully
        """
        if not self.config or not self.config.enabled:
            return False
        
        if event not in self.config.events:
            return False
        
        try:
            payload = self._format_payload(event, data)
            response = requests.post(
                self.config.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            return response.status_code < 400
        except Exception as e:
            print(f"Webhook error: {e}")
            return False
    
    def _format_payload(self, event: str, data: dict) -> dict:
        """Format payload based on webhook type"""
        
        if self.config.type == "slack":
            return self._format_slack(event, data)
        elif self.config.type == "discord":
            return self._format_discord(event, data)
        elif self.config.type == "teams":
            return self._format_teams(event, data)
        else:
            return self._format_generic(event, data)
    
    def _format_slack(self, event: str, data: dict) -> dict:
        """Format for Slack webhook"""
        emoji = self._get_emoji(event, data)
        title = self._get_title(event, data)
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {title}"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Suite:*\n{data.get('suite', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Model:*\n{data.get('model', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Accuracy:*\n{data.get('accuracy', 0):.1%}"},
                    {"type": "mrkdwn", "text": f"*Tests:*\n{data.get('passed', 0)}/{data.get('total', 0)}"},
                ]
            }
        ]
        
        if data.get("regression"):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⚠️ *Regression:* {data['regression']:.1%} drop from baseline"
                }
            })
        
        return {"blocks": blocks}
    
    def _format_discord(self, event: str, data: dict) -> dict:
        """Format for Discord webhook"""
        emoji = self._get_emoji(event, data)
        title = self._get_title(event, data)
        
        color = 0x00FF00 if event == "run_complete" and data.get("accuracy", 0) >= 0.7 else 0xFF0000
        
        return {
            "embeds": [{
                "title": f"{emoji} {title}",
                "color": color,
                "fields": [
                    {"name": "Suite", "value": data.get("suite", "N/A"), "inline": True},
                    {"name": "Model", "value": data.get("model", "N/A"), "inline": True},
                    {"name": "Accuracy", "value": f"{data.get('accuracy', 0):.1%}", "inline": True},
                    {"name": "Tests", "value": f"{data.get('passed', 0)}/{data.get('total', 0)}", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
    
    def _format_teams(self, event: str, data: dict) -> dict:
        """Format for Microsoft Teams webhook"""
        emoji = self._get_emoji(event, data)
        title = self._get_title(event, data)
        
        return {
            "@type": "MessageCard",
            "themeColor": "0076D7",
            "summary": title,
            "sections": [{
                "activityTitle": f"{emoji} {title}",
                "facts": [
                    {"name": "Suite", "value": data.get("suite", "N/A")},
                    {"name": "Model", "value": data.get("model", "N/A")},
                    {"name": "Accuracy", "value": f"{data.get('accuracy', 0):.1%}"},
                    {"name": "Tests", "value": f"{data.get('passed', 0)}/{data.get('total', 0)}"},
                ]
            }]
        }
    
    def _format_generic(self, event: str, data: dict) -> dict:
        """Format for generic webhook"""
        return {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
    
    def _get_emoji(self, event: str, data: dict) -> str:
        if event == "threshold_breach":
            return "🚨"
        elif event == "regression":
            return "⚠️"
        elif data.get("accuracy", 0) >= 0.8:
            return "✅"
        elif data.get("accuracy", 0) >= 0.5:
            return "⚡"
        else:
            return "❌"
    
    def _get_title(self, event: str, data: dict) -> str:
        suite = data.get("suite", "Unknown")
        if event == "run_complete":
            return f"Meridian: {suite} Complete"
        elif event == "regression":
            return f"Meridian: Regression in {suite}"
        elif event == "threshold_breach":
            return f"Meridian: Threshold Breach in {suite}"
        return f"Meridian: {event}"


# Convenience functions
_notifier = None

def get_notifier() -> WebhookNotifier:
    global _notifier
    if _notifier is None:
        _notifier = WebhookNotifier()
    return _notifier

def notify_run_complete(suite: str, model: str, accuracy: float, passed: int, total: int):
    """Send notification for completed run"""
    get_notifier().send("run_complete", {
        "suite": suite,
        "model": model,
        "accuracy": accuracy,
        "passed": passed,
        "total": total
    })

def notify_regression(suite: str, model: str, current: float, baseline: float):
    """Send notification for regression"""
    get_notifier().send("regression", {
        "suite": suite,
        "model": model,
        "accuracy": current,
        "baseline": baseline,
        "regression": baseline - current
    })

def notify_threshold_breach(suite: str, model: str, accuracy: float, threshold: float):
    """Send notification for threshold breach"""
    get_notifier().send("threshold_breach", {
        "suite": suite,
        "model": model,
        "accuracy": accuracy,
        "threshold": threshold
    })

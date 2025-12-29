# Contributing to Meridian

Thank you for your interest in contributing to Meridian.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Chrissis-Tech/Meridian.git
cd Meridian

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

## Code Style

We use `ruff` for linting:

```bash
# Check
ruff check core/ tests/

# Fix automatically
ruff check --fix core/ tests/
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/your-feature`)
3. **Make changes** and add tests
4. **Run tests** to ensure they pass
5. **Commit** with descriptive message
6. **Push** to your fork
7. **Open a Pull Request** against `main`

### PR Checklist

- [ ] Tests pass locally
- [ ] New functionality has tests
- [ ] Code follows existing style
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated

## Test Suites

When adding new test suites:

1. Create JSONL file in `suites/`
2. Follow existing schema with `id`, `prompt`, `expected`
3. Add capability and failure mode labels
4. Update DATASET_CARD.md if adding to core dataset

## Scoring Functions

When adding new scorers:

1. Create file in `core/scoring/`
2. Return `ScoringResult` dataclass
3. Add to `__init__.py` exports
4. Add tests in `tests/test_scoring.py`

## Questions?

Open a GitHub Discussion for questions or ideas.

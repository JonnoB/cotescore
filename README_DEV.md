# Development Guide

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install model dependencies (optional, for running models)
pip install -e ".[models]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cot_score --cov=models --cov=benchmarks

# Run specific test file
pytest tests/test_metrics.py

# Run specific test
pytest tests/test_metrics.py::TestCoverage::test_coverage_perfect_match
```

## Code Quality

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy cot_score models benchmarks
```

## Project Structure

```
cot_analysis/
├── cot_score/          # Core metrics library
│   ├── metrics.py      # Coverage and Overlap metrics
│   └── dataset.py      # NCSE dataset loader
├── models/             # Model loading and inference
│   └── loader.py       # Base model class
├── benchmarks/         # Benchmarking framework
│   └── runner.py       # Evaluation orchestration
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── data/               # Dataset storage (gitignored)
└── results/            # Evaluation results (gitignored)
```

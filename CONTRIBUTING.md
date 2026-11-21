# Contributing to QCPlot

Thank you for your interest in contributing to **QCPlot**! We welcome bug reports, feature requests, statistical formulation improvements, and code contributions.

---

## Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure an open, welcoming, and inclusive community.

---

## Getting Started

### 1. Fork & Clone
Fork the repository on GitHub and clone it locally:
```bash
git clone https://github.com/codeXparadise/QCPlot.git
cd QCPlot
```

### 2. Set Up Environment
We recommend using [uv](https://github.com/astral-sh/uv) for fast, isolated dependency management:
```bash
# Install dependencies with uv
uv sync

# Or with standard pip
pip install -r requirements.txt
pip install -e .
```

### 3. Initialize Models
Download default model weights and verify hardware acceleration (DirectML / CUDA / CPU):
```bash
uv run qcplot --init
```

---

## Development Guidelines

- **Code Quality**: Follow standard PEP 8 conventions. Use clear type annotations wherever possible.
- **Statistical Accuracy**: Any changes to control limits or unbiasing factors must conform to **ASTM E2587** and **ISO 7870-2** standards.
- **Zero-Regressions**: Test modifications across both variable charts ($X\text{-bar } R, X\text{-bar } s, I\text{-}MR$) and attribute charts ($p, np, c, u$) using datasets in `data/`.
- **Commit Messages**: Write concise, descriptive commit messages prefixing the changed scope (e.g., `feat:`, `fix:`, `docs:`, `math:`).

---

## Submitting Pull Requests

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear, descriptive messages.
3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Open a Pull Request against the `main` branch with a summary of changes, problem solved, and output screenshots if visual plotting is affected.

---

## Reporting Issues

If you encounter unexpected errors or statistical discrepancies:
- Check existing GitHub Issues first.
- Open a new Issue describing the expected vs actual behavior, input data structure, hardware device (DirectML / CUDA / CPU), and full terminal traceback.

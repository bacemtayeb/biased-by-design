# Contributing to Biased by Design

Thank you for your interest in improving this research. Contributions are welcome in the following areas:

## Areas for Contribution

### 1. Detection & Defense
- Implement detection methods for community-adapted LLM outputs
- Develop adversarial training approaches
- Create coordinated fingerprinting detectors
- Build platform-native detection models (Twitter/Grok, Reddit internal)

### 2. Evaluation
- Human evaluation infrastructure (annotation tools, Cohen's kappa)
- Cross-platform generalization tests (Reddit → Twitter/Facebook)
- Temporal dynamics (adapter drift as communities evolve)
- Alternative evaluation metrics

### 3. Reproducibility
- Data collection automation (Arctic Shift scraper)
- Pre-built Docker images for training environment
- Cloud deployment guides (AWS, GCP, Azure)
- Reproducibility checklist validation

### 4. Documentation
- Tutorial notebooks (Jupyter)
- Architecture diagrams
- API documentation
- Troubleshooting guides

## Contribution Guidelines

### Before Contributing

1. **Read the ethical considerations** in the main README
2. **Open an issue** to discuss major changes before starting work
3. **Check existing issues** to avoid duplicate work

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features
- Update relevant README files

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following code standards
3. **Test your changes** locally
4. **Update documentation** (README, docstrings, etc.)
5. **Submit a pull request** with:
   - Clear description of changes
   - Link to related issue (if applicable)
   - Screenshots/outputs if relevant

### Commit Messages

Use clear, descriptive commit messages:

```
Good:
- "Add detection bypass eval for GPT-4o detector"
- "Fix LoRA weight blending bug in adapter interpolation"
- "Document hardware requirements for v2 adapters"

Avoid:
- "fix bug"
- "update code"
- "changes"
```

## Ethical Guidelines

**Do NOT contribute:**
- Features that make malicious deployment easier (automation scripts, account management, posting tools)
- Obfuscation techniques to evade detection
- Pre-trained adapters ready for malicious use

**Do contribute:**
- Detection methods
- Defense mechanisms
- Evaluation tools
- Documentation improvements
- Reproducibility enhancements

## Dual-Use Concerns

This research has dual-use potential. When contributing:

1. **Think defensively** — prioritize detection over attack
2. **Document risks** — note potential misuse in PRs
3. **Request review** — tag maintainers for security-sensitive changes

## Questions?

Open an issue or email: bacemtayeb@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see [LICENSE](LICENSE)).

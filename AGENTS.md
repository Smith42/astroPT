# AGENTS.md

## Build/Lint/Test Commands

- `uv sync` - Install dependencies and sync environment
- `uv run ruff check` - Run linting (E, F, I rules, ignore E501)
- `uv run ruff format` - Format code
- `uv run pre-commit run --all-files` - Run all pre-commit hooks
- `python scripts/train.py --batch_size=32 --compile=False` - Single GPU training
- `torchrun --standalone --nproc_per_node=4 scripts/train.py` - Multi-GPU training

## Code Style Guidelines

- **Imports**: Standard library first, then third-party (torch, numpy, etc.), then local imports
- **Formatting**: Use ruff format, skip magic trailing comma disabled
- **Types**: Use dataclasses for configuration objects, type hints encouraged
- **Naming**: snake_case for variables/functions, PascalCase for classes, descriptive names
- **Error handling**: Use try/except for optional imports with fallback warnings
- **Docstrings**: Use triple quotes with brief description, Arguments section for complex functions
- **Line length**: E501 ignored (no strict limit), but keep reasonable
- **Comments**: Inline comments for complex logic, especially in model architecture
- **Constants**: ALL_CAPS for module-level constants
- **File structure**: Group related functionality, separate concerns (model, datasets, utils)

test:
	uv run pytest tests/ --cov=src --cov-report=term-missing --disable-warnings -q
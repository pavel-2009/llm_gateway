test:
	uv run pytest tests/ 

coverage:
	uv run pytest --cov=src/llm_gateway --cov-report=term-missing tests/

up:
	docker compose up -d
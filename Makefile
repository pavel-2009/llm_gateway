test:
	uv run pytest tests/ 

coverage:
	uv run pytest --cov=src/llm_gateway --cov-report=term-missing tests/

USE_OLLAMA_CONTAINER := $(shell awk -F= '$$1 == "USE_OLLAMA_CONTAINER" { print $$2; exit }' .env 2>/dev/null)

up:
	@if [ "$(USE_OLLAMA_CONTAINER)" = "true" ]; then \
		COMPOSE_PROFILES=ollama OLLAMA_BASE_URL=http://ollama:11434 docker compose up -d; \
	else \
		docker compose up -d; \
	fi

down:
	docker compose down

jaeger-logs:
	docker compose logs -f jaeger
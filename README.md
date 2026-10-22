# LLM Gateway

FastAPI-шлюз для моделей OpenAI, Anthropic и Ollama.

## Локальный запуск

Установите проект и запустите console script:

```bash
uv sync
uv run llm-gateway
```

По умолчанию сервер слушает `0.0.0.0:8000`. Адрес и порт задаются параметрами
`HOST` и `PORT`.

## Запуск в Docker

Создайте `.env` из примера и запустите Compose:

```bash
cp .env.example .env
docker compose up --build -d
```

В `.env` задаются те же настройки, что и при локальном запуске. Пример
направляет запросы Ollama на `host.docker.internal`, то есть на Ollama,
запущенную на хост-машине. Убедитесь, что Ollama запущена и модель из
`OLLAMA_MODEL` доступна. Для проверки доступности сервера выполните:

```bash
curl -f http://localhost:8000/openapi.json
```

Для проверки генерации через Ollama:

```bash
curl -f http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Ответь одним словом: привет","model":"light"}'
```

Посмотреть логи и остановить сервис:

```bash
docker compose logs -f gateway
docker compose down
```

В Compose уже настроен адрес `host.docker.internal` для доступа к хосту, в том
числе на Linux Docker Engine.

## Настройки

Параметры читаются из переменных окружения или из `.env` в рабочем каталоге.
В Compose файл `.env` передаётся контейнеру; используйте `.env.example` как
шаблон и не добавляйте `.env` с секретами в git.

`OPENAI_API_KEY` и `ANTHROPIC_API_KEY` необязательны: если облачные ключи
отсутствуют, шлюз пропускает соответствующих провайдеров и использует
настроенные fallback-модели, включая Ollama. Для Anthropic также поддерживается
прежнее имя `CLAUDE_API_KEY`. Таймауты и остальные параметры retry и circuit
breaker настраиваются через одноимённые переменные из `core/config.py`.
Некорректные критические настройки завершают запуск с ошибкой валидации.
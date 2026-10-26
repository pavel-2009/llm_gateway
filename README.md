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

Compose запускает Ollama, сохраняет её модели в volume `ollama_data` и
загружает `qwen3:8b` перед запуском шлюза. В Compose адрес Ollama автоматически
переопределяется на `http://ollama:11434`; `OLLAMA_BASE_URL` из `.env` остаётся
полезен для локального запуска шлюза вне Docker. Для проверки доступности
сервера выполните:

```bash
curl -f http://localhost:8000/openapi.json
```

Метрики приложения доступны на `http://localhost:8000/metrics`, а веб-интерфейс
Prometheus — на `http://localhost:9090`. Grafana с преднастроенным dashboard
«LLM Gateway» доступна на `http://localhost:3000` (локальные учётные данные по
умолчанию: `admin` / `admin`). В Grafana подключены Prometheus для графиков и
Jaeger для поиска трейсов. Отдельный интерфейс Jaeger доступен на
`http://localhost:16686`. Compose включает экспорт трасс OpenTelemetry через
OTLP/HTTP; трейсы связывают полный HTTP-запрос, публичную LLM-операцию,
выбор модели и fallback, policy/retry, вызов провайдера и задержку HTTP-запроса
к Ollama, OpenAI или Anthropic. Для streaming в span операции указывается время
до первого токена, а длительность span показывает время всего потока. OTLP
порты Jaeger доступны на хосте только через loopback. Содержимое prompt и ключи
провайдеров в атрибуты трейсов не записываются.
Для локального запуска без Compose экспорт выключен по умолчанию; задайте
`OTEL_ENABLED=true` и `OTEL_EXPORTER_OTLP_ENDPOINT`, чтобы включить его.
Перед публикацией Grafana за пределами локальной машины задайте свой
`GRAFANA_ADMIN_PASSWORD`.
Jaeger использует in-memory storage для локальной отладки: его трейсы очищаются
при перезапуске контейнера.

При первом запуске Compose ждёт, пока
сервис `ollama-model` скачает модель; до окончания загрузки gateway и его
метрики не запускаются. Порт gateway внутри Compose фиксирован на `8000`, а
переменная `PORT` задаёт порт публикации на хосте.

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

При `docker compose down -v` сохранённая модель также удаляется, и при следующем
запуске будет загружена заново.

## Нагрузочное тестирование шлюза без реальной модели

Нагрузочный сценарий использует Locust и небольшой Ollama-совместимый mock:
он случайно генерирует ответы, задержки и HTTP-ошибки, не загружая и не вызывая
настоящую модель. Для этого сценария **не запускайте стандартный `docker compose
up`**: Compose скачивает реальную модель Ollama.

Установите зависимости проекта (включая Locust из группы `dev`):

```bash
uv sync --dev
```

Запустите mock, указав вероятность ошибки и диапазон задержек. Он поддерживает
форматы Ollama для генерации, структурированного извлечения и NDJSON-stream:

```bash
MOCK_MODEL_ERROR_RATE=0.1 \
MOCK_MODEL_MIN_DELAY_SECONDS=0.02 \
MOCK_MODEL_MAX_DELAY_SECONDS=0.15 \
uv run uvicorn loadtest.mock_ollama:app --host 127.0.0.1 --port 11435
```

В другом терминале запустите шлюз без облачных ключей, настроив Ollama URL на
mock. `model=light` в сценарии направляет запросы только к этому провайдеру:

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11435 \
OLLAMA_MODEL=mock-model \
OPENAI_API_KEY= \
ANTHROPIC_API_KEY= \
uv run llm-gateway
```

В третьем терминале запустите Locust. Сценарий равномерно нагружает все три
публичные операции шлюза — `/generate`, `/extract` и `/stream`; каждая проходит
через HTTP API, middleware/метрики/трейсинг, gateway, resolver, policy/retry и
настоящий Ollama provider adapter. Вместо модели отвечает mock-сервис:

```bash
uv run locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
```

Откройте `http://localhost:8089`, задайте число пользователей и скорость их
запуска. Для запуска без UI и сохранения CSV-отчёта:

```bash
uv run locust -f loadtest/locustfile.py \
  --host http://127.0.0.1:8000 \
  --users 50 --spawn-rate 5 --run-time 2m --headless \
  --csv loadtest/results/llm-gateway-load
```

Результаты каждого прогона сохраняются в `loadtest/results/` в CSV-файлы
`llm-gateway-load_stats.csv`, `llm-gateway-load_stats_history.csv` и
`llm-gateway-load_failures.csv`. Locust автоматически создаёт каталог.

Locust измеряет пропускную способность отдельно по каждому endpoint, число
запросов и ошибок, а также
распределение времени ответа (включая p50/p95/p99 в отчёте). Метрики самого
шлюза доступны во время и после прогона на `http://localhost:8000/metrics`:
число запросов и их длительность по HTTP-маршрутам и статусам, длительность и
исход LLM-операций, активные запросы к провайдеру и число повторных попыток.
Ошибки mock намеренно считаются ошибками нагрузки, если после настроенных
повторных попыток шлюз не смог вернуть успешный ответ.

По умолчанию шлюз повторяет часть временных ошибок. Чтобы измерить исходную
долю ошибок mock без повторных попыток, задайте `MAX_RETRIES=0` при запуске
шлюза; чтобы нагрузочно проверить политику повторов, оставьте настройку по
умолчанию.

## Настройки

Параметры читаются из переменных окружения или из `.env` в рабочем каталоге.
В Compose файл `.env` передаётся контейнеру; используйте `.env.example` как
шаблон и не добавляйте `.env` с секретами в git.

`OPENAI_API_KEY` и `ANTHROPIC_API_KEY` необязательны: если облачные ключи
отсутствуют, шлюз пропускает соответствующих провайдеров и использует
настроенные fallback-модели, включая Ollama. Для Anthropic также поддерживается
прежнее имя `CLAUDE_API_KEY`. Таймауты и параметры retry настраиваются через
одноимённые переменные из `core/config.py`.
Некорректные критические настройки завершают запуск с ошибкой валидации.
# Structured Data Extractor

FastAPI service for extracting typed JSON from PDF documents. It parses PDF
text, asks an LLM to call a schema-specific tool, validates the returned
arguments with Pydantic, and stores extraction/audit records in PostgreSQL.

Supported document types:

- `invoice`
- `contract`
- `payslip`
- `receipt`

## Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI |
| PDF parsing | PyMuPDF |
| LLM calls | LiteLLM |
| Schemas | Pydantic v2 |
| Database | PostgreSQL + SQLModel |
| Background jobs | Arq + Redis |
| Tooling | uv, Ruff, Vulture, pytest |

## How It Works

```text
FastAPI endpoint
  -> DocumentService
  -> DocumentProcessor
  -> ClassifierService, if doc_type is omitted
  -> PageStrategyRunner
  -> ToolCallExtractor
  -> Validation + confidence scoring
  -> Audit + LLM usage rows
```

`DocumentService` is the document endpoint facade. It owns document-facing
flows such as uploads, existing-document extraction, batch extraction,
corrections, cost reports, and reads.

## Prerequisites

- Python 3.12
- uv
- Docker and Docker Compose
- One provider key matching your configured LiteLLM model:
  - `GEMINI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`

## Quick Start

```bash
cp .env.example .env
```

Set at least:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/extractor
ARQ_REDIS_URL=redis://localhost:6379
GEMINI_API_KEY=your_key
```

Start Postgres and Redis:

```bash
docker compose up db redis
```

Install dependencies and migrate:

```bash
uv sync
uv run alembic upgrade head
```

Run the API:

```bash
uv run fastapi dev
```

Run the worker when testing async extraction:

```bash
uv run arq app.workers.worker.WorkerSettings
```

API docs:

```text
http://localhost:8000/docs
```

## Docker

Run the full stack:

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`
- Arq worker

Stop:

```bash
docker compose down
```

Remove volumes too:

```bash
docker compose down -v
```

## Configuration

The app reads `.env` through Pydantic settings.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | unset | PostgreSQL async SQLAlchemy URL |
| `ARQ_REDIS_URL` | `redis://localhost:6379` | Redis URL for Arq |
| `STORAGE_LOCAL_DIR` | `./uploads` | Local PDF storage directory |
| `MAX_UPLOAD_SIZE_BYTES` | `26214400` | Upload limit, 25 MB |
| `LLM_EXTRACTION_MODEL` | `gemini/gemini-2.5-flash` | Model used for extraction |
| `LLM_CLASSIFIER_MODEL` | `gemini/gemini-2.5-flash` | Model used when `doc_type` is omitted |
| `LLM_MAX_RETRIES` | `2` | Tool-call retry count |
| `COST_MODEL_TOKEN_PRICES_USD` | sample map | Per-token model prices for reports |

Classifier and extraction models are separate. A practical setup is to use a
stable cheap model for classification and a stronger model for extraction:

```env
LLM_CLASSIFIER_MODEL=gemini/gemini-2.5-flash
LLM_EXTRACTION_MODEL=gemini/gemini-3-flash-preview
```

`COST_MODEL_TOKEN_PRICES_USD` values are per-token USD prices. If a provider
lists prices per 1M tokens, divide by 1,000,000.

## Endpoints

All endpoints are under `/api/v1`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Database-backed health check |
| `GET` | `/schemas` | List extraction schemas |
| `GET` | `/documents/{document_id}` | Fetch document status |
| `POST` | `/documents/extractions` | Upload and extract one PDF |
| `POST` | `/documents/extractions?document_id=...` | Extract an existing document |
| `POST` | `/documents/extractions/batch` | Extract up to 20 PDFs |
| `GET` | `/documents/extractions/{extraction_id}` | Fetch extraction result |
| `GET` | `/documents/extractions/{extraction_id}/audit` | Fetch raw audit details |
| `POST` | `/documents/extractions/{extraction_id}/correct` | Store human correction |
| `GET` | `/documents/extractions/cost-report` | Token and cost report |

## Examples

Automatic classification:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/extractions" \
  -F "file=@samples/receipt.pdf"
```

Explicit document type:

```bash
curl -X POST \
  "http://localhost:8000/api/v1/documents/extractions?doc_type=receipt" \
  -F "file=@samples/receipt.pdf"
```

Async extraction:

```bash
curl -X POST \
  "http://localhost:8000/api/v1/documents/extractions?doc_type=contract&async=true" \
  -F "file=@samples/contract.pdf"
```

Extract an existing document:

```bash
curl -X POST \
  "http://localhost:8000/api/v1/documents/extractions?document_id={document_id}&doc_type=invoice"
```

Batch extraction:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/extractions/batch" \
  -F "files=@samples/invoice-1.pdf" \
  -F "files=@samples/invoice-2.pdf"
```

Cost report:

```bash
curl "http://localhost:8000/api/v1/documents/extractions/cost-report"
```

Date-filtered cost report:

```bash
curl "http://localhost:8000/api/v1/documents/extractions/cost-report?from=2026-01-01T00:00:00&to=2026-01-31T23:59:59"
```

Human correction:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/extractions/{extraction_id}/correct" \
  -H "Content-Type: application/json" \
  -d '{
    "corrected_data": {
      "merchant_name": "Namecheap, Inc.",
      "receipt_date": "2026-05-31",
      "items": [],
      "total_amount": "6.99",
      "currency": "USD"
    },
    "corrected_by": "reviewer@example.com",
    "note": "Corrected receipt fields"
  }'
```

## Extraction Strategies

| Strategy | Behavior |
| --- | --- |
| `full` | One LLM call over the full document text |
| `smart` | First page, last page, and pages with amount keywords |
| `page_by_page` | Extract each page and merge schema fields |

When omitted, documents over 10 pages use `smart`; shorter documents use
`full`.

## Confidence Scores

Current confidence is a simple schema-driven heuristic:

- required field with value: higher confidence
- optional field with value: medium confidence
- missing or invalid field: `0`
- suspicious value: lowered confidence

This is not evidence-based confidence. It does not prove the value appeared
clearly in the PDF. Evidence-based scoring would compare extracted values back
to source text, which is more complex and should wait until confidence drives
human review or routing decisions.

## Tests

Run everything:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check .
```

Dead-code scan:

```bash
uv run vulture app tests --min-confidence 80
```

Opt-in Postgres integration tests:

```bash
POSTGRES_TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/test_db \
  uv run pytest tests/test_postgres_integration.py
```

Opt-in live LLM smoke tests:

```bash
RUN_LIVE_LLM_SMOKE=1 uv run pytest tests/test_live_llm_smoke.py
```

## Strategy Evaluation

Compare extraction strategies on a local PDF:

```bash
uv run python scripts/evaluate_strategies.py path/to/document.pdf \
  --doc-type invoice
```

The script writes `docs/evaluation/strategy-comparison.md` by default.

## Migrations

Apply migrations:

```bash
uv run alembic upgrade head
```

Create a migration after model changes:

```bash
uv run alembic revision --autogenerate -m "describe change"
```

Docker Compose applies migrations automatically in the API service.

## Project Structure

```text
app/
  api/              FastAPI dependencies and endpoints
  core/             settings, database, lifecycle, health
  infrastructure/   local storage
  integrations/llm/ LiteLLM client boundary
  lib/              PDF and confidence helpers
  models/           SQLModel database models
  repositories/     database queries
  schemas/          request, response, and document schemas
  services/         document workflow and extraction logic
  workers/          Arq worker
migrations/         Alembic migrations
scripts/            utility scripts
tests/              unit, endpoint, and opt-in integration tests
```

## Troubleshooting

`DATABASE_URL must be configured`

Set `DATABASE_URL` in `.env`, or run through Docker Compose.

Async extraction runs inline

Redis is unavailable. Start Redis and check `ARQ_REDIS_URL`.

Classification returns `unknown`

Pass `doc_type` explicitly, or use a stable classifier model. Preview models
  can be less reliable for exact-label classification.

Cost report fails for an unknown model

Add the model to `COST_MODEL_TOKEN_PRICES_USD`.

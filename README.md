# GridWise LLM Energy Optimizer

GridWise is a FastAPI service for the BUP CSE Fest 2026 online preliminary. It accepts a 24-hour campus demand, solar, tariff and battery scenario; interprets one to three operator notes; and returns a low-cost schedule that satisfies the official energy and directive constraints.

## Architecture

```text
Request
  -> LLM Interpreter
  -> Deterministic Guardrails
  -> PuLP/CBC Optimizer
  -> Independent Replay Validator
  -> JSON Response
```

The interpreter converts every note into one of the six official directive types. Guardrails validate that untrusted structured output. The optimizer schedules grid, solar and battery energy. The final validator independently replays the serialized response before HTTP 200 is allowed.

## Technology

- Python 3.11
- FastAPI, Uvicorn and Pydantic 2
- OpenAI SDK for the primary language model integration
- PuLP with CBC for linear optimization
- pytest and HTTPX for automated and external testing

## Current integration status

The production interpreter, deterministic guardrails, PuLP/CBC optimizer, response builder and independent replay validator are integrated. The merged automated suite passes, and all ten official public cases passed against the real Dockerized pipeline with runtime credentials.

## Environment variables

Copy `.env.example` to `.env` and fill only the credentials needed by the integrated interpreter:

| Variable | Purpose | Default |
|---|---|---|
| `PORT` | HTTP listening port | `8000` |
| `OPENAI_API_KEY` | Primary model API credential | none |
| `OPENAI_MODEL` | Primary model identifier | `gpt-5.6-terra` |
| `CODECRAFT_API_KEY` | Optional fallback credential | none |
| `CODECRAFT_BASE_URL` | Optional fallback API endpoint | `https://codecraftapi.com/v1` |
| `CODECRAFT_MODEL` | Optional fallback model | `claude-fable-5` |

Never commit `.env`, keys, tokens or provider credentials.

## Local setup

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS), then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the service:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check readiness:

```bash
curl -i http://localhost:8000/health
```

Expected body:

```json
{"status":"ok"}
```

To send a complete official request without copying 24 rows manually:

```bash
python -c "import json; d=json.load(open('BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json', encoding='utf-8')); print(json.dumps(d['cases'][0]['input']))" > sample-request.json
curl -X POST http://localhost:8000/optimize-energy -H "Content-Type: application/json" --data-binary @sample-request.json
```

This POST requires valid runtime model credentials.

## Tests

Run the complete local suite:

```bash
python -m pytest -q
```

Run all ten official public inputs against a running service:

```bash
python scripts/test_public.py --base-url http://localhost:8000
```

The runner checks HTTP/JSON behavior, response shape, interpretation semantics and latency. It does not require explanation text or an exact reference schedule. To test a deployed service:

```bash
BASE_URL=https://your-service.example python -m pytest -q tests/test_external_smoke.py
```

On PowerShell, set `$env:BASE_URL` first. Without `BASE_URL`, external tests skip cleanly.

## Docker

Build and run locally:

```bash
docker build -t gridwise:test .
docker run --rm -p 8000:8000 -e PORT=8000 --env-file .env gridwise:test
```

Credentials are supplied at runtime and are never copied into the image. The image has been build-verified, including exact `/health` behavior and a real 24-hour `/optimize-energy` response.

## Production deployment

Production UI:
https://gridwise-llm-henna.vercel.app/

Health:
GET https://gridwise-llm-henna.vercel.app/health

Optimization:
POST https://gridwise-llm-henna.vercel.app/optimize-energy

## Provider and optimizer

The primary interpreter uses the configurable `OPENAI_MODEL` through the OpenAI Responses API with structured output, a bounded retry, and deterministic validation. The CodeCraft configuration provides an OpenAI-compatible chat-completions fallback.

The optimizer uses PuLP and its bundled CBC solver over the 24-hour continuous linear model. Overlapping solar reductions use the strictest remaining usable-solar factor.

## Known limitations

- The production deployment passed 10/10 official public cases; observed latency was p50 8.115 seconds and p95 11.141 seconds.
- Hosted model availability depends on runtime credentials, quota and provider health.

## Security and credits

The service validates request data, treats model output as untrusted, sanitizes internal failures and independently replays completed schedules. Keep credentials in environment variables, rotate any exposed credential, and ensure local key files are never added to Git or container images.

This project uses Python, FastAPI, Uvicorn, Pydantic, HTTPX, the OpenAI SDK, PuLP/CBC and pytest. AI coding assistants were used during implementation and verification; the team remains responsible for architecture, correctness and submitted behavior.

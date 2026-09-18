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

OpenAI is the primary language-model interpreter and CodeCraft is the generative backup. The interpreter converts every note into one of the six official directive types, deterministic guardrails validate the untrusted structured output, and PuLP/CBC performs cost optimization. The final validator independently replays the serialized response before HTTP 200 is allowed.

## Technology

- Python 3.11
- FastAPI, Uvicorn and Pydantic 2
- OpenAI SDK for the primary language model integration
- PuLP with CBC for linear optimization
- pytest and HTTPX for automated and external testing

## Current integration status

The production interpreter, deterministic guardrails, PuLP/CBC optimizer, response builder and independent replay validator are integrated. The merged automated suite passes, and all ten official public cases passed against the real Dockerized pipeline with runtime credentials.

## Production

Base URL: https://gridwise-llm-henna.vercel.app/

Endpoints:

- `GET /health`
- `POST /optimize-energy`

Expected health response:

```json
{"status":"ok"}
```

## Environment variables

Copy `.env.example` to `.env` and configure these variable names:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `CODECRAFT_API_KEY`
- `CODECRAFT_BASE_URL`
- `CODECRAFT_MODEL`

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
curl http://localhost:8000/health
```

Expected body:

```json
{"status":"ok"}
```

## Sample request / response

Create the official SAMPLE-01 request without copying 24 rows manually:

```bash
python -c "import json; d=json.load(open('BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json', encoding='utf-8')); print(json.dumps(d['cases'][0]['input']))" > sample-request.json
```

Submit it:

```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  --data-binary @sample-request.json
```

Representative response structure:

```text
{
  "scenario_id": "...",
  "directive_interpretation": [...],
  "hourly_plan": [...],
  "total_grid_kwh": ...,
  "total_cost_bdt": ...,
  "peak_grid_kwh": ...,
  "plan_summary": "..."
}
```

This POST requires valid runtime model credentials. Complete official worked examples are in `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json`.

## Tests

Run the complete local suite:

```bash
python -m pytest -q
```

Verified current suite: **166 passed, 4 skipped, 28 subtests passed**.

Run all ten official public inputs against a running service:

```bash
python scripts/test_public.py --base-url http://localhost:8000 --timeout 30
```

Run the same official regression against production:

```bash
python scripts/test_public.py --base-url https://gridwise-llm-henna.vercel.app --timeout 30
```

Last verified official public regression: **10/10**.

The runner checks HTTP/JSON behavior, response shape, interpretation semantics and latency. It does not require explanation text or an exact reference schedule.

## Docker fallback

Build the submitted Dockerfile:

```bash
docker build -t gridwise-llm:submission .
```

Run it with credentials supplied at runtime:

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  gridwise-llm:submission
```

Check health:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

The image builds successfully from the submitted Dockerfile. Local Docker `/health` and a real `/optimize-energy` request were verified. The service exposes port 8000, Uvicorn binds to `0.0.0.0`, credentials are supplied only at runtime, and no secrets are baked into the image.

## Provider and optimizer

The primary interpreter uses the configurable `OPENAI_MODEL` through the OpenAI Responses API with structured output, a bounded retry, and deterministic validation. The CodeCraft configuration provides an OpenAI-compatible chat-completions fallback.

The optimizer uses PuLP and its bundled CBC solver over the 24-hour continuous linear model. Overlapping solar reductions use the strictest remaining usable-solar factor.

## Known limitations

- Hosted model availability depends on provider health, network access and quota.
- Latency can vary with hosted model response time.

## Security and credits

The service validates request data, treats model output as untrusted, sanitizes provider and internal failures, and independently replays completed schedules. Supply secrets only through environment variables, never commit `.env` or API keys, and never bake provider credentials into container images.

This project uses Python, FastAPI, Uvicorn, Pydantic, HTTPX, the OpenAI SDK, PuLP/CBC and pytest. AI coding assistants were used during implementation and verification; the team remains responsible for architecture, correctness and submitted behavior.

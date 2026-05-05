# CounterAI Backend

Rails API for CounterAI. The React frontend lives under `frontend/`.

## Run locally

Requires PostgreSQL and repo-root `.env` for `PGUSER` / `PGPASSWORD` (`../.env.example`).

```bash
bin/rails db:prepare
bin/rails server
```

Default: `http://localhost:3000`.

## Detector

[`app/jobs/detector_job.rb`](app/jobs/detector_job.rb) runs `model/classify.py` in a subprocess (paths overridable with `CLASSIFIER_*` env vars). The default checkpoint pin is documented in `model/docs/MODEL_ABLATION_PLAN.md` (Phase G6).

## HTTP endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/up` | Health check for load balancers. |
| `POST` | `/file_hashes/upload` | Multipart upload — field `file`. Images only (JPEG/PNG/WebP/GIF); 25 MB max. |
| `GET` | `/file_hashes/:hash` | Lookup `ai_status` by lowercase SHA-256 hex (polling after upload). 404 means “not stored yet”. |
| `POST` | `/file_hashes/check` | JSON `{ hashes: [...] }` → existence map. |
| `OPTIONS` | Above paths | CORS preflight. |

Production CORS: set **`FRONTEND_ORIGINS`** to your static frontend origin(s). Development still allows localhost Vite.

## Further ops

Production deployment patterns: repo-root [`DEPLOYMENT.md`](../DEPLOYMENT.md). Remaining detector hardening: [`DETECTOR_JOB_TODO.md`](../DETECTOR_JOB_TODO.md).

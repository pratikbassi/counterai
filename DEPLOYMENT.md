# CounterAI Deployment Guide

This repo is a monorepo with:

- `backend/` — Rails 8 API, PostgreSQL, Solid Queue, `DetectorJob` (subprocess to Python `model/classify.py`)
- `frontend/` — Vite + React static app
- `model/` — Training and inference (`classify.py`, pinned checkpoint for production)

## MVP deploy target

**Recommended for launch:** single Linux host running **Docker Compose** with Postgres in a companion container, optional **Caddy** for TLS (`--profile tls`). This path ships the classifier Python venv plus `model/` in one image (`Dockerfile` at repo root) so `Rails.root/../model` resolves to `/model` inside the container.

**Alternatives (not scripted in this MVP doc):**

- **Heroku** — `PROJECT_PATH=backend` subdir builds **omit** sibling `model/` and a multi-hundred-megabyte Torch stack unless you vendor them; expect slug-size friction. Prefer this Compose path or a container registry deploy.
- **Kamal** — [`backend/config/deploy.yml`](backend/config/deploy.yml) is scaffolded; point it at the **root** `Dockerfile` (or a dedicated image) once registry and hosts are chosen.

## Environment variables (runtime)

| Variable | Purpose |
|----------|---------|
| `RAILS_MASTER_KEY` | Required in production (from `backend/config/master.key`). |
| `DATABASE_URL` | PostgreSQL URL (Compose example below). |
| `CLASSIFIER_PYTHON` | Python that runs `classify.py` (default in image: `/opt/counterai/.venv/bin/python`). |
| `CLASSIFIER_SCRIPT` | Path to `classify.py` (default: `/model/classify.py`). |
| `CLASSIFIER_CHECKPOINT` | Pinned weights file (default: `/model/artifacts/best_real_fake_20260422_002356_seed42.pt`). |
| `CLASSIFIER_DEVICE` | `cpu` or `cuda` (default `cpu`). |
| `CLASSIFIER_TIMEOUT_SEC` | Subprocess wall-clock limit for `classify.py` (default `60`). |
| `FRONTEND_ORIGINS` | Comma-separated **exact** browser origins allowed for CORS in production (e.g. `https://app.example.com`). In development, `localhost:5173` is still allowed. |
| `SOLID_QUEUE_IN_PUMA` | Set `true` on a **single** web container so Solid Queue runs inside Puma (see `backend/config/puma.rb`). |
| `RAILS_MAX_THREADS` | Keep aligned with Puma threads and the DB pool (`backend/config/database.yml`). |

## Single-host Docker Compose (Postgres + web)

1. Place the promoted checkpoint under `model/artifacts/best_real_fake_20260422_002356_seed42.pt` (see [`model/docs/MODEL_ABLATION_PLAN.md`](model/docs/MODEL_ABLATION_PLAN.md) Phase G6).
2. Copy [`deploy/env.docker.example`](deploy/env.docker.example) to repo-root `.env`, set strong `POSTGRES_PASSWORD` and `RAILS_MASTER_KEY`, set `FRONTEND_ORIGINS` to your static site origin(s). `chmod 600 .env`.
3. Build and start:

   ```bash
   docker compose build
   docker compose up -d db
   docker compose run --rm web bin/rails db:prepare
   docker compose up -d web
   ```

4. API on the host: `http://127.0.0.1:8080` (maps container port 80). Health: `GET /up`.
5. **Optional TLS** — point DNS for `API_HOST` at the host, then:

   ```bash
   docker compose --profile tls up -d
   ```

   Caddy reads [`deploy/caddy/Caddyfile`](deploy/caddy/Caddyfile) and obtains Let’s Encrypt certificates.

### Postgres notes

- The `db` service has **no** published ports; only the internal Docker network can reach it.
- Backups: schedule a nightly `pg_dump` from the host (example in [MVP_LAUNCH_PLAN.md](../MVP_LAUNCH_PLAN.md) D9). Encrypt and copy off-box; run a restore drill before launch.

## Frontend (static)

From `frontend/`:

```bash
pnpm install
VITE_API_BASE_URL=https://api.example.com pnpm run build
```

Deploy `frontend/dist/` to your static host. Set `FRONTEND_ORIGINS` on the API to that origin.

## Launch smoke checklist

1. `curl -sf http://127.0.0.1:8080/up` (or `https://$API_HOST/up` behind Caddy) returns 200.
2. `POST /file_hashes/upload` with a small PNG returns 201 and a 64-char `hash`.
3. Poll `GET /file_hashes/:hash` until `ai_status` is `ai_detected` or `ai_not_detected` (not `unknown`), or confirm `DetectorJob` logs show the pinned checkpoint basename.
4. Confirm Postgres is not exposed on a public interface (`ss` / hosting firewall).
5. Confirm an unknown browser origin does **not** receive `Access-Control-Allow-Origin`.
6. **Before public traffic:** plan rate limiting (see [`MVP_LAUNCH_PLAN.md`](../MVP_LAUNCH_PLAN.md)).

## Bootstrap runbook (first production host)

See [MVP_LAUNCH_PLAN.md](../MVP_LAUNCH_PLAN.md) card **D10** for the full ordered checklist (VM provisioning, `.env`, `db:prepare`, HTTPS, curl upload, backups).

### VM hardening (summary)

SSH key-only login, `ufw` allowing only 22 / 80 / 443 from the internet as needed, unattended security updates. Details: **D11** in [MVP_LAUNCH_PLAN.md](../MVP_LAUNCH_PLAN.md).

## Detector integration (reminders)

- Subprocess inference is intentional for MVP; consider a dedicated inference HTTP service when queue latency warrants it.
- Optional hardening backlog: timeouts (implemented via `CLASSIFIER_TIMEOUT_SEC`), explicit JSON contract from `classify.py`, retries / error columns, observability (`DETECTOR_JOB_TODO.md`).

## Frontend / API constraints

- **Image-only uploads** — JPEG / PNG / WebP / GIF; content type and extension are both enforced server-side.
- **CORS** — production allows only origins listed in `FRONTEND_ORIGINS` plus dev localhost.

## Older `backend/` Dockerfile

[`backend/Dockerfile`](backend/Dockerfile) still builds backend-only Rails **without** the Python stack or `/model`; use it only if you intentionally provide the classifier elsewhere. For MVP single-host deployment, prefer the repo-root **`Dockerfile`**.

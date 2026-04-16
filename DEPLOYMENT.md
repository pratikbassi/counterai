# CounterAI Deployment Guide

This repo is a monorepo with:
- `backend`: Rails API (already Docker/Kamal-ready)
- `frontend`: Vite React app (builds to static files)
- `model`: Python training/inference code used by `DetectorJob` in production

## What deployment solutions already exist

### 1) Kamal-based backend deployment (already scaffolded)
- Existing config: `backend/config/deploy.yml`
- Existing dependency: `kamal` gem in `backend/Gemfile`
- Existing production container: `backend/Dockerfile`
- Best when you want repeatable container deploys to one or more Linux servers.

### 2) Direct Docker deployment for backend (already supported)
- `backend/Dockerfile` includes a production build and run flow.
- Useful for simple VM/container-host setups before adopting full Kamal workflows.

### 3) Static hosting for frontend (already supported by Vite build)
- `frontend` builds static assets via `npm run build`.
- Can be hosted on CDN/static platforms (Cloudflare Pages, Netlify, Vercel static, S3+CloudFront, Nginx).

## Recommended production setup

For a low-ops setup with clean scaling boundaries:
- Deploy `backend` as a Docker container (Kamal or direct Docker).
- Deploy `frontend` as static assets on a CDN/static host.
- Set `VITE_API_BASE_URL` at frontend build time to your backend URL.

This separates traffic-heavy static delivery from API compute, which scales better and is cheaper.

### Detector-aware topology (assumes detector hardening is complete)
- Web/API container: handles HTTP requests and enqueues jobs.
- Job worker container(s): runs `DetectorJob` with Python/model runtime.
- Shared persistent storage: uploaded files and DB/queue files (or object storage + Postgres in scaled setups).
- Optional GPU worker pool for higher detector throughput.

## Backend deployment (simple Docker path)

From `backend/`:

```bash
docker build -t counterai-backend .
docker run -d \
  --name counterai-backend \
  -p 3000:80 \
  -e RAILS_MASTER_KEY=your_master_key_here \
  -e DETECTOR_PYTHON=/opt/counterai/model/.venv/bin/python \
  -e DETECTOR_SCRIPT=/opt/counterai/model/classify.py \
  -e DETECTOR_CHECKPOINT=/opt/counterai/model/artifacts/best_real_fake.pt \
  -e DETECTOR_DEVICE=cpu \
  -e DETECTOR_TIMEOUT_SEC=60 \
  -e DETECTOR_MAX_RETRIES=3 \
  -v counterai_backend_storage:/rails/storage \
  counterai-backend
```

Notes:
- `RAILS_MASTER_KEY` is required in production.
- Keep `/rails/storage` on a persistent volume (SQLite DB + uploads + queue DB files).
- Health check endpoint: `GET /up`.
- Ensure detector runtime artifacts are present in the container/host:
  - Python environment
  - classifier script
  - trained checkpoint

### Separate web and worker processes (recommended)

For better throughput and isolation, run web and worker separately:

```bash
# Web/API
docker run -d --name counterai-web -p 3000:80 ... counterai-backend

# Worker (same image, different command)
docker run -d --name counterai-worker ... counterai-backend ./bin/jobs
```

Scale workers horizontally as queue latency grows.

## Backend deployment (Kamal path)

From `backend/`:
1. Update `config/deploy.yml`:
   - `servers.web`
   - `servers.job` (dedicated worker hosts/containers)
   - `registry.server` and credentials
   - optional `proxy.host` and SSL settings
2. Set secrets (including `RAILS_MASTER_KEY`) in `.kamal/secrets`.
3. Set clear env for detector runtime on relevant roles:
   - `DETECTOR_PYTHON`
   - `DETECTOR_SCRIPT`
   - `DETECTOR_CHECKPOINT`
   - `DETECTOR_DEVICE`
   - `DETECTOR_TIMEOUT_SEC`
   - retry/error-handling envs used by your finalized detector implementation
4. Deploy:

```bash
bin/kamal setup
bin/kamal deploy
```

## Frontend deployment

From `frontend/`:

```bash
npm ci
VITE_API_BASE_URL=https://api.your-domain.com npm run build
```

Deploy `frontend/dist/` to your static host.

## Important current constraints

- Current backend CORS logic should explicitly allow your deployed frontend origin(s).
- If running multi-node, avoid local-only storage coupling:
  - move from SQLite to Postgres
  - move uploads to shared object storage (or shared volume with strong guarantees)
- Detector pipeline should run with:
  - timeout enforcement
  - retry/error metadata
  - explicit label contract (`is_ai`/`class_id`) to avoid mapping drift
  - media routing (image vs video)

## Detector deployment checklist

- Confirm model checkpoint exists at `DETECTOR_CHECKPOINT`.
- Verify Python binary and script paths are correct and executable.
- Smoke-test one image and one video path through `DetectorJob`.
- Validate job retries and timeout behavior in production logs.
- Confirm confidence/model-version metadata is being persisted.
- Set alerts on detector error rate and queue delay.

## Scaling tradeoffs (quick view)

- **Fastest to launch:** single VM + web+worker in one host + frontend static host.
- **Better reliability:** separate web/worker roles + Kamal + monitored nodes + backups.
- **Higher throughput:** dedicated worker autoscaling and optional GPU workers.
- **Most scalable architecture:** Postgres + object storage + inference service (HTTP/gRPC) instead of per-job subprocess.


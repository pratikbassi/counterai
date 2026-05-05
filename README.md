# CounterAI

Web app for detecting AI-generated content in **images**.

## Repo layout

- **backend** — Rails API (file upload, hashing, file_hashes DB, detector job)
- **frontend** — React + TypeScript + Vite (File tester UI)
- **model** — training and inference (see `model/README.md`, including `ingest_upload_to_train.py` to copy images from `upload/` into `train_data` and append `train.csv`)

## Prerequisites

- **Ruby** 3.4.7 (see `backend/.ruby-version`; use [rbenv](https://github.com/rbenv/rbenv), [asdf](https://asdf-vm.com/), or another version manager)
- **Bundler** (`gem install bundler` if needed)
- **PostgreSQL** 14+ (local dev; production on Heroku uses [Heroku Postgres](https://devcenter.heroku.com/articles/heroku-postgresql))
- **Node.js** 20+ and **pnpm** (the frontend uses `pnpm-lock.yaml`)

## Local development

### 1. PostgreSQL on Ubuntu

Install the server and client tools:

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

Start PostgreSQL and optionally enable it on boot:

```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql   # optional
sudo systemctl status postgresql   # should show active (running)
```

`database.yml` connects over TCP to `localhost` using `PGUSER` / `PGPASSWORD` (see repo-root `.env`). Set a password for the DB role you use (often `postgres`):

```bash
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'choose-a-strong-password';"
```

If connection fails with “password authentication failed”, ensure `listen_addresses` includes `localhost` in `/etc/postgresql/<version>/main/postgresql.conf` and that `pg_hba.conf` allows TCP auth for local connections (for example `scram-sha-256` or `md5` for `127.0.0.1/32`); then restart: `sudo systemctl restart postgresql`.

After PostgreSQL is running, continue with `.env` and the backend steps below to create databases and run migrations.

### 2. Environment variables

At the **repository root**, copy the example env file and edit it:

```bash
cp .env.example .env
```

Set at least `PGUSER` and `PGPASSWORD` to match your local PostgreSQL role. Rails loads this file early via `backend/config/boot.rb` so `database.yml` sees the correct credentials.

### 3. Backend (Rails API)

```bash
cd backend
bundle install
bin/rails db:create db:migrate
bin/rails server
```

API default: `http://localhost:3000`. See **backend/README.md** for API behavior and the detector job.

### 4. Frontend (Vite)

In another terminal:

```bash
cd frontend
pnpm install
pnpm dev
```

Dev server defaults to `http://localhost:5173`. The UI calls the API at `http://localhost:3000` unless you set `VITE_API_BASE_URL` (for example in `frontend/.env.local`).

## Deploying to Heroku

The deployable Rails app lives in **`backend/`**. From a machine with the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and a [Heroku account](https://signup.heroku.com/):

1. **Create the app** (replace names as you like):

   ```bash
   heroku create your-app-name
   ```

2. **Attach Postgres** — Heroku sets `DATABASE_URL` for you; `backend/config/database.yml` uses it in production for the primary DB and, by default, for Solid Cache / Queue / Cable as well.

   ```bash
   heroku addons:create heroku-postgresql:essential-0 -a your-app-name
   ```

   Pick another [plan](https://devcenter.heroku.com/articles/heroku-postgres-plans) if you need more capacity.

3. **Point the build at `backend/`** — Use a subdirectory buildpack so the Ruby build runs inside `backend`:

   ```bash
   heroku buildpacks:add -i 1 https://github.com/timanovsky/subdir-heroku-buildpack -a your-app-name
   heroku buildpacks:add -i 2 heroku/ruby -a your-app-name
   heroku config:set PROJECT_PATH=backend -a your-app-name
   ```

4. **Required config vars**

   - **`RAILS_MASTER_KEY`** — Value from `backend/config/master.key` (do not commit this file to public repos).  
     `heroku config:set RAILS_MASTER_KEY="$(cat backend/config/master.key)" -a your-app-name`
   - **`SOLID_QUEUE_IN_PUMA`** — Set to `true` so Solid Queue runs inside Puma and `DetectorJob` can execute on a single dyno:  
     `heroku config:set SOLID_QUEUE_IN_PUMA=true -a your-app-name`

5. **Deploy** from this monorepo root (Heroku builds the `backend` subtree because of `PROJECT_PATH`):

   ```bash
   git push heroku main
   ```

   Use your tracked branch name if it is not `main`.

The **`backend/Procfile`** defines `release: bundle exec rails db:prepare` (migrations on each release) and `web: ./bin/thrust ./bin/rails server`. After deploy, open the app URL or run `heroku logs --tail -a your-app-name` to verify boot.

**Frontend:** Host the Vite app separately (for example Netlify or Vercel). Set `VITE_API_BASE_URL` at **build time** to your API’s public URL. Set **`FRONTEND_ORIGINS`** on the Rails API so `FileHashesController` can emit CORS headers for your static origin (`DEPLOYMENT.md`).

### Single-host Docker (recommended MVP)

Compose file, Postgres, and optional TLS (Caddy) are documented in [`DEPLOYMENT.md`](DEPLOYMENT.md). Mirror variables from [`deploy/env.docker.example`](deploy/env.docker.example) into repo-root `.env`.

## Current status

- **File tester (frontend)** — User can select an image, upload it, poll until detection finishes, and see:
  - Whether the image hash was **found in the database** or **newly added**
  - **AI content** status: *AI Detected*, *AI Not Detected*, or *Unknown* while processing
- **Backend API** — `POST /file_hashes/upload` stores the image, computes SHA-256, creates/finds a `FileHash`, and enqueues `DetectorJob`.
- **`GET /file_hashes/:hash`** — Returns `{ hash, found_in_database, ai_status }` for polling after upload.

## Detector implementation

[`backend/app/jobs/detector_job.rb`](backend/app/jobs/detector_job.rb) invokes `model/classify.py` with environment-overridable `CLASSIFIER_*` paths (see [`DEPLOYMENT.md`](DEPLOYMENT.md)).

Additional production hardening (retries, label contract, video path, observability): [`DETECTOR_JOB_TODO.md`](DETECTOR_JOB_TODO.md).

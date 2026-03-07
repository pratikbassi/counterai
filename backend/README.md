# CounterAI Backend

Rails API backend for CounterAI. Serves the API used by the frontend.

## Architecture

Monolithic Rails app: REST API, ActiveRecord models, and business logic in one codebase. Frontend is a separate React app that talks to this API.

## Run

```bash
rails s
```

Server runs by default at `http://localhost:3000`.

## Current status

- **POST /file_hashes/upload** — Accepts a file, stores it under `storage/uploads/`, computes SHA-256, and creates or finds a `FileHash` with `ai_status` (unknown / ai_detected / ai_not_detected). Returns hash, filename, size, path, `found_in_database`, and `ai_status`. Enqueues `DetectorJob` with the saved file path.
- **POST /file_hashes/check** — Batch check whether given hashes exist in the DB.
- **DetectorJob** — Enqueued after each upload with the file path. Currently a **placeholder** (logs + 3s sleep). AI detection logic is not implemented.

## Work left: DetectorJob

`DetectorJob` (`app/jobs/detector_job.rb`) is called with the **absolute path** of the uploaded file. It should:

1. **Run AI detection** on that file (image or video) using your model or external service.
2. **Update the corresponding `FileHash`** — The job currently receives only the file path. You’ll need to pass the hash (or `FileHash` id) as a second argument when enqueuing so the job can find the record and set:
   - `ai_status` to `:ai_detected` or `:ai_not_detected`.
3. **Error handling** — On failure, either leave as `unknown`, set a failure state, or retry. Keep the job idempotent where possible (same path/hash can be run more than once).

The `FileHash` model already has `ai_status` with enum values `unknown`, `ai_detected`, `ai_not_detected`. The upload response and frontend already surface these; only the job’s detection and update logic need to be implemented.

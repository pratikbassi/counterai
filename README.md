# CounterAI

Web app for detecting AI-generated content in photographs and videos.

## Repo layout

- **backend** — Rails API (file upload, hashing, file_hashes DB, detector job)
- **frontend** — React + TypeScript + Vite (File tester UI)
- **model** — (see `model/README.md`)

## Current status

- **File tester (frontend)** — User can select an image, upload it, and see:
  - Whether the image hash was **found in the database** or **newly added**
  - **AI content** status: *AI Detected*, *AI Not Detected*, or *Unknown AI content*
- **Backend API** — Upload endpoint stores the file, computes SHA-256, creates/finds a `FileHash` record with `ai_status` (unknown / ai_detected / ai_not_detected), and enqueues `DetectorJob` with the file path.
- **DetectorJob** — Placeholder only (logs and sleeps). **Real AI detection is not implemented yet**; see “Work left” below.

## Work left: DetectorJob

The job receives the uploaded file path and should:

1. **Run AI detection** on the file (image/video) using your chosen model or service.
2. **Update the `FileHash` record** with the result:
   - Set `ai_status` to `ai_detected` or `ai_not_detected` (and optionally store confidence or metadata if you add columns).
3. **Handle errors** (e.g. mark as `unknown` or retry) and consider idempotency (same file path / hash run multiple times).

Until this is implemented, all entries stay in **Unknown AI content**. The frontend and API already support the three states; only the job logic needs to be added.

See **backend/README.md** for how to run the backend and for more detail on the detector job.

# Detector Job: Current State and Remaining Work

This document is specifically about `backend/app/jobs/detector_job.rb` and what is left to make it production-ready.

## Is `DetectorJob` still a placeholder?

No, not anymore.

`DetectorJob` currently does real work:
- Looks up `FileHash` by `hash_value`
- Verifies file exists
- Uses a record lock to avoid duplicate/stale updates
- Calls Python classifier (`model/classify.py`) via `Open3.capture3`
- Parses JSON output
- Maps model label to `ai_status`
- Persists `ai_status` to DB

So the main integration path is wired. What remains is production hardening and correctness gaps.

## What still needs to be done

## 1) Make Python runtime and model path configurable

Current code assumes:
- Python binary: `model/.venv/bin/python`
- Script path: `model/classify.py`
- Checkpoint default inside Python script: `model/artifacts/best_real_fake.pt`

### Why this matters
- Deploys can fail if `.venv` path differs or checkpoint is not present at that location.
- Containerized environments usually require explicit env-driven paths.

### Action
- Add env vars in Rails job for:
  - classifier python executable
  - classifier script path
  - checkpoint path
  - device (`cpu`/`cuda`/`auto`)
- Pass checkpoint explicitly to classifier CLI (`--checkpoint`).

## 2) Add timeout and subprocess safety controls

Current `Open3.capture3` call has no timeout.

### Why this matters
- A hanging Python process can stall queue workers and reduce throughput.

### Action
- Add max execution time (e.g., `DETECTOR_TIMEOUT_SEC`).
- Kill process on timeout and keep `ai_status` unchanged (`unknown`) or move to failure state (see item 4).
- Log timeout as structured error with hash + file path.

## 3) Strengthen label mapping contract

Current mapping uses substring matching:
- AI detected if label includes tokens like `1`, `fake`, `ai`, `synthetic`, `generated`.

### Why this matters
- Can produce false mappings if label naming changes.
- Coupling is implicit, not versioned.

### Action
- Define explicit contract with classifier output, for example:
  - classifier returns canonical key: `class_id` (`0|1`) or `is_ai` (`true|false`)
- Prefer exact mapping over fuzzy substring includes.
- Keep backward compatibility for old label strings during transition.

## 4) Add failure state + retry policy

Current behavior on classifier failure:
- Logs error and returns, leaving record in `unknown`.

### Why this matters
- You cannot distinguish “not yet processed” vs “failed repeatedly”.
- Operations/monitoring cannot easily identify bad files or broken model runtime.

### Action
- Add columns (recommended):
  - `detection_attempts` (integer)
  - `last_detection_error` (text)
  - `detected_at` (datetime)
- Configure ActiveJob retries for transient failures, with capped attempts/backoff.
- After max retries, keep status deterministic (`unknown` + error metadata) or add explicit failure enum.

## 5) Gate by media type and support video path

Current classifier script is image-oriented (`PIL.Image.open` on one file).

### Why this matters
- Backend accepts image/video uploads, but current classifier path is image-only.
- Video files may fail in classifier and remain `unknown`.

### Action
- Add media-type sniffing before enqueue or in job.
- Route by type:
  - image -> current `classify.py`
  - video -> dedicated video pipeline/service
- If unsupported type, set deterministic failure metadata.

## 6) Add confidence handling policy

Current code persists only enum status.

### Why this matters
- No confidence threshold control means model uncertainty is not visible.
- Product cannot tune precision/recall tradeoff later without reprocessing.

### Action
- Persist model metadata:
  - confidence
  - model name/version
  - checkpoint id/hash
- Optionally use threshold:
  - if confidence < threshold -> keep `unknown` (or “needs review” if you add state)

## 7) Add observability and metrics

Current logs are useful but unstructured for analytics.

### Why this matters
- Hard to monitor latency, error rate, and queue backlogs at scale.

### Action
- Emit structured logs/events with:
  - hash
  - duration
  - outcome (success/error/timeout)
  - model version
- Add counters/histograms via your monitoring stack.

## 8) Add tests specifically for detector integration

Current tests mostly cover controller flows; detector integration coverage is limited.

### Why this matters
- Regression risk when changing model output format, timeouts, and retry logic.

### Action
- Add job specs for:
  - successful JSON parse and status update
  - invalid JSON
  - subprocess non-zero exit
  - missing file
  - duplicate/stale lock path
  - timeout behavior
  - label mapping contract

## Suggested implementation order (fastest safe path)

1. Configurable paths/env + timeout
2. Explicit label contract (`class_id`/`is_ai`)
3. Retry/error metadata columns
4. Confidence/model metadata persistence
5. Media-type routing (image vs video)
6. Metrics and dashboards
7. Full test suite hardening

## Performance and scaling tradeoffs

- **Current approach (Python subprocess per file)** is simple and good for low volume, but process startup adds latency and limits throughput.
- **Higher scale option:** run a long-lived model inference service (HTTP/gRPC) and call it from Rails jobs.
  - Pros: lower per-request overhead, easier GPU scheduling, better autoscaling.
  - Cons: more infrastructure and service-to-service ops.
- **Pragmatic near-term path:** keep subprocess model, add timeout/retries/metrics first, then move to inference service when queue latency becomes an issue.


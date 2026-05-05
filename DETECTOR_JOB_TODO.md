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

## 1) Make Python runtime and model path configurable — **DONE (2026-04-23)**

`DetectorJob` now freezes four paths/values at class-load time, each with an ENV override and a production-safe default:

| Constant | ENV override | Default |
|---|---|---|
| `PYTHON_BIN` | `CLASSIFIER_PYTHON` | `model/.venv/bin/python` |
| `CLASSIFY_SCRIPT` | `CLASSIFIER_SCRIPT` | `model/classify.py` |
| `CHECKPOINT_PATH` | `CLASSIFIER_CHECKPOINT` | `model/artifacts/best_real_fake_20260422_002356_seed42.pt` (Phase G6 winner, see `model/docs/MODEL_ABLATION_PLAN.md`) |
| `DEVICE` | `CLASSIFIER_DEVICE` | `cpu` |

The job now passes `--checkpoint` explicitly to `classify.py`, so the deployed model **no longer depends on which training run last overwrote `best_real_fake.pt`**.

### Tradeoffs
- **Pinned stamp vs "latest" symlink.** Production pins the exact stamped file so training runs cannot silently reshuffle the deployed model. The cost is a manual change to update production when a new winner is promoted (edit this file + bump `DEFAULT_CHECKPOINT_FILENAME` or set `CLASSIFIER_CHECKPOINT`). Preferred over the symlink-style default because "you trained something last night" should not rotate production.
- **Class-load freezing.** Constants are evaluated once per Rails process. Changing ENV requires a worker restart; this matches Rails idioms and avoids per-request ENV lookups.

### Remaining follow-up
- Nothing blocking; revisit the default when the next Phase (H, or a newer G-class winner) is promoted.

## 2) Add timeout and subprocess safety controls — **DONE (implementation: `CLASSIFIER_TIMEOUT_SEC`)**

`Open3.capture3` is wrapped in `Timeout.timeout`, configured by `CLASSIFIER_TIMEOUT_SEC` (see `DetectorJob.classifier_timeout_sec`). On timeout or non-zero exit the record stays `unknown` and errors are logged without noisy full-path / full-stderr dumps.

### Remaining follow-up
- Optionally `Process.kill`/`Open3` process group kill for hard hung native code (Ruby `Timeout` does not always stop the child process).

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

The API and UI are **image-only** for the MVP (JPEG/PNG/WebP/GIF). Video and other types are rejected at upload with 415.

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

1. ~~Configurable paths/env + timeout~~ (`CLASSIFIER_*`, `CLASSIFIER_TIMEOUT_SEC`)
2. Explicit label contract (`class_id`/`is_ai`)
3. Retry/error metadata columns
4. Confidence/model metadata persistence
5. Video pipeline for supported formats (beyond current image-only uploads)
6. Metrics and dashboards
7. Extend job/request/integration tests (beyond current harness)

## Performance and scaling tradeoffs

- **Current approach (Python subprocess per file)** is simple and good for low volume, but process startup adds latency and limits throughput.
- **Higher scale option:** run a long-lived model inference service (HTTP/gRPC) and call it from Rails jobs.
  - Pros: lower per-request overhead, easier GPU scheduling, better autoscaling.
  - Cons: more infrastructure and service-to-service ops.
- **Pragmatic near-term path:** keep subprocess model, add timeout/retries/metrics first, then move to inference service when queue latency becomes an issue.


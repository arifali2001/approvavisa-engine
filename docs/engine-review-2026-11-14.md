# Engine review: quality and performance (14 Nov 2026)

What I checked:
- the code at `ea090e6`
- a stage-level benchmark on Arif's PC (12 threads, Python 3.11, the repo's `.venv`)
- one measured resize test

The VPS (4 vCPU, 3 workers) will be slower than these numbers.

## Measured
Input: `approvavisa/public/images/showcase-eu-raw.jpg` upscaled to 11 MP (a typical phone photo), Spain DNI,
`editing_mode=true` (the sites' flow).

| Call | Total | rembg + refine | validate ×2 | face.analyze | crown.detect |
|---|---|---|---|---|---|
| Preview `/v1/validate` | **5.2 s** | 3.2 s (62 %) | 1.4 s | ×6 | ×5 |
| Download `/v1/process` | **5.2 s** | 3.2 s | 1.2 s | ×5 | ×4 |

At 1 MP the download is 2.4 s, of which rembg is 1.8 s. So model inference is ~1.5 s, and **~1.3 s per call is
spent pushing 11 MP arrays around** (the mask resize, the alpha refine at 0.5 s, the background analysis at 0.9 s).

Resize test (a zone plate shrunk 3.5×, like 11 MP → a 600 dpi 35×45 photo). This is leftover detail above the
output's resolution limit, which shows up as aliasing:

| Method | Aliasing |
|---|---|
| `cv2.INTER_LANCZOS4` (current) | **73.6** |
| `cv2.INTER_AREA` | 6.3 |
| Pillow `LANCZOS` | 0.4 |

## Findings, worst first

### Image quality
1. **The downscale aliases (measured above).** `photo_processor.py:261` and `:310` shrink a 3–5× larger crop with
   OpenCV's Lanczos4. That kernel doesn't widen for big reductions. Expect jagged hair edges against the white,
   moiré on shirts and patterned clothing, and "crunchy" skin. **Fix:** Pillow `LANCZOS` (or `INTER_AREA`) for every
   downscale. It's one line each, and the biggest visible win.
2. **Colour spill on the hair edges.** Defringing is switched off (`background.py:247`, `bgr_clean = bgr`). The
   comment is right that un-premultiplying rembg's straight colours made dark halos. But without any foreground
   estimation, half-transparent hair pixels keep the **old wall colour**: a beige, green or grey tint around the
   hair on a white backdrop. This is the tell-tale "cut out" look. **Fix:** foreground colour estimation on the
   edge band only (`pymatting.estimate_foreground_ml`, already installed as a rembg dependency, or the faster
   "blur-fusion" method).
3. **The alpha clean-up deletes fine hair.** `MORPH_OPEN` with a 3×3 kernel on the alpha (`background.py:142`) removes
   1–2 px strands and flyaways, which gives "helmet hair". **Fix:** drop the open. Remove only small disconnected
   blobs away from the person (connected components).
4. **rembg runs on the whole frame.** IS-Net works at 1024 px, so on a full-body or wide photo the head gets only
   a few hundred model pixels, and the hair edge is upscaled from a coarse mask. **Fix:**
   1. Find the face first (cheap).
   2. Cut a region of interest around head and shoulders with a margin.
   3. Run rembg on that region.

   This gives sharper hair edges *and* less time (see performance, item 3).
5. **No colour management.** `decode_base64_image` ignores the ICC profile. iPhone photos are often Display P3, and
   reading them as sRGB gives dull, slightly wrong skin tones. The output also carries no profile. **Fix:** convert
   to sRGB with `PIL.ImageCms` on input, and embed sRGB on output.
6. **No gentle exposure or white-balance step.** A face lit by warm indoor light on a pure white backdrop looks
   pasted in. **Fix:** a capped, camera-level correction measured on the face and skin (exposure ±0.5 EV, white
   balance towards neutral). Keep it strictly global, with no local retouching, so it's no more than a camera does.
7. **The crown can come from the wrong place.** `crown_detector.py:124` takes the topmost person pixel in the whole
   image. A raised hand, a second person or a mis-segmented object above the head makes the head look taller, and
   the crop too small. **Fix:** search only in the columns of the face box (±60 % of its width).

### Performance and reliability
1. **The endpoints block the server.** `/validate`, `/process` and `/portrait` are `async def` but do 5 s of CPU work
   inline. While a photo is processing, that worker can't answer anything else, including `/v1/health`. With all 3
   workers busy, Docker's 10 s health check can fail and Coolify may restart the engine in the middle of a rush.
   **Fix:**
   1. Run the work in a thread with a per-worker lock (MediaPipe isn't thread-safe).
   2. Keep `/v1/health` instant.
   3. Return a fast "busy, retry" (HTTP 503) when a short queue is full, instead of piling up.
2. **The customer pays for a second full run.** The preview runs the whole pipeline, then the download runs
   **exactly the same work again** (rembg included), about 5 s more for no gain. It also means the paid file isn't
   guaranteed to be pixel-identical to the preview the customer approved. **Fix:** cache the prepared result by
   `sha256(image) + document + options` (in memory with a short TTL, or on disk/R2) and serve the download from it.
3. **Everything runs at full resolution.**
   - The output is ~830×1060 px (2.1 MP for the print copy at 600 dpi), but the alpha refine, the background
     analysis, segmentation and the composite all run on the 11 MP frame.
   - **Fix:** work on the head-and-shoulders region at about 2× the print size.
   - Estimate (not measured): −40–60 % time, and much less peak RAM (today ~1.6 GB per worker at peak).
4. **Duplicate work inside one request.**
   - face.analyze runs 5–6× and crown detection 4–5×.
   - The print sheet calls `process()` again from scratch (`photo_processor.py:275`).
   - The eye-level fallback reruns the whole pipeline, **rembg included** (`:236`).
   - **Fix:** compute the face, crown and crop once in source coordinates and derive both crops from them.
5. **CPU over-subscription.** Each onnxruntime session uses all cores by default. Three workers at once fight over
   4 vCPU. **Fix:** set `intra_op_num_threads` to about cores ÷ workers in `_get_session`.
6. **Smaller items:**
   - The API key is accepted in `?api_key=`, where it ends up in logs; make it header only.
   - The key compare isn't constant-time; use `secrets.compare_digest`.
   - Error responses include raw exception text.
   - There's no per-stage timing in the logs.
   - Base64 JSON adds about a third more upload size compared with a multipart upload.

### Model choice (to test, not assume)
`isnet-general-use` is a general object model. Portrait-specific models usually do better on hair:
- `birefnet-portrait` (MIT, in rembg, but slower on CPU)
- a real matting refinement step

**Test:**
1. Build a golden set of 40 real, hard photos: curly hair, dark hair on a light wall, glasses, hijab, beige walls,
   low light.
2. Run each candidate through the same pipeline.
3. Compare edge quality side by side and the time on the VPS.
4. Pick by evidence.

Avoid BRIA RMBG: its licence isn't free for commercial use.

## Suggested order
1. **Day 1, quick wins (low risk):**
   - resize fix (quality 1)
   - threadpool + lock + health (performance 1)
   - ONNX threads (performance 5)
   - preview→download cache (performance 2)

   Effect: no more blocked health checks, about half the CPU per sale, sharper photos.
2. **Days 2–3, quality core:**
   - region-of-interest pipeline (quality 4 + performance 3)
   - foreground estimation (quality 2)
   - keep hair strands (quality 3)
   - sRGB (quality 5)
   - crown limited to the face columns (quality 7)
   - single-pass geometry (performance 4)
3. **Day 4, evidence:**
   - golden set + a visual diff report
   - model A/B on the VPS
   - gentle exposure/white balance (quality 6), judged on the golden set
4. Every step keeps the existing 70+ tests green and adds a regression test with a real photo.

The benchmark script is `bench_engine.py` in the review session's scratchpad. Copy it into `scripts/` when this work
starts.

---

## Done on 14 Nov 2026 (same day)

Everything above except "Model choice" and the exposure/white-balance idea (quality 6) is implemented. Nothing
is committed or deployed yet: a push to `main` auto-deploys on Coolify.

| Item | Change | Where |
|---|---|---|
| Quality 1, resize aliasing | Pillow antialiased Lanczos for every downscale; a Gaussian prefilter before the portrait warp | `image_utils.resize_hq`, `photo_processor`, `portrait` |
| **Dark hair halo (new bug found)** | rembg's default cutout is premultiplied by *black*; the engine then composited it again, darkening every edge (alpha 0.5: 152 instead of 177). Only the mask is now taken from rembg, and colours come from the photo | `background.remove_background` |
| Quality 2, colour spill | Blur-fusion foreground colour estimation (Forte & Pitie 2021) on edge pixels only (alpha 3-252) | `background._estimate_foreground` |
| Quality 3, lost strands | The morphological open is replaced by removal of *detached* specks only | `background._drop_detached_specks` |
| Quality 4 + perf 3, working region | rembg, segmentation, refine and crop run on a head-and-shoulders region at >= 2x the output size. The camera matrix is mapped back to the upload. If the crop would touch an inner edge of the region, or anything fails there, the full frame is processed as before | `photo_processor._plan_roi`, `_process_uncached(_roi=...)` |
| Quality 5, colour management | Non-sRGB ICC profiles (e.g. Display P3) are converted to sRGB on input; untagged/sRGB input is untouched | `image_utils._to_srgb` |
| Quality 7, crown | The crown is searched only above the face (face box +-60 %) | `photo_processor` (crown block), `CrownDetectionResult.person_mask` |
| Perf 1, blocking | Production runs one web process + `WORKERS` job processes (`JOB_PROCESSES`). /v1/health stays instant; more than 5 waiting photos per process gives 503 + Retry-After; a crashed job process rebuilds the pool | `core/jobs.py`, `main.py`, `Dockerfile` |
| Perf 2, double work per sale | Disk cache keyed by pixels + spec + options: the paid download reuses the preview's photo (the same pixels the customer saw). It's on only with the job pool | `core/prepared_cache.py` |
| Perf 4, rembg rerun | The eye-level fallback reuses the replaced background | `_process_uncached(_isolated=...)` |
| Perf 5, threads | Tried cores / processes per job process: on the 4-core VPS that gave 1 thread and a 45 s preview, so each process now uses all cores again (as before); `ENGINE_THREADS` overrides | `jobs.cpu_threads_per_job` |
| Small | Constant-time API key compare; per-job timing log line (`job=... seconds=...`) | `deps.py`, `jobs.py` |

**Not done, on purpose:**
- **Exposure/white-balance correction.** Several authorities forbid digitally altered photos. It would also hide
  real retake reasons from the exposure check.
- **The model A/B.** It needs ~40 real, hard customer-like photos and a VPS run. BiRefNet-portrait is about 1 GB and
  slower on CPU, too heavy for the 8 GB box today.

### Verification
- **Tests:** 124 pass (114 existing + 10 new in `tests/test_quality_speed_2026_11.py`).
- **Before/after harness** (`scripts/regression_snapshot.py`):
  - 9 sample photos (2 also at 11 MP) × ES DNI / IT / US / GB passport × preview, download and plain validate
    = 132 calls.
  - Result: see the table below.
- **Production-mode smoke test** (2 job processes, local):
  - /v1/health stayed under 50 ms while 4 photos ran.
  - Preview → paid download: **5.2 s → 1.3 s**, and the downloaded file is identical to the preview's photo.
- **11 MP preview inline:** 5.2 s → 4.6 s on this PC. That's with the new edge-colour step, which costs ~0.4 s.

### Deploy notes
- `WORKERS` now means photo-job processes (the default is still 3). Uvicorn always runs one web process. RAM is about
  the same as before: 3 × ~1 GB, plus ~0.2 GB for the web process.
- The cache uses `/tmp/approvavisa-prepared` inside the container (≤ 1.5 GB, 6 h TTL). A restart clears it, which
  is harmless.
- After deploying, watch the Coolify logs for `Job pool ready: 3 processes` and the `job=... seconds=...` lines.

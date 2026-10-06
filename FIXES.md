# Overnight fixes: Drishti AI

Branch `overnight-fixes`. Nothing pushed.

## Secrets: read this first

- `.env` is **not tracked** and was never committed (checked `git log --all -- .env`). It holds a `SERPER_API_KEY` and stays local. If you ever pasted that key elsewhere, rotate it.
- The **original Django `SECRET_KEY` (`django-insecure-znsm...`) is in the public repo history** (commit 0e8e93a). It is a development key and nothing depends on it, but never reuse it if this is deployed. I did not rewrite history. The code now reads `DJANGO_SECRET_KEY` from the environment and refuses to start without one unless `DJANGO_DEBUG=1`.

## What was broken (audit)

1. The "98% accuracy" claim: no benchmark existed. Weights were hand-tuned.
2. `Organika/sdxl-detector` outputs labels `artificial`/`human`; the code looked for "fake"/"ai"/"generated" in the label, so this model could never vote AI.
3. Copy-move check `matchTemplate(image, image)` returns one number (the image against itself). The score was meaningless.
4. Frequency score was unbounded (could exceed 1), depended on image size, and dumped the whole spectrum into the result.
5. JPEG block-grid check had an unreachable `elif` (the 0.8 score could never happen).
6. ELA wrote `temp_ela_*.jpg` into the working directory (left in the repo root), normalised the difference by its maximum, and saved heatmaps to `media/` forever.
7. Detectors returned 0.0 on failure, indistinguishable from "clean". EXIF always returned 0.4 on stripped images.
8. EXIF "file modified before photo taken" compared against the server's upload time and could never fire.
9. NaN risk (kurtosis of flat images), missing `None` checks, `os` imported after use.
10. Decision engine: heartbeat (a video signal) used on still images, verdicts like "AI generated image" stated as fact, "20+ detectors" (there are 15), webcam weights lowered by invented factors.
11. Views: extension-only validation, no size limit, files kept forever, `DEBUG=True` hard-coded, `ALLOWED_HOSTS=[]`, file path kept in the session.
12. Video: the pulse (rPPG) check used the first 30 frames (about 1 second), so its 0.8 to 3 Hz band had two FFT bins. The blink counter counted "Haar cascade found no eyes" as a blink. NaN correlation on flat frames. Frame JPEGs written next to uploads.
13. Text: printed "We found exact matches of this text across the web!" whenever a search engine returned any result (it nearly always does), "Looks original!" otherwise, and claimed "100% private" while sending text to serper.dev.
14. Debug leftovers in the repo root (temp images, scratch scripts, print-only test scripts).
15. `import transformers` pipelines failed here because of a broken TensorFlow install; `USE_TF=0` is now set.

## What I changed

- `ddb9c3e` Removed debug files from the repo (moved to ignored `_leftovers/`, including untracked big files such as `final_test.txt`; nothing deleted).
- `2fdd4b0` Rewrote the 15 detectors (full resolution, in memory, finite scores in [0,1], `applicable` flag), classifier wrapper with correct labels, detector registry.
- `0a73e4f` Learned combiner, benchmark scripts, tests, rebuilt Django app and UI, video and text fixes.
- Final commit: README, FIXES, screenshots, requirements.
- Combiner: logistic regression per task. The explanation is exact: each check's contribution is its weight times its standardised score, shown in the report. Checks with no measurable signal on the calibration split get zero weight.
- App: temp-dir uploads with validation, nothing stored, env-based settings and `.env.example`, POST-only chat, a keyword helper that says it is one.
- UI: plain report layout, light/dark by system preference, one accent colour, verdict worded "Flagged for human review", per-check weight and contribution, ELA heatmap, error states, works at 390 px (page width equals viewport in the screenshots in `docs/screenshots/`; I did not capture the old neon UI as a "before").

## Results

- `pytest`: 39 passed.
- Task A (edited vs authentic, CASIA JPEG-origin held-out test, n=609): ROC AUC 0.83 (CI 0.79 to 0.86), accuracy 0.74, precision 0.57, recall 0.79. External: Columbia AUC 0.48 (chance), Coverage 0.72. On the unfiltered split the first model (AUC 0.80) was no better than a file-extension-and-size baseline (0.82), so the shipped model uses JPEG-origin images only.
- Task B (AI vs real, Tiny-GenImage held-out, n=1,750, JPEG q90 control): ROC AUC 0.90 (CI 0.88 to 0.91), accuracy 0.82, precision 0.84, recall 0.79. A format/size-only baseline gets 1.0, so this is an upper bound.
- ViT deepfake classifier AUC 0.47 (dropped). Organika Swin 0.62 (kept, small gain).
- ELA is inverted on CASIA (AUC 0.28). Eight checks carry no signal on Task A.
- Details and licences: README.md, `metrics/*.json`.

## Still not done / didn't work

- It does not generalise: chance on Columbia. Do not call this a working splice detector.
- Task B cannot separate "AI" from "different resolution and processing" in this dataset. No size-matched control fit in the download budget.
- No benchmark for video or text (deliberate).
- Face-based checks never had data that helped (CASIA, ImageNet and GenImage are not face-swap sets). Untested for face swaps.
- Not tested: phone photos, WhatsApp/Instagram re-uploads, WebP, newer generators (FLUX, SDXL-turbo, GPT-image).
- The CASIA/Columbia/Coverage mirror is a third-party Hugging Face repo with no licence field.
- The sample screenshot is a CASIA tampered image. The app flags it as possibly AI-generated (0.88) and not as edited (0.32). That is an honest example of the limits above.
- Old uploads in `media/` and `db.sqlite3` are still on disk (ignored by git); I did not delete them.

## Check yourself

- Run it with `DJANGO_DEBUG=1`, upload your own real and edited phone photos, and see whether the flags make sense. I could not test those.
- First run downloads the Hugging Face model.
- Video module on a real webcam clip (no webcam here). The old "webcam mode" no longer exists.
- Rotate the Serper key if it was shared. Decide whether the old Django key matters.
- Update the portfolio: remove "98% accuracy"; use the numbers above with their caveats.

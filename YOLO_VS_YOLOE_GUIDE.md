# YOLO vs. YOLOE — Which to Use, and How

`vision_engine.py` now supports two real detection engines side by side, selected
by config, not code. This guide is the practical "which one, and how" reference —
for the research behind *why* YOLOE exists and its real accuracy numbers, see
`FINE_TUNING_YOLO_GUIDE.md` (Section 2); this guide focuses on using it in this app.

## TL;DR

| | **YOLO** (`VISION_MODEL_TYPE=yolo`, default) | **YOLOE** (`VISION_MODEL_TYPE=yoloe`) |
|---|---|---|
| Detects | Fixed 80 COCO classes only | All 80 COCO classes **plus** a ~250-item curated home vocabulary (kitchen, living room, bedroom, bathroom, workspace) — ~297 classes by default, or a custom list via `VISION_CUSTOM_CLASSES` |
| Covers `person`/`keys`/`wallet`/`sunglasses`? | `person` only — not COCO classes, no version of YOLO fixes this | All of them, by default |
| Covers general home items (`mug`, `remote`, `charger`, `pillow`, ...)? | Only the handful that happen to be COCO classes (`cup`, `remote`, `book`, ...) | Broad coverage by design — see "Where the class list comes from" below |
| Accuracy | 40.9 mAP (YOLO26n, on its known 80 classes) | ~30 mAP (YOLOE-S, zero-shot, on a ~90-class benchmark) — a real, meaningful step down, and likely lower still on a ~297-class list (more candidates competing per detection) |
| Training/dataset work required | None (pretrained) | None (pretrained, zero-shot) |
| Default weights | `yolo26n.pt` | `yoloe-26n-seg.pt` |

**Use YOLO for anything already in COCO's 80 classes** — it's more accurate and
that's what it's built for. **Use YOLOE when you also need the broader class
coverage** (`keys`, `wallet`, `sunglasses`, or general home items YOLO structurally
can't detect) — accept the accuracy tradeoff in exchange for zero fine-tuning/dataset
work.

## Where the class list comes from

An open-vocabulary model only ever detects classes it was explicitly told about via
`set_classes()` — there's no implicit "detect everything" mode. Two real design
questions came up building `DEFAULT_CUSTOM_CLASSES`, both worth knowing:

**"Why not just use YOLOE's built-in prompt-free vocabulary and skip maintaining a
list at all?"** Investigated and rejected — verified directly against Ultralytics'
own benchmarks: the prompt-free checkpoints (`-pf` suffix) score dramatically lower
accuracy (nano: 16.6 mAP vs. ~30 mAP for prompted mode) while being *heavier*
per-frame (15.8B FLOPs vs. 6.0B at the same nano tier) — a worse tradeoff on both
axes for CPU deployment. Its built-in vocabulary (4,585 classes, borrowed from the
RAM++ tagging model) also doesn't include `wallet` at all.

**"So where did ~250 curated home items come from, if not prompt-free mode or
hand-brainstorming?"** From that same RAM++ tag list — used as *source material* for
`set_classes()`, not as the runtime vocabulary. Every candidate term was checked
against the real 4,585-entry list, keeping only genuine matches (or a verified
close equivalent already in the list — e.g. `bathtub` isn't a tag, but `bath` is),
then filtered down to physical objects plausibly found at home. Deliberately
excluded: wild animals, vehicles, outdoor scenery, buildings, professions, and
non-object tags that shouldn't be object-detection targets at all (`aerobics`,
`action film`, `argument` are tags in that source list — clearly not detectable
objects). A handful of genuinely common home items aren't in RAM++ at all
(`kettle`, `mop`, `bucket`, `doorbell`, `flashlight`, `wallet`) and were added
manually.

An earlier version of this defaulted to only 5 narrow classes and consequently
could not detect a person in frame at all, despite `person` being an ordinary, easy
COCO class — fixed by unioning in the full COCO list. If you set
`VISION_CUSTOM_CLASSES` yourself, remember it fully **replaces** the default rather
than adding to it.

## How switching actually works

Both engines share the exact same `scan_frame()` contract (`_UltralyticsScanMixin`
in `vision_engine.py`) — `app.py` doesn't know or care which one is active. Three
env vars control everything:

| Var | Applies to | Default | Purpose |
|---|---|---|---|
| `VISION_MODEL_TYPE` | Both | `yolo` | `yolo` or `yoloe` — which engine backs `VisionTracker` |
| `YOLO_MODEL_PATH` | Both | `yolo26n.pt` (YOLO) / `yoloe-26n-seg.pt` (YOLOE) | Weights file — same var name for both, interpreted per whichever engine is active |
| `VISION_CUSTOM_CLASSES` | YOLOE only | 80 COCO classes + ~250 curated home items (~297 total, see below) | Comma-separated class names passed to `set_classes()` — **replaces** the default entirely, doesn't add to it |

```bash
# Switch to YOLOE, keep default classes
VISION_MODEL_TYPE=yoloe

# Switch to YOLOE with your own class list
VISION_MODEL_TYPE=yoloe
VISION_CUSTOM_CLASSES=keys,wallet,sunglasses

# Back to YOLO (or just remove/comment out VISION_MODEL_TYPE — it's the default)
VISION_MODEL_TYPE=yolo
```

**Docker/Codespaces gotcha, already handled:** `docker-compose.yml`'s `app` service
now explicitly lists all three vars under `environment:` with `${VAR:-default}`
syntax. Without the `:-default` fallback, an unset `.env` var becomes an *empty
string* inside the container (not "absent"), which would silently override
`vision_engine.py`'s own Python-side defaults — already fixed, nothing to do here,
just worth knowing if you ever add a new vision-related env var.

## Testing both, side by side

Same bare-minimum Docker setup as `TESTING_LOCAL_YOLO.md` / `TESTING_GITHUB_CODESPACES.md`
(2 of 4 containers — `app` + `postgres`), just toggling one env var between runs:

**1. Test YOLO first** (the default — no `.env` changes needed):
```bash
docker compose up -d --build --no-deps app postgres
docker logs -f vision_assist_app   # wait for boot
```
Log in (`test_user`/`test_user`), take a photo with your phone/keys/wallet visible,
note the results. `keys`/`wallet` should **not** appear — confirms the class gap.

**2. Switch to YOLOE:**
```bash
cat >> .env << 'EOF'
VISION_MODEL_TYPE=yoloe
EOF
docker compose up -d --no-deps app   # recreates just the app container with new env — no rebuild needed
docker logs -f vision_assist_app     # confirm: "[INFO] YOLOEVisionEngine loaded ... with classes=[...]"
```
Take the **same photo** again. You should see the same everyday classes YOLO
reported (`Person`, `Cell phone`, etc.) plus, potentially, `keys`/`wallet`/`sunglasses`
if any are in frame. Compare confidence scores directly against what YOLO reported
in step 1 — expect YOLOE's numbers to run lower across the board, even on classes
both engines know — that's the accuracy tradeoff from the TL;DR table, observed
directly rather than just read about. If `person` or another obvious COCO class is
missing entirely (not just low-confidence, but never listed at all), check
`DEFAULT_CUSTOM_CLASSES` hasn't been narrowed by a `VISION_CUSTOM_CLASSES` override
somewhere — that class simply isn't in the active vocabulary if so.

**3. Confirm which engine is actually active** without digging through logs — the
camera panel's caption now shows it directly: `Engine: YOLOEVisionEngine · Confidence
threshold: 0.5`.

**4. Switch back:** remove/comment the `VISION_MODEL_TYPE` line from `.env`, then
`docker compose up -d --no-deps app` again.

Everything else — login, bare-minimum container set, cleanup commands — is
identical to `TESTING_LOCAL_YOLO.md`/`TESTING_GITHUB_CODESPACES.md`; this guide
only covers the part that's different (the engine switch itself).

## Why not just always use YOLOE?

Since YOLOE can detect *everything* YOLO can plus the custom classes, it's tempting
to make it the default. Don't, for two concrete reasons:

1. **Accuracy regresses on the classes that already work.** `phone`/`backpack`/`handbag`
   detection would get measurably worse (YOLO26n's 40.9 mAP vs. YOLOE's ~30 mAP
   zero-shot) for no benefit, since YOLO already handles those classes well.
2. **It's still not a substitute for fine-tuning if `keys`/`wallet`/`sunglasses`
   need to work *reliably*** — YOLOE is the fast, zero-effort option to prototype
   with; a properly fine-tuned model (see `FINE_TUNING_YOLO_GUIDE.md`) is the
   higher-accuracy path once/if YOLOE's zero-shot results prove insufficient for
   real use, not just a demo.

YOLOE's right role here is exactly what this guide's testing section does with it:
a fast way to *see* whether the class gap is solvable without fine-tuning at all,
before investing in datasets and GPU time.

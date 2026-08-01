# YOLO vs. YOLOE — Which to Use, and How

`vision_engine.py` now supports two real detection engines side by side, selected
by config, not code. This guide is the practical "which one, and how" reference —
for the research behind *why* YOLOE exists and its real accuracy numbers, see
`FINE_TUNING_YOLO_GUIDE.md` (Section 2); this guide focuses on using it in this app.

## TL;DR

| | **YOLO** (`VISION_MODEL_TYPE=yolo`, default) | **YOLOE** (`VISION_MODEL_TYPE=yoloe`) |
|---|---|---|
| Detects | Fixed 80 COCO classes only | Any class you name via `VISION_CUSTOM_CLASSES` |
| Covers `keys`/`wallet`/`sunglasses`? | ❌ Never — not COCO classes, no version fixes this | ✅ Yes — that's the whole point |
| Covers `phone`/`backpack`/`handbag`? | ✅ Yes, and more accurately | ✅ Yes, but less accurately than YOLO |
| Accuracy | 40.9 mAP (YOLO26n, on its known 80 classes) | ~30 mAP (YOLOE-S, zero-shot, on LVIS) — a real, meaningful step down |
| Training/dataset work required | None (pretrained) | None (pretrained, zero-shot) |
| Default weights | `yolo26n.pt` | `yoloe-26n-seg.pt` |

**Use YOLO for anything already in COCO's 80 classes** — it's more accurate and
that's what it's built for. **Use YOLOE only for the classes YOLO structurally
can't detect** (`keys`, `wallet`, `sunglasses`, or any other custom item) — accept
the accuracy tradeoff in exchange for zero fine-tuning/dataset work.

## How switching actually works

Both engines share the exact same `scan_frame()` contract (`_UltralyticsScanMixin`
in `vision_engine.py`) — `app.py` doesn't know or care which one is active. Three
env vars control everything:

| Var | Applies to | Default | Purpose |
|---|---|---|---|
| `VISION_MODEL_TYPE` | Both | `yolo` | `yolo` or `yoloe` — which engine backs `VisionTracker` |
| `YOLO_MODEL_PATH` | Both | `yolo26n.pt` (YOLO) / `yoloe-26n-seg.pt` (YOLOE) | Weights file — same var name for both, interpreted per whichever engine is active |
| `VISION_CUSTOM_CLASSES` | YOLOE only | `keys,phone,wallet,sunglasses,backpack` | Comma-separated class names passed to `set_classes()` |

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
Take the **same photo** again. This time `keys`/`wallet`/`sunglasses` can appear —
compare confidence scores directly against what YOLO reported for `phone`/`backpack`
in step 1. Expect YOLOE's confidence numbers to run lower across the board, even on
classes both engines know — that's the accuracy tradeoff from the TL;DR table,
observed directly rather than just read about.

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

# Closing VisionAssist's Class Gap: Zero-Shot, Fine-Tuning, and Everything Between

Milestone 4 task: **"Fine tune Yolo model"** (currently unassigned on the tracker).

**Revision note:** the first version of this guide went straight to "gather datasets
and fine-tune" without first checking whether something *avoids* that work entirely.
It doesn't — this version leads with a genuine zero-training option, found only after
that gap was flagged and researched properly.

Planning/reference document — the class gap (`keys`/`wallet`/`sunglasses`) is not
yet closed; this guide is the plan for whoever picks up that Milestone 4 task next.

---

## 1. The problem, precisely

Standard MS-COCO — the dataset every public YOLO checkpoint (v5 through the current
YOLO26) is pretrained on — has 80 fixed classes. Checking `FallbackVisionEngine`'s
mock pool against that list:

| Item | COCO class? |
|---|---|
| `backpack` | ✅ Yes |
| `handbag` | ✅ Yes (reported as "handbag", not "purse") |
| `phone` | ✅ Yes (reported as `cell phone`) |
| `keys` | ❌ **No** |
| `wallet` | ❌ **No** |
| `sunglasses` | ❌ **No** |

No version of YOLO — v5 through YOLO26 — fixes this by being "better." They're all
trained on the same 80-class COCO label set.

## 2. Try this first — zero-shot detection, no training at all

Before investing in dataset gathering and training, there's a real option that needs
**neither**: **YOLOE**, Ultralytics' open-vocabulary model. Pass it the exact class
names you want at inference time — no training, no labeled data, no GPU rental.
Verified directly against Ultralytics' own docs — the nano checkpoint (matching this
repo's CPU-only convention) is `yoloe-26n-seg.pt`:

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")           # nano — CPU-appropriate, confirmed real checkpoint name
model.set_classes(["keys", "wallet", "sunglasses"])   # set once after loading
results = model.predict("path/to/image.jpg")
```

**Note on the `-seg` suffix:** every publicly listed YOLOE checkpoint ships with a
segmentation head (`yoloe-26n-seg.pt` through `yoloe-26x-seg.pt`, plus `-11` and
`-v8` variants) — there's no detection-only build. This still returns bounding boxes
as part of its output (segmentation implies detection), which is actually a good
match for this repo's current `scan_frame()` contract — it already returns
`{"detections": [{"label", "confidence", "box"}], "annotated_frame": ...}`, so a
YOLOE integration would slot its own boxes/confidence into the same shape rather
than needing a new one. The extra mask output would just go unused. There are also
`-pf` ("prompt-free") variants that detect a large built-in vocabulary without
calling `set_classes()` at all — not the right fit here since we want three
*specific* class names, but worth knowing they exist.

**How it actually works:** YOLOE adds a text-prompt pathway on top of the YOLO
architecture — a CLIP-style text encoder maps your class names into the same
embedding space the model uses internally, so it can match "keys" or "wallet"
against visual concepts learned during its own (much larger, open-vocabulary)
pretraining, without ever having seen a COCO-style "keys" label. The open-vocabulary
machinery re-parameterizes into a regular closed-set YOLO at inference, so it costs
no extra runtime speed once `set_classes()` is called.

**Real accuracy numbers, verified:** YOLOE was directly compared against YOLO-World
(a similar, earlier open-vocabulary model, also Ultralytics-supported) — **YOLOE26-S
scores 29.9% mAP on LVIS**, +11.4 AP over YOLO-World-S at the same size, while
running 1.4x faster and using a third of the training resources. That confirms
YOLOE over YOLO-World for this use case, and gives a real ballpark: zero-shot
accuracy (~30% mAP, on the very benchmark that contains our missing classes) is
meaningfully below what fine-tuning would target (YOLO26n hits 40.9% mAP on COCO's
*known* classes) — the real, quantified version of "less accurate than fine-tuning,"
not just an assertion.

**Recommendation: test this first, on real photos, before doing anything else in
this guide.** If `keys`/`wallet`/`sunglasses` detection accuracy is good enough for
a capstone demo despite that gap, the rest of this document — dataset gathering,
class remapping, freeze-layer tuning, GPU rental — may not be needed at all. Only
fall through to fine-tuning (Section 4 onward) if YOLOE's zero-shot accuracy proves
insufficient on real testing.

**Version note — confirmed, not assumed:** YOLOE was added to the `ultralytics`
package in version `8.3.99` (verified directly via GitHub's commit history for the
YOLOE source path, dated 2025-03-30). This repo's `ultralytics==8.4.106` pin is far
newer, so YOLOE support is definitely already present — no version bump, no need to
double-check first.

**Code change if this works:** same `YOLO_MODEL_PATH`-style config pattern applies —
`YOLOVisionEngine` would need a small branch to instantiate `YOLOE` instead of `YOLO`
and call `set_classes()` once at load time, since YOLOE is a distinct (if closely
related) class in the `ultralytics` package. Worth a short spike before committing to
the fine-tuning path below.

## 3. Another zero-training option: ask a multimodal LLM directly

Worth naming explicitly, since it's genuinely viable for this app's specific
contract. GPT-4o, Gemini, and Claude can all analyze an image and answer "is a
wallet/keys/sunglasses visible?" directly — verified research finding: these models
are documented as **unreliable at returning precise pixel bounding boxes**, which is
usually the dealbreaker for using them as object detectors.

**Partial update:** this limitation used to not matter at all, since `scan_frame()`
originally returned bare labels. That's since changed — the camera panel now shows
the annotated frame with real bounding boxes and per-detection confidence, so a
multimodal-LLM path would lose that visual feature (these models can name what's in
frame but not reliably draw a box around it). The label-only prompt below still
works fine for the *detection* half of the job: *"Which of these items are visible:
phone, keys, wallet, sunglasses, backpack, handbag? Answer with just the matching
labels."* — just know you'd be trading the box overlay away if this replaced YOLO
entirely, rather than being used as a quick prototype check alongside it.

**Real tradeoffs, honestly:** costs money per API call (unlike everything else in
this guide), adds network latency, requires an internet connection and an API key —
which cuts against this project's local/offline-capable design (the entire reason
YOLO runs in-process rather than calling a cloud service). Reasonable as a fast
manual-prototype check (`"does GPT-4o even recognize my wallet in this photo?"`)
before investing in anything else here, but not a fit for the production path this
app is built around.

## 4. If neither zero-shot option is accurate enough: fine-tuning vs. embeddings

Milestone 4 also has a separate **"Using Yolo with embeddings (for customized
objects)"** task. Four tools now, not two — pick based on what's actually needed:

- **YOLOE zero-shot (Section 2):** fastest to try, no data/training, moderate
  accuracy ceiling (~30% mAP on LVIS, verified above).
- **Multimodal LLM (Section 3):** fastest of all to *prototype*, but costs money
  per call and needs internet — not a fit for this app's local-first design.
- **Fine-tuning (Sections 5–10 below):** teaches the model a new *generic category*
  — "wallets in general" — with real labeled examples. Highest accuracy ceiling for
  a shared class, at the cost of dataset + GPU time.
- **Embeddings/similarity search:** recognizes *a specific instance* — "Shubham's
  wallet, specifically" — via a detector (even YOLOE or a generic one) cropping
  candidate regions, then a CLIP-style embedding matching the crop against a
  per-user reference photo.

Not mutually exclusive: YOLOE could plausibly get working detection *today*,
fine-tuning could improve accuracy later if needed, and embeddings would then tell
*whose* wallet it is — layers, not competing choices.

## 5. Real, existing datasets — no manual labeling needed

Two source options, genuinely different in character:

### Option A: Roboflow Universe (community-contributed, per-class)

All found on [Roboflow Universe](https://universe.roboflow.com), downloadable
directly in YOLO format:

**Keys:**
- [Keysdetection by SHEHAB EMAD](https://universe.roboflow.com/shehab-emad-n2q9i/keysdetection) — 882 images, purpose-built for keys detection (not keyboard keys)
- [keys by loganqin](https://universe.roboflow.com/loganqin/keys-8royn) — smaller, supplementary

**Wallet:**
- [wallet by Mohamed Hussam](https://universe.roboflow.com/mohamed-hussam-cq81o/wallet-sn9n2) — 418 images
- [wallet by "valuable object detection"](https://universe.roboflow.com/valuable-object-detection/wallet-mjzrc) — 304 images
- ["lost product" by wallet](https://universe.roboflow.com/wallet/lost-product/dataset/1) — 492 images, smartphone+wallet pairs (nice domain match — lost-item photos, not studio shots)

**Sunglasses:**
- [Sunglasses by "Data"](https://universe.roboflow.com/data-lpsgu/sunglasses-ow62i) — reported **9,997 images**, CC BY 4.0 — by far the largest source for this class. *Verification note: a direct re-fetch of this page to confirm the count/license was blocked by Roboflow's anti-bot protection (HTTP 403) — this figure comes from search-result summaries, not a directly confirmed page load. Double-check the actual number and license on the page yourself before relying on it.*

**Bonus — multi-class lost-and-found datasets:**
- [Lost & Found by lab9](https://universe.roboflow.com/lab9-qh40y/lost-found-hdfto) — 79 images, 11 classes including `key`, `phone`, `glasses`
- [Lost and Found by Kst](https://universe.roboflow.com/kst-lo6da/lost-and-found-nlb4x) — 4,500 images, 6 classes including `Key`

### Option B: LVIS (single professionally-annotated source, all 3 classes at once)

Newly researched, and arguably the better starting point: **LVIS** (Large Vocabulary
Instance Segmentation, Facebook AI Research) has **1,203 classes over ~160K images**
built on top of COCO's own image set — and confirmed (directly from Ultralytics'
`lvis.yaml`, not a search summary) to include **all three** missing classes:
`604: key`, `1155: wallet/billfold` (combined class — not just "wallet"), and
`1034: sunglasses`. Ultralytics ships first-class support:

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="lvis.yaml", epochs=100, imgsz=640)   # auto-downloads the 20.7GB dataset
```

Advantages over hand-curating Roboflow datasets: one consistent, professional
annotation standard (not several different community contributors' conventions),
built-in `lvis.yaml` config, and no manual class-ID remapping across sources. The
tradeoff is a large download (20.7GB) and training against all 1,203 classes unless
you filter down to just the ones you need — worth scripting a subset extraction
(images/labels containing only `key`/`wallet`/`sunglasses`, plus a sample of the
original 80 for the forgetting-prevention step below) rather than training the full
long-tail dataset.

**Recommendation:** try LVIS's subset first for consistency; fall back to
Roboflow's per-class datasets (Option A) if LVIS's own `key`/`wallet`/`sunglasses`
image counts turn out too sparse once filtered (LVIS is long-tailed by design — rare
classes have fewer examples than common ones).

### Processing required before either option is training-ready

Downloading isn't the same as being ready to train. Real, concrete steps:

1. **Annotation quality — don't skip this.** Roboflow Universe hosts community
   contributions of wildly varying quality; documented, recurring issues include
   export/label mismatches after download, and negative bounding-box coordinates
   introduced by certain augmentation pipelines. **Before merging anything, open a
   random sample (~50–100 images) of each downloaded dataset in an annotation
   viewer and manually verify boxes actually align with the objects.** LVIS, being
   a professionally maintained research dataset, needs this check far less — a real
   point in its favor.

2. **Class imbalance.** The sunglasses dataset alone (9,997 images) dwarfs the keys
   (~900) and wallet (~400–500) sources by 10–20x. Left as-is, the fine-tuned model
   will likely be much better at sunglasses than at keys/wallet. Mitigate with
   either undersampling the sunglasses set down closer to the other two classes'
   size, or oversampling/augmenting (flips, crops, color jitter) the keys/wallet
   images to close the gap — don't just merge and train on the raw counts.

3. **Deduplication.** Community datasets sometimes include duplicate or near-
   duplicate images (the same photo re-uploaded, or slightly cropped variants).
   Worth a quick perceptual-hash pass (e.g. `imagehash` library) before merging,
   since duplicates that land in *both* train and val splits silently inflate
   validation accuracy without teaching the model anything new.

4. **Class ID remapping.** Each source dataset numbers its own classes
   independently (e.g. several might call their single class `0`). Before merging,
   remap every dataset's class IDs to one consistent scheme — e.g. `80 = keys`,
   `81 = wallet`, `82 = sunglasses` — in both each `data.yaml`'s `names` list and
   the first column of every label `.txt` file. The single most common mistake when
   combining datasets from different sources.

5. **Re-split train/val after merging, don't just concatenate each source's own
   split.** Each downloaded dataset already comes pre-split into train/val/test by
   its original author, using their own ratios. Simply concatenating four
   datasets' `train/` folders and four `val/` folders risks near-duplicate or
   augmented copies of the same source image landing on both sides of *your*
   merged split (common when a source dataset's own split already contains
   augmented variants of the same photo). Re-shuffle and re-split the *merged*
   pool yourself (e.g. 80/20) after combining, rather than trusting the
   individually-inherited splits to compose correctly.

**Downloading (either option):** on a Roboflow Universe dataset page → "Dataset" in
the left sidebar → **Download Dataset** → format **YOLOv8** (label format is
identical across v5–YOLO26) → gives `data.yaml` + `images/` + `labels/`, or via
Python: `roboflow.download_dataset(url, model_format="yolov8")`. For LVIS, Ultralytics
handles the download automatically on first `model.train(data="lvis.yaml", ...)` call.

**Merging multiple Roboflow downloads:** Roboflow's own **"Merge Datasets"** UI
feature (three-dot menu on a project), or programmatically:
```python
import supervision as sv
ds1 = sv.DetectionDataset.from_yolo(images_directory_path="keys/images", annotations_directory_path="keys/labels", data_yaml_path="keys/data.yaml")
ds2 = sv.DetectionDataset.from_yolo(images_directory_path="wallet/images", annotations_directory_path="wallet/labels", data_yaml_path="wallet/data.yaml")
merged = ds1.merge([ds2])
```

**Licensing:** check the license tab on each Roboflow dataset page before use — most
are CC BY 4.0 (free, requires attribution), a few public domain. LVIS is a standard
research dataset, check its own terms separately. Fine either way for a capstone
project; just don't skip checking.

## 6. Don't let the model forget the original 80 classes

Ultralytics' own fine-tuning guidance is explicit: *"Include examples of the
original classes in the training data alongside the new classes"* is the most
reliable way to prevent "catastrophic forgetting" — where a model fine-tuned only on
new classes gets *worse* at detecting classes it already knew (phone, backpack,
handbag). Freezing layers helps somewhat but is a weaker substitute.

Practical approach: pull a few hundred to ~1,000 COCO images (via `fiftyone`'s COCO
zoo downloader, or Ultralytics' bundled `coco128` sample as a minimal check) and fold
them into the merged training set with original labels intact.

## 7. Which model to fine-tune: YOLO26n

Ultralytics released **YOLO26** in January 2026 — now this repo's shipped default
(see `app/vision_engine/vision_engine.py`). Real numbers, nano tier:

| Model | Params | mAP@.5:.95 | CPU ONNX latency |
|---|---|---|---|
| YOLO11n *(previous default)* | 2.6M | 39.5 | 56.1 ms |
| **YOLO26n** *(current default)* | **2.4M** | **40.9** | **38.9 ms*** |

*\*40.9 mAP and "up to 43% faster CPU vs YOLO11n" are raw-source confirmed (Ultralytics'
`yolo26.md`). The specific 38.9ms figure comes from a JS-rendered benchmark table on
the live docs site that couldn't be re-fetched as raw HTML — consistent with the
confirmed claims (a ~31% reduction from YOLO11n's raw-confirmed 56.1ms, under the
confirmed 43% ceiling), but not independently re-verified at the same rigor.*

Smaller, more accurate, and ~30% faster on CPU — plus NMS-free. Same
AGPL-3.0/Enterprise licensing as every other Ultralytics version.

`app/requirements.txt` already pins `ultralytics==8.4.106`, confirmed to support
both YOLO26 and YOLOE (Section 2's version note) — no dependency bump needed for
either path in this guide.

## 8. Fine-tuning mechanics — layers, weights, settings

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # pretrained COCO weights — starting point, not from scratch
results = model.train(
    data="lostfound.yaml",
    epochs=100,
    imgsz=640,
    freeze=10,          # see freeze guidance below
    optimizer="AdamW",
    lr0=0.001,
    patience=15,        # early stopping if val mAP plateaus
)
```

**What `freeze` actually does** (Ultralytics' documented layer mapping):
- Layers **0–10**: backbone — general feature extraction (edges, textures, shapes).
  This is the knowledge transfer learning reuses; layer 10 is the final C2PSA block.
- Layers **11+**: neck and detection head — where class-specific learning happens.
- `freeze=10` → backbone stays fixed, neck+head train. `freeze=11` → whole backbone
  frozen. `freeze=None` → everything trains.

**Which to use, by scenario** (Ultralytics' own guidance):

| Your situation | `freeze` value |
|---|---|
| Large dataset, similar domain to COCO | `None` (train everything) |
| Small dataset, similar domain | `10` (protect the backbone) |
| Very small dataset | `23` (only the detection head trains) |
| Domain very different from COCO | `None` (backbone needs to adapt too) |

**For this project:** the merged dataset (keys + wallet + sunglasses + COCO-mixed)
will likely land in the low thousands of images — "small dataset, similar domain."
**Start with `freeze=10`.** Re-run with `freeze=None` if validation mAP on the 3 new
classes stays weak after ~50 epochs — safer here than usual since the dataset already
includes mixed-in COCO examples (Section 6).

**Learning rate & epochs:** fine-tuning converges faster than training from scratch.
`lr0=0.001` with `AdamW` is a stable starting point; rely on `patience=15` early
stopping rather than guessing an epoch count.

## 9. Dataset format required

Standard Ultralytics YOLO format (unchanged across every version, v5 through YOLO26):

```
lostfound/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── lostfound.yaml
```

Each label file is a `.txt` with the same basename as its image — one line per
object: `class_id x_center y_center width height` (all normalized 0–1).

`lostfound.yaml`:
```yaml
path: ./lostfound
train: images/train
val: images/val

nc: 83   # 80 original COCO classes + keys, wallet, sunglasses
names: ['person', 'bicycle', 'car', ..., 'toothbrush',   # full 80 COCO names, in order
        'keys', 'wallet', 'sunglasses']                   # appended as classes 80, 81, 82
```

## 10. Free GPU options — real numbers, verified

| Platform | GPU | Weekly quota | Session cap | Notes |
|---|---|---|---|---|
| **Kaggle Notebooks** | T4 or P100 | ~30 hrs/week | 12 hrs | More consistent/predictable than Colab — recommended primary option |
| **Google Colab (free)** | T4 (16GB VRAM) | ~15–30 hrs/week, dynamic | 12 hrs | No credit card needed, but GPU availability isn't guaranteed at peak times |
| **AWS EC2** | g4dn/g5/p-series | **None free** | — | Confirmed: AWS free tier has **no GPU instances at all**, any usage bills at standard on-demand rates |

**Recommendation: Kaggle first.** For a nano model on a merged dataset in the low
thousands of images, expect roughly 1–3 hours for 50–100 epochs on a single T4 —
comfortably within either free tier's caps.

## 11. Step-by-step

0. **Try YOLOE zero-shot first (Section 2), or a multimodal LLM prototype (Section 3).**
   If either is accurate enough on real test photos, stop here — no GPU, no dataset
   work needed.
1. Create a free Kaggle account → New Notebook → Settings → Accelerator → **GPU T4 x2** (or x1)
2. `pip install ultralytics roboflow supervision` (torch/CUDA are already preinstalled on Kaggle GPU notebooks)
3. Choose LVIS subset (Option B) or curate Roboflow datasets (Option A) — Section 5
4. Run the processing checklist from Section 5 (quality audit, imbalance, dedup, class-ID remap, re-split) — **do not skip this and go straight to training**
5. Pull ~500–1,000 COCO images with original labels and fold them in (Section 6)
6. Write the combined `lostfound.yaml` (Section 9)
7. Run the `model.train(...)` call from Section 8
8. Validate: `model.val()` — check per-class mAP specifically for `keys`, `wallet`, `sunglasses`, not just overall mAP
9. Best weights land at `runs/detect/train/weights/best.pt` — download this file back from Kaggle

## 12. Code changes needed in this repo — good news, there aren't many

Because `YOLOVisionEngine` already resolves its weights path via `YOLO_MODEL_PATH`
(this repo's model-agnostic architecture) and already returns per-detection
confidence/box data plus an annotated frame, **no changes to `vision_engine.py`'s
core logic are needed for the fine-tuning path.** `ultralytics.YOLO()` accepts a
local file path directly, so a custom fine-tuned checkpoint works exactly like a
public one — its boxes and confidence scores flow through the same display code
already built for the pretrained model, automatically. (The YOLOE zero-shot path,
Section 2, needs a small additional branch to instantiate `YOLOE` and call
`set_classes()` — noted there.)

What's actually needed for fine-tuning:
1. **Host the fine-tuned `.pt` file somewhere the container can reach** — a
   team/infra decision: Hugging Face Hub (free, versioned, simplest), Git LFS (keeps
   it in-repo, adds tooling overhead), or a private S3 bucket (most setup). Do
   **not** commit the raw `.pt` file via normal git — `.gitignore` already excludes
   `*.pt`, and multi-MB binaries bloat git history badly without LFS.
2. **Bake it into the Docker image** (mirroring how `yolo26n.pt` is pre-downloaded
   today in `app/Dockerfile`) — `COPY` it in at build time, or `RUN` a fetch against
   wherever it's hosted.
3. **Point `YOLO_MODEL_PATH` at it** — e.g.
   `YOLO_MODEL_PATH=/workspace/models/visionassist_finetuned_v1.pt` in
   `docker-compose.yml`'s `app` service environment. Zero change to
   `vision_engine.py` itself.
4. `FallbackVisionEngine.simulated_pool` needs no change — stays the degrade-path
   mock regardless of which real model is active.
5. Once this actually happens, update `CLAUDE.md`/`README.md` the same way PR #16
   did for the model switches — model choice, version pin, file location.

# Local YOLO Test — Bare Minimum Setup

How to test the "take a photo, detect my phone" scenario end to end, using the
smallest possible Docker footprint.

Model in use: **YOLO26n** (`app/vision_engine/vision_engine.py`'s `DEFAULT_LOCAL_WEIGHTS`).
The camera panel is enabled by default (`VISION_ENABLED = True` in `app/app.py`) —
no local edit needed to see it.

## Why only 2 of the 4 containers

`docker-compose.yml` defines `app`, `ollama`, `postgres`, `qdrant`. Only `app` and
`postgres` are needed for this test:
- Login and item registration require Postgres.
- Camera detection (`VisionTracker.scan_frame()`) runs in-process inside the `app`
  container — no Ollama or Qdrant involved at all.
- Ollama is only touched by voice/LLM features, which this test doesn't use.
- Qdrant isn't wired into any application code yet, regardless of this test.

Skipping `ollama` (large image) and `qdrant` keeps the footprint down.

## Credentials reference — what's set, and what's a placeholder

| Where | Value | Notes |
|---|---|---|
| `.env` → `OPENAI_API_KEY` | *(your existing key, untouched)* | Not needed for this test — vision-only, no voice |
| `.env` → `POSTGRES_USER` | `postgres` | Dev placeholder |
| `.env` → `POSTGRES_PASSWORD` | `devpassword123` | Dev placeholder — **local testing only, not secure** |
| `.env` → `POSTGRES_DB` | `vision_assist` | Matches `docker-compose.yml` |
| `secrets/postgres_password.txt` | `devpassword123` | Must byte-for-byte match `.env`'s `POSTGRES_PASSWORD` |
| `secrets/qdrant_api_key.txt` | `unused-for-this-test` | Never actually used — Qdrant isn't started here |
| App login | `test_user` / `test_user` (or `admin` / `AdminPassword123`) | Not something set manually — auto-seeded by `precheck_db.py` the first time the database is empty |

**Not covered by this setup:** Ollama/Qdrant containers aren't running at all. If you
later want voice/LLM features too, you'll need `docker-compose up -d ollama` and a
model pull (`docker exec -it vision_assist_llm_local ollama pull llama3` — several GB).

## 1. `.env`

Repo root. Keep any existing `OPENAI_API_KEY` — just add the Postgres vars if missing:
```bash
OPENAI_API_KEY=sk-...           # existing value, not needed for this test
POSTGRES_USER=postgres
POSTGRES_PASSWORD=devpassword123
POSTGRES_DB=vision_assist
```

## 2. Docker secrets (plaintext — not base64)

```bash
mkdir -p secrets
printf '%s' 'devpassword123' > secrets/postgres_password.txt   # must match .env exactly
printf '%s' 'unused-for-this-test' > secrets/qdrant_api_key.txt
```

**Gotcha:** the `postgres` container's real password is whatever's in
`secrets/postgres_password.txt` (read verbatim). The `app` container authenticates
using `.env`'s plain `POSTGRES_PASSWORD`. These must match byte-for-byte or you'll
hit `password authentication failed`. Don't base64-encode the secrets file — keep
both plaintext and identical.

## 3. Bring up only `app` + `postgres`

```bash
docker-compose up -d --build --no-deps app postgres
```

`--no-deps` is the key flag — without it, Compose auto-starts `ollama` and `qdrant`
too, since `app` lists them as `depends_on`.

First build downloads torch/opencv/ultralytics and pre-fetches `yolo26n.pt` —
expect several minutes.

## 4. Watch it boot

```bash
docker logs -f vision_assist_app
```

Wait for `[INFO] DB migrations completed successfully.` then Streamlit's
`You can now view your Streamlit app...`. Ctrl+C just stops following logs.

## 5. Log in

`http://localhost:8501` — skip registration, a test account is auto-seeded on first boot:
- Username: `test_user`
- Password: `test_user`

## 6. Take the photo

Right-hand "👁️ Live Camera Workspace" panel — a **"Detection engine" radio button**
(YOLO / YOLOE, defaults to YOLO) sits above the camera widget, letting you compare
both engines on the same photo without restarting anything (see
`YOLO_VS_YOLOE_GUIDE.md`); both are pre-warmed right after login so switching is
instant either way. Below that, the panel shows the active confidence threshold →
camera icon → hold your phone up clearly in frame → allow the browser camera
permission → snap it.

## 7. What to expect

- The photo redisplays as an **annotated image with bounding boxes drawn around each
  detected item** — this isn't the raw photo anymore. Boxes/labels/confidence are
  drawn by the app's own `_draw_detections()` helper (plain `cv2.rectangle`/
  `cv2.putText`), not `ultralytics.Results.plot()` — this guarantees what's drawn
  can never diverge from the text detections reported below the image.
- COCO's real class name is **"cell phone"**, not "phone" — expect
  `🎯 Detected on Feed: Cell phone (91%)` (confidence percentage per label).
- Your face/body will likely also register as **"person"** (a real COCO class) —
  expect something like `🎯 Detected on Feed: Person (78%), Cell phone (91%)`.
  That's correct, not a bug — detections are filtered by confidence only, not by
  relevant-item class.
- Each detected label is written to Postgres and shows up in the "📋 System Status
  Log" table at the bottom of the page.

## 8. Confirm it was real inference, not the mock

```bash
docker logs vision_assist_app | grep "YOLOVisionEngine"
```
Expect: `[INFO] YOLOVisionEngine loaded 'yolo26n.pt' (conf>=0.5).`
If instead you see `FallbackVisionEngine initialized`, real loading failed —
check the lines above that one for the actual error.

## Stopping, relaunching, and cleanup

Three levels, from lightest to heaviest:

**Pause (fastest resume, containers stay on disk):**
```bash
docker-compose stop app postgres
# ...later...
docker-compose start app postgres
```
Nothing is rebuilt or re-seeded — picks up exactly where you left off.

**Remove containers, keep data (most common — "I'm done for today"):**
```bash
docker-compose down
```
Stops and removes the `app`/`postgres` containers, but the named `postgres_storage`
volume survives — `test_user` and anything you registered are still there next time.
Relaunch with:
```bash
docker-compose up -d --no-deps app postgres
```
No `--build` needed unless you've changed `Dockerfile` or `requirements.txt` again —
Compose reuses the image already built.

**Full reset (wipe the database too — "start completely clean"):**
```bash
docker-compose down -v
```
Also deletes the `postgres_storage` volume — `test_user`/`admin` get re-seeded from
scratch on the next `up`. Use this if the DB gets into a weird state.

**Reclaim disk space entirely** (removes the built image too, not just containers):
```bash
docker-compose down -v
docker rmi visionassist_user-module-app
```
Next `docker-compose up --build` starts from scratch — full multi-minute rebuild.

**Check what's actually running at any point:**
```bash
docker ps                          # running containers
docker images | grep visionassist  # built image(s) and their size
docker system df                   # overall Docker disk usage
```

---

Cross-reference: same underlying test, run inside a GitHub Codespace instead of
locally, in `TESTING_GITHUB_CODESPACES.md`. For the class-coverage gap this test
will surface (`keys`/`wallet`/`sunglasses` won't be detected), see
`FINE_TUNING_YOLO_GUIDE.md`.

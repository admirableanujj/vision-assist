# Testing on GitHub Codespaces

Same "take a photo, detect my phone" test as `TESTING_LOCAL_YOLO.md`, but running
the whole stack in a GitHub Codespace instead of your Mac. This sidesteps
the local disk-space issue entirely (Codespaces gives you a fresh disk) and Docker
comes preinstalled — no `.devcontainer` config needed, confirmed this repo doesn't
have one, so the default Codespaces image (which ships Docker via docker-in-docker)
is what you'll get.

## Why this actually works for the camera test

The one thing worth understanding before starting: `st.camera_input()` captures video
**in your browser**, client-side — it's not the Codespace's remote machine that needs
a webcam. The Streamlit *server* runs in the cloud; the *browser tab* viewing it (on
your laptop) is what accesses your webcam and uploads the photo. Codespaces forwards
ports over **HTTPS automatically**, which is exactly what browsers require to grant
camera permissions at all (`getUserMedia` refuses on a plain, unencrypted `http://`
origin) — so this is actually more turnkey than a bare EC2 instance would be, which
would need you to set up TLS yourself for the same permission prompt to appear.

## Free tier — what to expect

GitHub Free personal accounts include Codespaces hours billed in **core-hours**
(roughly 60 hours/month on a 2-core machine, proportionally less on bigger machines —
exact current numbers are worth checking on your own GitHub billing page, since these
can change). Pick at least a 4-core/8GB machine type if offered — comfortable margin
for building torch/opencv/ultralytics — knowing it burns through the monthly
allowance faster than a 2-core box.

## 1. Create the codespace

Web UI (simplest, no extra auth needed):
1. Go to `https://github.com/admirableanujj/vision-assist`
2. Confirm the branch selector is on `main` (this is merged, no feature branch needed)
3. Green **Code** button → **Codespaces** tab → **Create codespace on main**
   (click the "..." / gear icon first if you want to pick a bigger machine type before creating)

CLI alternative — needs one extra scope grant first:
```bash
gh auth refresh -h github.com -s codespace   # interactive, run this yourself
gh codespace create --repo admirableanujj/vision-assist --branch main
```

## 2. Open a terminal, confirm Docker

Once the codespace finishes building (opens VS Code in browser automatically):
```bash
docker --version
docker compose version   # note: "docker compose" (plugin), not "docker-compose" — Codespaces images ship the newer plugin form
```
If `docker-compose` (hyphenated) isn't found, just use `docker compose` (space) for
every command below — same flags, same behavior.

## 3. `.env` and secrets — fresh in this environment

These are gitignored, so a new codespace starts without them — same setup as local:
```bash
cat >> .env << 'EOF'
OPENAI_API_KEY=sk-placeholder
POSTGRES_USER=postgres
POSTGRES_PASSWORD=devpassword123
POSTGRES_DB=vision_assist
EOF

mkdir -p secrets
printf '%s' 'devpassword123' > secrets/postgres_password.txt   # must match .env exactly
printf '%s' 'unused-for-this-test' > secrets/qdrant_api_key.txt
```

## 4. Bring up only `app` + `postgres`

```bash
docker compose up -d --build --no-deps app postgres
```
Same bare-minimum reasoning as local — Ollama/Qdrant aren't touched by this test.

## 5. Watch it boot

```bash
docker logs -f vision_assist_app
```
Wait for `[INFO] DB migrations completed successfully.` then Streamlit's boot message.

## 6. Open the forwarded port

Codespaces auto-detects port 8501 and shows a notification — click **Open in Browser**.
If you miss it: **Ports** tab (bottom panel in VS Code) → right-click `8501` → **Open in Browser**.
Leave port visibility as **Private** — since you're viewing it through your own
authenticated GitHub session, that's already secure; no need to make it public.

## 7. Log in and take the photo

- Username: `test_user`, password: `test_user` (auto-seeded — no need to register)
- Right panel, "👁️ Live Camera Workspace" (shows the active confidence threshold
  above the camera widget) → camera icon → allow the browser camera permission
  prompt (this is *your laptop's* camera, even though Streamlit runs remotely) →
  hold your phone in frame → snap it

The photo redisplays as an **annotated image with bounding boxes drawn around each
detected item** (rendered by Ultralytics, not the raw photo). Expect
`🎯 Detected on Feed: Cell phone (91%)` (and likely `Person (78%)` too, since your
face/body will be in frame — that's correct, not a bug; see `TESTING_LOCAL_YOLO.md`
for why). Percentages are the model's actual confidence per detection.

## 8. Confirm real inference, not the mock

```bash
docker logs vision_assist_app | grep "YOLOVisionEngine"
```
Expect: `[INFO] YOLOVisionEngine loaded 'yolo26n.pt' (conf>=0.5).`

## Stopping / deleting the codespace (don't burn free hours idly)

**Stop** (keeps everything — containers, disk state — resumes fast, no compute billed while stopped):
```bash
gh codespace stop   # or: github.com/codespaces → "..." → Stop
```
Codespaces also auto-stop after ~30 min of inactivity by default.

**Delete entirely** (once you're fully done — frees any lingering storage allowance):
```bash
gh codespace delete   # or: github.com/codespaces → "..." → Delete
```

Nothing here touches this repo's committed files — it's purely a disposable
environment for running the test. Cross-reference: same underlying test, run
locally instead, in `TESTING_LOCAL_YOLO.md`. For the class-coverage gap this test
will surface (`keys`/`wallet`/`sunglasses` won't be detected), see
`FINE_TUNING_YOLO_GUIDE.md`.

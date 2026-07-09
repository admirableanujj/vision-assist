# Task 3 — FastAPI Architecture

> **Owner:** Shubham | **Stack:** FastAPI · Uvicorn · JWT (python-jose) · bcrypt · SQLAlchemy · WebSockets

---

## Router Map

```mermaid
graph TD
    APP["🚀 FastAPI App\nbackend/main.py\nport 8000"]

    APP --> AUTH["/api/auth\nrouters/auth.py"]
    APP --> ITEMS["/api/items\nrouters/items.py"]
    APP --> ZONES["/api/zones\nrouters/zones.py"]
    APP --> CAMS["/api/cameras\nrouters/cameras.py"]
    APP --> ASST["/api/assistant\nrouters/assistant.py"]
    APP --> SCAN["/api/scan\nrouters/scan.py"]
    APP --> DET["/api/detections\nrouters/detections.py"]
    APP --> REM["/api/reminders\nrouters/reminders.py"]
    APP --> WS["/ws/alerts\nrouters/ws.py"]

    AUTH -->|"no JWT needed"| OPEN(["POST /register\nPOST /login"])
    ITEMS -->|"JWT required"| ITEM_EP(["GET /items\nPOST /items\nDELETE /items/{id}"])
    ZONES -->|"JWT required"| ZONE_EP(["GET /zones\nPOST /zones"])
    CAMS -->|"JWT required"| CAM_EP(["GET /cameras\nPOST /cameras\nDELETE /cameras/{id}"])
    ASST -->|"JWT required"| ASST_EP(["POST /assistant"])
    SCAN -->|"JWT required"| SCAN_EP(["POST /scan"])
    DET -->|"JWT required"| DET_EP(["GET /detections\nGET /detections/latest"])
    REM -->|"JWT required"| REM_EP(["GET /reminders\nPOST /reminders\nGET /reminders/due"])
    WS -->|"JWT in query param"| WS_EP(["WS /ws/alerts"])

    style APP fill:#2C3E50,color:#fff
    style AUTH fill:#E74C3C,color:#fff
    style ITEMS fill:#3498DB,color:#fff
    style ZONES fill:#3498DB,color:#fff
    style CAMS fill:#3498DB,color:#fff
    style ASST fill:#E67E22,color:#fff
    style SCAN fill:#9B59B6,color:#fff
    style DET fill:#1ABC9C,color:#fff
    style REM fill:#F39C12,color:#fff
    style WS fill:#27AE60,color:#fff
```

---

## Authentication Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as /api/auth
    participant DB as Database
    participant MW as JWT Middleware

    Note over C,DB: Registration
    C->>A: POST /register {email, password, full_name}
    A->>DB: SELECT user WHERE email=?
    alt email already taken
        A-->>C: 400 Email already registered
    else new user
        A->>DB: INSERT user (bcrypt hash password)
        A-->>C: 200 UserOut {id, email, full_name}
    end

    Note over C,DB: Login
    C->>A: POST /login {email, password}
    A->>DB: SELECT user WHERE email=?
    A->>A: bcrypt.verify(password, hash)
    alt invalid credentials
        A-->>C: 401 Incorrect email or password
    else valid
        A->>A: create_access_token() — JWT exp 24h
        A-->>C: 200 {access_token, token_type: "bearer"}
    end

    Note over C,MW: Any protected request
    C->>MW: GET /api/items\nAuthorization: Bearer <token>
    MW->>MW: decode + validate JWT
    alt invalid / expired
        MW-->>C: 401 Could not validate credentials
    else valid
        MW->>DB: load user by email
        MW-->>A: user object injected via Depends()
        A-->>C: 200 data
    end
```

---

## File Structure

```
backend/
├── main.py                  # app factory, router registration, CORS, startup
├── auth.py                  # JWT create/decode, bcrypt, get_current_user dependency
├── config.py                # Settings (SECRET_KEY, ALGORITHM, DB URL from .env)
├── models/                  # SQLAlchemy ORM (see Task 2)
├── schemas/                 # Pydantic request/response models
│   ├── user.py
│   ├── item.py
│   ├── zone.py
│   ├── camera.py
│   ├── alert.py
│   ├── assistant.py         # AssistantRequest, AssistantResponse
│   ├── detection.py
│   └── reminder.py
├── routers/
│   ├── auth.py
│   ├── items.py
│   ├── zones.py
│   ├── cameras.py
│   ├── assistant.py
│   ├── scan.py
│   ├── detections.py
│   ├── reminders.py
│   └── ws.py               # WebSocket hub
└── services/
    ├── assistant.py         # AssistantBrain (Task 1)
    ├── vision.py            # YOLO detect wrapper
    ├── camera_worker.py     # background frame loop
    └── alerts.py            # zone breach checker
```

---

## Key API Contracts

### POST /api/assistant
```
Request:  { "text": "where is my phone?" }
Response: {
    "reply":  "Your phone was last seen in the living room, 3 minutes ago.",
    "intent": "locate_item",
    "data":   { "item": "phone", "zone": "living room", "seconds_ago": 180 }
}
```

### POST /api/scan
```
Request:  multipart/form-data — image file (JPEG)
Response: {
    "detections": [
        { "object_class": "cell phone", "confidence": 0.91, "zone_name": "sofa area", "bbox": [120,80,240,200] }
    ]
}
```

### GET /api/reminders/due
```
Response: [{ "id": 3, "text": "Take medication", "remind_at": "2026-06-26T09:00:00" }]
Side effect: marks returned reminders done=true in same transaction
```

---

## WebSocket Alert Protocol

```mermaid
sequenceDiagram
    participant B as Browser
    participant WS as /ws/alerts
    participant CAM as Camera Worker
    participant HUB as WS Hub

    B->>WS: Connect ws://host/ws/alerts?token=<JWT>
    WS->>WS: validate JWT → register socket in HUB[owner_id]

    loop continuous detection
        CAM->>CAM: YOLO detects item outside home zone
        CAM->>HUB: broadcast(owner_id, alert_message)
        HUB->>B: push JSON alert {"message": "Keys left living room", "item": "keys"}
        B->>B: AlertBanner shows ⚠️ in real time
    end

    B->>WS: disconnect
    WS->>HUB: remove socket
```

---

## Environment Variables

```bash
# .env
SECRET_KEY=your-random-32-char-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440   # 24h
DATABASE_URL=sqlite:///./visionassist.db
OPENAI_API_KEY=sk-...              # for LLM fallback
CAMERA_SOURCE=0                    # 0=webcam, or rtsp:// URL
```

---

## Startup Command

```bash
# inside container
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# interactive API docs auto-generated at:
# http://localhost:8000/docs   (Swagger UI)
# http://localhost:8000/redoc  (ReDoc)
```

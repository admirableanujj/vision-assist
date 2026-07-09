# Integration Plan — Voice + Vision + Backend

> **Rubric:** Integration [15%] — Seamless integration of voice assistant, computer vision, and other components.

This doc covers how all subsystems wire together into a single working product. Each owner's component is a black box with a defined contract; this doc is the contract registry.

---

## End-to-End Data Flow

```mermaid
flowchart TD
    subgraph Browser ["① Browser — React/Vite SPA"]
        MIC["🎤 Mic\nWeb Speech API STT"]
        CAM["📷 Webcam\ncapture frame every 3s"]
        TTS["🔊 Speaker\nWeb Speech API TTS"]
        BANNER["⚠️ Alert Banner\nWebSocket listener"]
    end

    subgraph Backend ["② FastAPI — port 8000"]
        ASST_R["POST /api/assistant"]
        SCAN_R["POST /api/scan"]
        WS_R["WS /ws/alerts"]
    end

    subgraph Services ["③ AI Services"]
        BRAIN["AssistantBrain\nLLM intent → DB lookup"]
        YOLO["YOLO Vision\nultralytics detect()"]
        CAM_W["Camera Worker\nbg thread — optional"]
    end

    subgraph DB ["④ SQLite / PostgreSQL"]
        DET_T["Detection table"]
        ITEM_T["Item table"]
        ALERT_T["Alert table"]
    end

    MIC -->|"transcript text + JWT"| ASST_R
    ASST_R --> BRAIN
    BRAIN -->|"SELECT latest detection"| DET_T
    BRAIN -->|"reply text"| TTS

    CAM -->|"JPEG frame + JWT"| SCAN_R
    SCAN_R --> YOLO
    YOLO -->|"detections[]"| DET_T
    YOLO -->|"check zone breach"| ALERT_T
    ALERT_T -->|"broadcast"| WS_R
    WS_R --> BANNER

    CAM_W -->|"frames (local)"| YOLO

    style Browser fill:#7B68EE,color:#fff
    style Backend fill:#3498DB,color:#fff
    style Services fill:#E67E22,color:#fff
    style DB fill:#27AE60,color:#fff
```

---

## Component Contracts

### STT → AssistantBrain (Anuj → Shubham)

```
POST /api/assistant
Authorization: Bearer <JWT>
Content-Type: application/json

Body:   { "text": "<transcribed string from Web Speech API>" }
Reply:  { "reply": "...", "intent": "locate_item", "data": {...} }
```

- Anuj's STT module calls this endpoint with whatever text the browser produces.
- Shubham's brain handles everything from that point.
- No shared state — purely HTTP.

---

### YOLO → Detection DB (Sanjay → Shubham)

```
POST /api/scan
Authorization: Bearer <JWT>
Content-Type: multipart/form-data

Body:   image=<JPEG bytes>
Reply:  { "detections": [ { "object_class", "confidence", "zone_name", "bbox" } ] }
```

- The browser (or Sanjay's camera worker) sends a frame.
- Shubham's YOLO wrapper detects, assigns zones, writes to Detection table, checks zone breaches.
- Sanjay's camera worker calls this same endpoint — no code change needed on either side.

---

### Alert → WebSocket → Browser (Shubham → Frontend)

```
WS ws://localhost:8000/ws/alerts?token=<JWT>

Server pushes: { "type": "alert", "message": "Keys left living room", "item": "keys", "ts": "..." }
```

- On zone breach, Shubham's alerts service calls `hub.broadcast(owner_id, payload)`.
- Browser receives it on the open socket and shows the alert banner.
- No polling needed.

---

## Integration Sequence — The "Where is my phone?" Golden Path

```mermaid
sequenceDiagram
    actor U as User
    participant BR as Browser
    participant API as FastAPI
    participant AB as AssistantBrain
    participant YV as YOLO Vision
    participant DB as Database

    Note over BR,YV: Background — Camera Worker running every 3s
    BR->>API: POST /api/scan (JPEG frame)
    API->>YV: detect(frame, zones)
    YV->>DB: INSERT detection (object_class="cell phone", zone="living room", ts=now)

    Note over U,DB: User asks a question
    U->>BR: speaks "where is my phone?"
    BR->>BR: Web Speech API → "where is my phone?"
    BR->>API: POST /api/assistant {text: "where is my phone?"}
    API->>AB: handle(text, user)
    AB->>AB: LLM classifies → locate_item, item_name="phone"
    AB->>DB: SELECT item WHERE name~"phone" AND owner_id=user.id
    AB->>DB: SELECT detection WHERE object_class=item.object_class ORDER BY ts DESC LIMIT 1
    DB-->>AB: zone="living room", ts=30s ago
    AB-->>API: {reply: "Your phone was last seen in the living room, 30 seconds ago."}
    API-->>BR: 200 AssistantResponse
    BR->>BR: Web Speech TTS speaks reply
    U-->>U: hears the answer
```

---

## Integration Test Checklist

| # | Test | Expected | Owner |
|---|---|---|---|
| 1 | POST /api/assistant `{"text": "what time is it"}` with valid JWT | Returns current time in reply | Shubham |
| 2 | POST /api/scan with a photo containing a phone | Detection inserted in DB, `object_class=cell phone` | Shubham + Sanjay |
| 3 | POST /api/assistant `{"text": "where is my phone"}` after test 2 | Returns "living room" or wherever detected | Shubham |
| 4 | Item moves outside home zone → WebSocket push fires | Browser receives `{ type: "alert" }` within 1 scan cycle | Shubham + Anuj |
| 5 | STT transcript sent from browser voice capture → TTS speaks reply | Full round-trip voice in, voice out | All |
| 6 | All routes return 401 without JWT | 401 Unauthorized | Shubham |

---

## ChromaDB — Conversation Memory (Enhancement)

ChromaDB is already in the stack (`chromadb` in `requirements.txt`). We can use it to persist the last N conversation turns per user, enabling follow-up queries like:

```
User: "where is my phone?"
AI:   "Living room."
User: "when was it last there?"  ← needs memory of previous turn
```

```mermaid
flowchart LR
    TEXT["New user text"] --> CHROMA["ChromaDB\nquery last 5 turns\nfor this user"]
    CHROMA --> CONTEXT["Conversation context\n+ new text"]
    CONTEXT --> LLM["LLM classify_intent\nwith context"]
    LLM --> HANDLE["Handle intent"]
    HANDLE --> CHROMA2["ChromaDB\nstore this turn"]
```

This adds meaningful innovation value (15% rubric criterion) with minimal effort since ChromaDB is already installed.

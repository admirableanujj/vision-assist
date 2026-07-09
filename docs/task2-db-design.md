# Task 2 — DB Design

> **Owner:** Shubham | **Due:** 6/26/2026 | **Stack:** SQLAlchemy ORM · SQLite (dev) → PostgreSQL (prod)

---

## Entity-Relationship Diagram

```mermaid
erDiagram
    USER {
        int      id               PK
        string   email            UK
        string   hashed_password
        string   full_name
        datetime created_at
    }

    ITEM {
        int      id               PK
        int      owner_id         FK
        string   name
        string   object_class
        int      home_zone_id     FK
        string   description
        datetime created_at
    }

    CAMERA {
        int      id               PK
        int      owner_id         FK
        string   name
        string   source
        string   location
        datetime created_at
    }

    ZONE {
        int      id               PK
        int      camera_id        FK
        string   name
        int      x1
        int      y1
        int      x2
        int      y2
    }

    DETECTION {
        int      id               PK
        int      camera_id        FK
        string   object_class
        float    confidence
        string   zone_name
        string   bbox
        datetime timestamp
    }

    REMINDER {
        int      id               PK
        int      owner_id         FK
        string   text
        datetime remind_at
        bool     done
    }

    ALERT {
        int      id               PK
        int      owner_id         FK
        string   message
        string   item_name
        datetime created_at
        bool     read
    }

    USER    ||--o{ ITEM      : "owns"
    USER    ||--o{ CAMERA    : "owns"
    USER    ||--o{ REMINDER  : "has"
    USER    ||--o{ ALERT     : "receives"
    CAMERA  ||--o{ ZONE      : "defines"
    CAMERA  ||--o{ DETECTION : "captures"
    ITEM    }o--o| ZONE      : "home zone"
```

---

## Why These Design Decisions

| Decision | Rationale |
|---|---|
| `object_class` on `Item` (not a name) | YOLO returns class labels — item maps its friendly name to the exact class YOLO detects |
| `home_zone_id` nullable on `Item` | Items without a home zone still get tracked; alerts only fire when a home zone is set |
| `Detection` is append-only | Never update detections — query the latest one to answer "where is my X?" |
| `done=true` marks Reminder in same GET call | Ensures each reminder fires exactly once (no double-speak) |
| Separate `Camera` entity | Enables per-camera scoping — each user can have multiple cameras/rooms |
| `Alert.read` flag | Frontend can show unread count badge without deleting alerts |

---

## File Structure

```
backend/
└── models/
    ├── __init__.py       # exports Base, engine, SessionLocal, init_db
    ├── database.py       # SQLAlchemy engine + session factory
    ├── user.py
    ├── item.py
    ├── camera.py
    ├── zone.py
    ├── detection.py
    ├── reminder.py
    └── alert.py
```

---

## SQLite → PostgreSQL Switch

The only line that changes between dev and prod:

```python
# dev (default)
DATABASE_URL = "sqlite:///./visionassist.db"

# prod — set in .env
DATABASE_URL = "postgresql://user:pass@db:5432/visionassist"
```

SQLAlchemy handles everything else. No model changes required.

---

## Indexes to Add (Performance)

```python
# On Detection — the two most-queried columns
Index("ix_detection_object_class", Detection.object_class)
Index("ix_detection_timestamp",    Detection.timestamp)

# On Alert — for unread count queries
Index("ix_alert_owner_read", Alert.owner_id, Alert.read)
```

---

## DB Lifecycle Flow

```mermaid
sequenceDiagram
    participant App as FastAPI startup
    participant DB as SQLite / PostgreSQL
    participant YOLO as Camera Worker

    App->>DB: init_db() — CREATE TABLE IF NOT EXISTS
    App->>DB: INSERT user (register)
    App->>DB: INSERT item (user registers "my keys" → object_class="key")
    YOLO->>DB: INSERT detection (object_class="key", zone="living room", ts=now)
    App->>DB: SELECT latest detection WHERE object_class="key"
    App-->>App: reply "Your keys were last seen in the living room, 2 min ago"
```

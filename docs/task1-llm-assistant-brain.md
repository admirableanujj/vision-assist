# Task 1 — LLM-Powered Assistant Brain

> **Owner:** Shubham | **Due:** 6/26/2026 (after STT module by Anuj)
> **Stack:** OpenAI API (function calling) · LangChain (optional) · FastAPI · SQLAlchemy

---

## Why LLM — Not Just Keyword Matching

The original plan was keyword rules with an LLM fallback. We're going further: **use the LLM itself as the intent classifier** via OpenAI function calling (structured outputs).

| Approach | "Where's my phone?" | "Have you seen my charger?" | "Find the keys near the couch" |
|---|---|---|---|
| Keyword matching | ✅ matches "where" | ❌ misses | ❌ misses |
| LLM function calling | ✅ | ✅ | ✅ |

Function calling forces the model to return a structured JSON intent — no parsing, no regex, no edge cases.

---

## Architecture

```mermaid
flowchart TD
    INPUT(["🎤 Voice Input\n(from Anuj's STT module)"])
    TEXT["Transcribed Text\ne.g. 'where is my phone?'"]

    subgraph BRAIN ["AssistantBrain — services/assistant.py"]
        direction TB
        CLASSIFY["🧠 LLM Intent Classifier\nOpenAI function calling\nreturns structured JSON"]

        CLASSIFY --> TIME["⏰ time_query\nReturn current time"]
        CLASSIFY --> DATE["📅 date_query\nReturn today's date"]
        CLASSIFY --> LOCATE["📍 locate_item\nDB lookup → latest Detection"]
        CLASSIFY --> REMINDER["🔔 reminder_ack\nAcknowledge (UI handles creation)"]
        CLASSIFY --> QA["💬 general_qa\nOpenAI chat completion\nwith system context"]

        LOCATE --> DB[("SQLite / PostgreSQL\nDetection table")]
    end

    REPLY["📢 Structured Response\n{reply, intent, data}"]
    TTS(["🔊 Text-to-Speech\n(browser Web Speech API)"])

    INPUT --> TEXT --> BRAIN --> REPLY --> TTS

    style BRAIN fill:#FFF3E0,stroke:#E67E22
    style CLASSIFY fill:#E67E22,color:#fff
    style DB fill:#27AE60,color:#fff
```

---

## Intent Schema (OpenAI Function Calling)

The LLM is given one function to call: `classify_intent`. It must return one of these:

```json
{
  "name": "classify_intent",
  "parameters": {
    "type": "object",
    "properties": {
      "intent": {
        "type": "string",
        "enum": ["time_query", "date_query", "locate_item", "reminder_ack", "general_qa"]
      },
      "item_name": {
        "type": "string",
        "description": "The item to locate — only present when intent is locate_item"
      },
      "reminder_text": {
        "type": "string",
        "description": "What the user wants to be reminded about — only for reminder_ack"
      }
    },
    "required": ["intent"]
  }
}
```

The model is **forced** to call this function — it cannot output free text for classification. This gives deterministic, parseable intent every time.

---

## Intent Routing Flow

```mermaid
flowchart LR
    TEXT["User text"] --> LLM["LLM\nclassify_intent()"]

    LLM -->|intent=time_query| T["datetime.now()\n→ 'It is 3:42 PM'"]
    LLM -->|intent=date_query| D["datetime.today()\n→ 'Today is Thursday, June 26'"]
    LLM -->|intent=locate_item\nitem_name=phone| L["SELECT * FROM detection\nWHERE object_class = item.object_class\nORDER BY timestamp DESC\nLIMIT 1"]
    LLM -->|intent=reminder_ack| R["'Got it! Use the Reminders\npanel to set the time.'"]
    LLM -->|intent=general_qa| QA["Second LLM call\nchat completion\nwith assistant context"]

    L --> FOUND{"row found?"}
    FOUND -->|yes| LOC_R["'Your phone was last seen\nin the living room, 5 min ago'"]
    FOUND -->|no| NOT_R["'I haven't spotted your phone yet.\nMake sure it's visible to the camera.'"]

    style LLM fill:#E67E22,color:#fff
    style QA fill:#9B59B6,color:#fff
```

---

## Two-Stage LLM Usage

```mermaid
sequenceDiagram
    participant U as User
    participant A as /api/assistant
    participant B as AssistantBrain
    participant O as OpenAI API
    participant DB as Database

    U->>A: POST {text: "where is my phone?"}
    A->>B: handle(text, user)

    Note over B,O: Stage 1 — Intent Classification (fast, cheap)
    B->>O: ChatCompletion(tools=[classify_intent], tool_choice="required")
    O-->>B: tool_call {intent: "locate_item", item_name: "phone"}

    Note over B,DB: Local lookup — no second LLM call needed
    B->>DB: SELECT item WHERE name ILIKE "%phone%" AND owner_id=user.id
    B->>DB: SELECT detection WHERE object_class=item.object_class ORDER BY ts DESC LIMIT 1
    DB-->>B: {zone: "living room", timestamp: 5min ago}
    B-->>A: {reply: "Your phone was last seen in the living room, 5 minutes ago.", intent: "locate_item", data: {...}}
    A-->>U: 200 AssistantResponse

    Note over B,O: Stage 2 — Only for general_qa (more tokens, optional key)
    B->>O: ChatCompletion(messages=[system_ctx, user_msg])
    O-->>B: "The Eiffel Tower is 330 metres tall."
    B-->>A: {reply: "...", intent: "general_qa", data: {}}
```

---

## System Prompt Design

The system prompt for **general_qa** gives the LLM its persona and constraints:

```
You are VisionAssist, a helpful home AI assistant.
You help users locate misplaced belongings using computer vision.
You can answer general questions, tell the time/date, and locate registered items.

Current date/time: {datetime}
Registered items for this user: {item_list}

Be concise and friendly. Respond in 1-2 sentences maximum.
If the user asks you to locate an item, you have already handled it via a database lookup — do not guess locations.
```

---

## LLM Cost Optimisation

| Intent | LLM calls | Estimated tokens | Cost per query |
|---|---|---|---|
| time_query | 1 (classify only) | ~80 | ~$0.00004 |
| date_query | 1 (classify only) | ~80 | ~$0.00004 |
| locate_item | 1 (classify only) | ~100 | ~$0.00005 |
| reminder_ack | 1 (classify only) | ~80 | ~$0.00004 |
| general_qa | 2 (classify + chat) | ~500 | ~$0.00025 |

Using `gpt-4o-mini` keeps costs negligible. Only general Q&A hits the model twice.

---

## Graceful Degradation

```mermaid
flowchart TD
    KEY{"OPENAI_API_KEY\nset?"}
    KEY -->|yes| LLM_PATH["LLM classification\n+ general Q&A"]
    KEY -->|no| RULE_PATH["Keyword fallback\ncovers time / date / locate"]

    RULE_PATH --> RULE_QA{"general question?"}
    RULE_QA -->|yes| CANNED["Canned reply:\n'I can help locate items,\ntell the time, or set reminders.\nFor general questions, an\nOpenAI key is needed.'"]
    RULE_QA -->|no| RULE_HANDLE["Handle time / date / locate\nnormally"]

    style LLM_PATH fill:#27AE60,color:#fff
    style RULE_PATH fill:#E74C3C,color:#fff
    style CANNED fill:#95a5a6,color:#fff
```

The assistant is **fully functional** without an API key for the three core intents. General Q&A degrades gracefully to an informative canned message.

---

## Enhancements We Can Add

| Enhancement | Complexity | Value |
|---|---|---|
| **Conversation memory** — remember last 5 exchanges per user (ChromaDB already in stack) | Medium | User can say "find it again" |
| **Proactive suggestions** — if item unseen for >1hr, LLM drafts a reminder | Medium | Delight feature |
| **Multi-item query** — "where are my keys AND wallet?" | Low | Parse two item names from intent |
| **Confidence caveat** — if detection confidence <0.6, say "I think I saw it in..." | Low | Honesty / trust |
| **Switch to Claude** — `anthropic` SDK, `tool_use` instead of `function_calling` | Low | Better reasoning, same pattern |

# Semantic Change Detection Engine

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet%205.5-D97757)
![spaCy](https://img.shields.io/badge/NLP-spaCy%20%2B%20MiniLM-09A3D5)
![Tests](https://img.shields.io/badge/tests-32%20passed-brightgreen)

A REST API that compares two versions of a text and tells you **whether the meaning
changed**, **how** it changed and **why it matters**. It is built for policies, contracts,
terms of service and product documentation, where one word ("may" → "may not",
"anytime" → "within 7 days") can change someone's rights.

The engine is an **agentic pipeline**. Fast, local NLP finds the clauses whose wording
changed. Then a **Claude Sonnet 5.5** agent examines each changed clause with tools,
classifies the change, rates its severity and explains the risk. The result comes back as
structured JSON and as a Markdown report for human reviewers.

```text
POST /v1/detect
{ "text_a": "Users can cancel their subscription anytime.",
  "text_b": "Users can cancel their subscription within 7 days." }

→ { "change_detected": true, "change_type": "restriction_added", "severity": "high",
    "confidence": 0.95, "evidence": ["anytime → within 7 days"],
    "summary": "Cancellation changed from anytime to only within 7 days, which narrows users' rights." }
```

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [The Claude agent](#the-claude-agent)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Change types and severity](#change-types-and-severity)
- [Configuration](#configuration)
- [Testing](#testing)
- [Postman collection](#postman-collection)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)

## Features

- **Meaning, not wording.** It tells a paraphrase apart from a real change, and it
  catches small edits that reverse the meaning.
- **Clause-level analysis.** Each sentence is split into clauses, and every change is
  traced to the exact clause and phrase.
- **Handles restructured documents.** It still matches sentences that were moved, split
  in two or merged.
- **12 change categories and 3 severity levels**, including restriction added or removed,
  obligation strengthened or weakened, numeric change and meaning reversed.
- **Risk explanation.** Each change comes with who is affected and the legal, financial
  or operational risk.
- **Two outputs:** a fixed JSON verdict for systems and a Markdown report for people.
- **Cost-aware.** Only changed clauses go to Claude. If only case, punctuation or sentence
  order changed, the answer comes without any Claude call.
- **Robust agent loop.** Tool inputs use strict schemas and are validated in code, and
  the agent gets a chance to correct a bad call. Prompts are cached between turns, and the
  document text is treated as data, never as instructions.

## How it works

![Architecture of the Semantic Change Detection Engine](docs/architecture.svg)

| Stage | Module | What it does |
|---|---|---|
| 1. Clause segmentation | `segmentation.py` | spaCy `en_core_web_sm` splits each text into sentences and clauses. A coordinated clause with its own subject becomes a new unit, and each line (for example a bullet) is its own unit. |
| 2. Sentence embeddings | `embeddings.py` | `sentence-transformers/all-MiniLM-L6-v2` turns each unit into a normalized vector. The cosine similarity of two units is the dot product of their vectors. |
| 3. Alignment and clustering | `alignment.py` | Units with identical words match first. The Hungarian algorithm (SciPy `linear_sum_assignment`) pairs the remaining units by similarity. A pair below the 0.45 threshold becomes a deletion plus an insertion. An unmatched unit joins a neighbouring pair when that raises the similarity, which detects split and merged sentences. |
| 4. Claude evaluation agent | `agent.py`, `prompts.py` | Claude Sonnet 5.5 receives every unit whose words changed, inspects it with tools, records a finding per unit and submits a verdict for the whole document. |
| 5. Structured output | `models.py`, `report.py` | The API returns the verdict as JSON and renders a Markdown report with the risk of each change. |

After stage 3, every aligned unit has one status:

| Status | Meaning | Sent to Claude |
|---|---|---|
| `unchanged` | Same words at the same position (case and punctuation ignored). | No |
| `moved` | Same words at a different position. | No |
| `modified` | The words changed. | Yes |
| `inserted` | The unit exists only in text B. | Yes |
| `deleted` | The unit exists only in text A. | Yes |

### Why embeddings alone are not enough

Embeddings show which clauses *correspond*, but not whether the meaning changed. With
MiniLM, the pair whose meaning is reversed even scores **higher** than a true paraphrase:

| Text A | Text B | Cosine | Meaning |
|---|---|---|---|
| Users may share their credentials. | Users may not share their credentials. | 0.88 | Reversed |
| Users must log in to continue. | Login is required to continue. | 0.82 | Same |

No similarity threshold can separate these cases. That is why every changed clause goes
to the Claude agent, which gets the similarity only as a hint.

### When Claude is not called

If no clause changed its words, the engine answers on its own, instantly and for free:

- `no_change`: the texts differ only in case, spacing or punctuation.
- `paraphrase`: the same sentences appear in a different order.

## Tech stack

| Component | Technology | Cost | Where it runs |
|---|---|---|---|
| REST API | FastAPI + Uvicorn | Free, open source | Your machine |
| Clause segmentation | spaCy 3.8, `en_core_web_sm` | Free, open source (MIT) | Your machine |
| Sentence embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, open source (Apache 2.0) | Your machine (CPU) |
| Alignment | SciPy, NumPy | Free, open source | Your machine |
| Evaluation agent | **Claude Sonnet 5.5** via the Anthropic Python SDK | Paid per token | Anthropic API |

The embedding model is downloaded once from Hugging Face (about 90 MB) and then loaded
from the local cache. It needs no API key, and your texts never leave your machine for
stages 1 to 3. Only the changed clauses (with their neighbouring context) are sent to
Claude in stage 4.

**Typical cost.** With Claude Sonnet 5.5 ($2 per million input tokens, $10 per million
output tokens), the `/v1/analyze` example below used 3,359 input and 845 output tokens,
which is about **2 US cents** per request. Every response from `/v1/analyze` reports its
own token counts.

## The Claude agent

Stage 4 runs a tool-use loop on the Claude Messages API. Claude decides which tools to
call and when it is done. The code validates every tool input.

| Tool | Purpose |
|---|---|
| `inspect_unit` | Returns the word-level edits of one unit and the text before and after it, for example to resolve "this fee" or "they". |
| `record_finding` | Records the verdict for one changed unit: change type, severity, confidence, summary, evidence and risk. |
| `submit_verdict` | Records the verdict for the whole document and ends the loop. |

**Loop rules**

- An invalid tool input goes back to Claude as an error result, so Claude can correct the call.
- `submit_verdict` is rejected until every changed unit has a finding.
- If Claude replies with text only, the loop reminds it to use the tools.
- If Claude refuses or the reply is cut off, the loop stops before running any tool, and
  the API returns `502`.
- If there is no verdict within `SCD_AGENT_MAX_TURNS` turns (default 10), the API returns `502`.

**Request settings**

| Setting | Value |
|---|---|
| Model | `claude-sonnet-5-5` (Claude Sonnet 5.5), configurable with `SCD_LLM_MODEL` |
| Thinking | Adaptive (`thinking: {"type": "adaptive"}`) |
| Effort | `medium` (`output_config.effort`), configurable with `SCD_LLM_EFFORT` |
| Max output per turn | 16,000 tokens |
| Tool schemas | `strict: true`, so tool inputs always match their schemas |
| Tool choice | `auto`. Claude Sonnet 5.5 does not accept forced tool use. |
| Prompt caching | On. Later turns read the tools, system prompt and earlier turns from the cache. |
| Conversation | Append-only. Each assistant turn goes back to the API unchanged. |

The prompt marks the document texts as data and tells Claude never to follow
instructions inside them. The engine also escapes `</` in the texts, so a document cannot
close a prompt tag.

## Quick start

### Requirements

- **Python 3.11 or later.** Check with `python --version` (on Windows, `py --version`).
- **A Claude API key.** Create one in the [Claude Console](https://platform.claude.com).
- **About 1.5 GB of disk space.** The first install downloads PyTorch and the other
  dependencies, and the first start downloads the MiniLM model.

### 1. Clone the repository

```bash
git clone https://github.com/Nisarg13/semantic-change-engine.git
cd semantic-change-engine
```

### 2. Add your API key

Copy the example settings file:

```bash
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env`. Then open `.env` and set your key:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

> **Keep `.env` private.** It holds your API key. It is listed in `.gitignore`, so git
> will not commit it.

### 3. Install

macOS and Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

pip also installs the spaCy English model from its wheel URL. Create `.venv` on each
computer; a `.venv` folder copied from another computer does not work.

### 4. Run the tests (optional)

The tests use a scripted Claude client, so they need no API key and cost nothing.

```bash
.venv/bin/pytest            # Windows: .venv\Scripts\pytest
```

Expected result: `32 passed`.

### 5. Start the server

macOS and Linux:

```bash
.venv/bin/uvicorn semantic_change.api:create_app --factory --port 8000 --env-file .env
```

Windows (PowerShell):

```powershell
.venv\Scripts\uvicorn semantic_change.api:create_app --factory --port 8000 --env-file .env
```

The server loads the spaCy and MiniLM models at startup. When the terminal shows
`Application startup complete`, it is ready. Press `Ctrl+C` to stop it.

### 6. Try it

- **Interactive docs:** open <http://127.0.0.1:8000/docs>, pick an endpoint, then select
  **Try it out** → **Execute**.
- **curl** or **PowerShell:** see the examples in [API reference](#api-reference).
- **Postman:** import the collection, as described in [Postman collection](#postman-collection).

## API reference

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | Status, version and the configured model names. |
| `POST` | `/v1/detect` | The verdict only. |
| `POST` | `/v1/analyze` | The verdict, every changed unit with its finding, pipeline counts, the agent trace (turns, tool calls, tokens) and the Markdown report. |

Both `POST` endpoints take the same body:

```json
{
  "text_a": "Original text",
  "text_b": "Modified text"
}
```

### `POST /v1/detect`

curl (macOS and Linux):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"text_a": "Users can cancel their subscription anytime.",
       "text_b": "Users can cancel their subscription within 7 days."}'
```

PowerShell:

```powershell
$body = @{
  text_a = "Users can cancel their subscription anytime."
  text_b = "Users can cancel their subscription within 7 days."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/detect -ContentType "application/json" -Body $body
```

Response (a real run with `claude-sonnet-5-5`; Claude writes the summary, so the wording
can differ between runs):

```json
{
  "change_detected": true,
  "change_type": "restriction_added",
  "summary": "Cancellation changed from anytime to only within 7 days, which narrows users' rights.",
  "severity": "high",
  "confidence": 0.95,
  "evidence": ["anytime → within 7 days"]
}
```

A paraphrase is recognised as no change in meaning:

```jsonc
// text_a: "Users must log in to continue."
// text_b: "Login is required to continue."
{
  "change_detected": false,
  "change_type": "paraphrase",
  "summary": "The sentence was reworded, but the meaning is the same. Login is still required to continue.",
  "severity": "low",
  "confidence": 0.95,
  "evidence": []
}
```

### `POST /v1/analyze`

```bash
curl -s -X POST http://127.0.0.1:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text_a": "Refunds are available within 30 days of purchase. Shipping is free for all orders.",
       "text_b": "Shipping is free for all orders. Refunds are available within 14 days of purchase. A processing fee of $5 applies to all refunds."}'
```

Response from a real run with `claude-sonnet-5-5` (the second unit and the report are
shortened here):

```json
{
  "result": {
    "change_detected": true,
    "change_type": "numeric_change",
    "summary": "The refund window was cut from 30 to 14 days. A new $5 processing fee on all refunds was also added. Both changes make refunds less favorable to customers.",
    "severity": "high",
    "confidence": 0.85,
    "evidence": [
      "within 30 days → within 14 days",
      "added: A processing fee of $5 applies to all refunds."
    ]
  },
  "units": [
    {
      "id": 2,
      "status": "modified",
      "similarity": 0.931,
      "moved": true,
      "original": "Refunds are available within 30 days of purchase.",
      "modified": "Refunds are available within 14 days of purchase.",
      "finding": {
        "change_detected": true,
        "change_type": "numeric_change",
        "summary": "The refund window was shortened from 30 days to 14 days after purchase.",
        "severity": "high",
        "confidence": 0.95,
        "evidence": ["within 30 days → within 14 days"],
        "risk": "Customers lose refund eligibility for days 15 to 30 and may miss the new, shorter deadline and lose money."
      }
    },
    { "id": 3, "status": "inserted", "...": "..." }
  ],
  "pipeline": { "units_a": 2, "units_b": 3, "flagged_units": 2 },
  "agent": {
    "model": "claude-sonnet-5-5",
    "turns": 1,
    "tool_calls": 3,
    "input_tokens": 3359,
    "output_tokens": 845
  },
  "report": "# Semantic change report\n..."
}
```

| Field | Content |
|---|---|
| `result` | The verdict, the same as `/v1/detect`. |
| `units` | Every unit that is not `unchanged`: status, similarity, original and modified text, and Claude's finding with its `risk`. |
| `pipeline` | Unit counts of each text and the number of units sent to Claude. |
| `agent` | Model, turns, tool calls and token counts. `turns` is 0 when Claude was not called. |
| `report` | The Markdown report. |

The `report` field from the same run renders like this:

```markdown
# Semantic change report

| Field | Value |
|---|---|
| Meaning changed | **Yes** |
| Change type | `numeric_change` |
| Severity | **HIGH** |
| Confidence | 0.85 |

**Summary.** The refund window was cut from 30 to 14 days. A new $5 processing fee on
all refunds was also added. Both changes make refunds less favorable to customers.

## Changes (2)

### 1. Numeric change — **HIGH** (confidence 0.95)

- **Unit:** Modified, similarity 0.93, moved
- **Original:** “Refunds are available within 30 days of purchase.”
- **Modified:** “Refunds are available within 14 days of purchase.”
- **What changed:** The refund window was shortened from 30 days to 14 days after purchase.
- **Why it matters:** Customers lose refund eligibility for days 15 to 30 and may miss
  the new, shorter deadline and lose money.
- **Evidence:** `within 30 days → within 14 days`

### 2. Restriction added — **HIGH** (confidence 0.80)

- **Unit:** Added, similarity 0.00
- **Modified:** “A processing fee of $5 applies to all refunds.”
- **What changed:** A new $5 processing fee now applies to all refunds, reducing the
  amount customers get back.
- **Why it matters:** Customers receive less money than they paid. ...
```

### Errors

| Status | Cause |
|---|---|
| `422` | The body is not valid, or a text is longer than `SCD_MAX_TEXT_CHARS` characters. |
| `502` | The Claude agent failed: Claude refused, the reply was cut off, or the Claude API returned an error. The `detail` field gives the cause. |
| `503` | No valid Claude credentials. Set `ANTHROPIC_API_KEY`. |

Requests where no words changed still succeed without credentials, because they do not
call Claude.

## Change types and severity

| `change_type` | `change_detected` | Meaning | Example (A → B) |
|---|---|---|---|
| `no_change` | `false` | Identical apart from case, spacing or punctuation. | "Users can cancel anytime." → "users can cancel anytime" |
| `paraphrase` | `false` | Different words, same meaning. | "Users must log in to continue." → "Login is required to continue." |
| `meaning_reversed` | `true` | The statement now says the opposite. | "may share" → "may not share" |
| `restriction_added` | `true` | A new limit: time window, condition, cap, fee or narrower scope. | "cancel anytime" → "cancel within 7 days" |
| `restriction_removed` | `true` | A limit is removed or loosened. | "within 14 days" → "at any time" |
| `obligation_strengthened` | `true` | A duty becomes stronger or new. | "may attend" → "must attend" |
| `obligation_weakened` | `true` | A duty becomes weaker or applies in fewer cases. | "will notify" → "may notify" |
| `numeric_change` | `true` | An amount, percentage, count, duration, rate or date changes. | "$10" → "$15" |
| `entity_change` | `true` | A party, product or place is replaced. | "administrators" → "managers" |
| `content_change` | `true` | The meaning changes in another way. | "view the reports" → "edit the reports" |
| `information_added` | `true` | New detail without a restriction or duty effect. | "on iOS" → "on iOS and Android" |
| `information_removed` | `true` | A detail or clause is removed. | "a charger and a case" → "a charger" |

| `severity` | Rule |
|---|---|
| `high` | Affects rights, duties, permissions, money or deadlines, or reverses the meaning. |
| `medium` | Affects facts, parties, scope or details with moderate impact. |
| `low` | No effect or a minor effect on the meaning. |

## Configuration

All settings come from environment variables, normally set in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Your Claude API key (required for requests where words changed). |
| `SCD_LLM_MODEL` | `claude-sonnet-5-5` | The Claude model of the agent. |
| `SCD_LLM_EFFORT` | `medium` | Thinking depth: `low`, `medium`, `high`, `xhigh` or `max`. |
| `SCD_LLM_MAX_TOKENS` | `16000` | Output token limit for each agent turn. |
| `SCD_AGENT_MAX_TURNS` | `10` | Maximum agent turns per request. |
| `SCD_SPACY_MODEL` | `en_core_web_sm` | spaCy pipeline for stage 1. |
| `SCD_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for stage 2. |
| `SCD_ALIGN_THRESHOLD` | `0.45` | Minimum cosine similarity to pair two units. |
| `SCD_MAX_TEXT_CHARS` | `20000` | Maximum length of each input text. |

To use a different Claude model, set it in `.env`, for example
`SCD_LLM_MODEL=claude-opus-5-5` for the most demanding documents.

## Testing

```bash
.venv/bin/pytest
```

| File | Scope |
|---|---|
| `tests/test_stages.py` | Stages 1 to 3 with the real spaCy and MiniLM models. |
| `tests/test_agent.py` | The agent loop and the API request settings, with a scripted Claude client. |
| `tests/test_api.py` | The REST API end to end: real stages 1 to 3 and a scripted Claude client. |

The tests make no calls to the Claude API, so they need no API key.

Static checks (lint, format, type check):

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
```

## Postman collection

`postman/semantic-change-engine.postman_collection.json` contains 7 requests, each with
tests.

1. In Postman, select **Import** and choose the collection file.
2. Start the server on port 8000. For another host or port, change the collection
   variable `baseUrl`.
3. Send a single request, or run them all with **Run collection**.

| Request | Calls Claude |
|---|---|
| Health | No |
| Detect - example 1: restriction added | Yes |
| Detect - example 2: paraphrase | Yes |
| Detect - meaning reversed | Yes |
| Analyze - policy with several changes | Yes |
| Detect - no change (no Claude call) | No |
| Detect - invalid request (422) | No |

Postman can also import the OpenAPI schema from <http://127.0.0.1:8000/openapi.json>.

## Troubleshooting

| Problem | Solution |
|---|---|
| `python3: command not found`, or Python older than 3.11 | Install Python 3.11+ from <https://www.python.org/downloads/>. |
| `address already in use` | Another program uses port 8000. Start with `--port 8001` and use that port in your requests. |
| `503` with `No Claude credentials` | The server did not get the key. Start it with `--env-file .env` and check that `.env` contains `ANTHROPIC_API_KEY`. |
| `503` with `Claude rejected ANTHROPIC_API_KEY` | The key is wrong or was deleted. Create a new key in the Claude Console and update `.env`. |
| `502` | The Claude API returned an error, for example the credit is used up. The `detail` field gives the cause. |
| The first start is slow | The first start downloads the MiniLM model. Later starts use the cached copy. |
| Warnings about Hugging Face symlinks or unauthenticated requests (Windows) | Safe to ignore. The model still downloads and caches correctly; no Hugging Face token is needed. |

## Project structure

```text
semantic-change-engine/
├── .env.example          Settings template (copy it to .env)
├── pyproject.toml        Package metadata, dependencies and tool config
├── docs/
│   └── architecture.svg  Architecture diagram used in this README
├── postman/
│   └── semantic-change-engine.postman_collection.json
├── src/semantic_change/
│   ├── api.py            REST API (FastAPI)
│   ├── pipeline.py       Runs the stages in sequence
│   ├── segmentation.py   Stage 1: clause segmentation (spaCy)
│   ├── embeddings.py     Stage 2: sentence vectors (sentence-transformers)
│   ├── alignment.py      Stage 3: Hungarian alignment and clustering (SciPy)
│   ├── agent.py          Stage 4: Claude Sonnet 5.5 agent and its tools
│   ├── prompts.py        System prompt and task message of the agent
│   ├── report.py         Stage 5: Markdown report
│   ├── models.py         Request and response schemas (pydantic)
│   ├── taxonomy.py       Change types and severity levels
│   └── config.py         Settings from SCD_* environment variables
└── tests/                Tests for the stages, the agent loop and the REST API
```

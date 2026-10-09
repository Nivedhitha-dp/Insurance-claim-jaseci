# Claims Desk: an AI agent for total-loss vehicle claims

Claims Desk reads a total-loss claim folder (PDFs, crooked scans, customer emails) and pulls out the
key facts. It checks the documents against each other, decides whether the claim is `complete`,
`incomplete` or `needs_review`, and talks to the customer about what's still missing.

## Contents

1. [Setup](#1-setup)
2. [Example output](#2-example-output)
3. [Approach and architecture](#3-approach-and-architecture)
4. [Tool design rationale](#4-tool-design-rationale)
5. [How the agent chooses tools](#5-how-the-agent-chooses-tools)
6. [Key decisions and trade-offs](#6-key-decisions-and-trade-offs)
7. [What we'd do with more time](#7-what-wed-do-with-more-time)

---

## 1. Setup

### Requirements

- **Python 3.10+**
- **Node.js 18+**, only if you want to change the web UI. A ready-built copy is included in
  `frontend/dist/`, so running the UI needs Python only.
- **No GPU.** OCR (RapidOCR) runs on the CPU, and its models are installed with the Python package.
- **An LLM API key (optional):** a free key from [Google AI Studio](https://aistudio.google.com/apikey)
  (Gemini) and/or [Groq](https://console.groq.com/keys). Without a key, the system runs on its
  rule-based offline planner.

### Step 1: Install

From the project folder (`take-home-ai-engineer/`):

```bash
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1          # Windows PowerShell
.venv\Scripts\activate.bat          # Windows cmd
source .venv/bin/activate           # macOS / Linux

pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Step 2: Configure

Copy the example settings file:

```bash
cp .env.example .env                # Windows cmd: copy .env.example .env
```

Then open `.env` and paste your key(s):

- **`GEMINI_API_KEY`** and/or **`GROQ_API_KEY`**. The default chain is Gemini first, then Groq.
- **Only a Groq key?** Set `LLM_PROVIDER=groq`.
- **No key at all?** Set `LLM_PROVIDER=none` to run fully offline with the rule-based planner.

`.env` is git-ignored, so your keys are never committed.

### Step 3: Run the tests (offline, about 2 seconds)

```bash
pytest -q
```

The tests always use the offline planner on a temporary copy of the claims, so they need no API key
and don't touch `outputs/`.

### Step 4: Process the claims

```bash
python -m claims_agent run              # all 5 claims -> outputs/CLM-00X.json + outputs/prioritization.json
python -m claims_agent run CLM-003      # a single claim (doesn't rebuild prioritization.json)
python -m claims_agent run -q           # same, without printing each tool call
```

This overwrites the files in `outputs/`. The first run is slower: OCR takes about 5 seconds per
scanned page, and results are cached in `.cache/` for later runs. With free-tier API keys, a full
run takes several minutes because requests are paced to stay within rate limits. Each claim's output
shows which model, or the offline planner, handled each turn (`driven_by`).

### Step 5: Talk to the agent as the customer (terminal)

```bash
python -m claims_agent chat CLM-002            # continue the saved session for this claim
python -m claims_agent chat CLM-002 --fresh    # start the claim over
```

Type a reply as the customer. Use `/attach path/to/file.pdf` to upload a document, and `/quit` to
exit. Sessions are saved in `.state/`.

### Step 6: Web UI

The UI is already built and included in the repository, so one command starts it:

```bash
python -m claims_agent serve            # then open http://127.0.0.1:8000
```

Pick a claim to see its documents, extracted fields, issues and the agent's message. Process it
again to watch each tool call live, reply as the customer, or upload a document. `serve` also accepts
`--host` and `--port`.

**Changing the UI (needs Node.js 18+):**

```bash
cd frontend
npm install
npm run dev                             # live-reload UI at http://localhost:5173, keep `serve` running
npm run build                           # rebuild frontend/dist when you're done
```

### Optional settings (`.env`)

| Setting | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` / `LLM_FALLBACK` | `gemini` / see `.env.example` | Primary provider and the fallback chain (`gemini`, `groq`, `openrouter`, `anthropic`, `openai`, `ollama`, `none`) |
| `STRICT_SETTLEMENT_MATH` | `false` | Settlement arithmetic errors block finalization |
| `WARNINGS_BLOCK_FINALIZE` | `false` | Any reviewer warning blocks finalization |
| `MAX_AGENT_STEPS` | `24` | Maximum tool-calling steps per turn |
| `TURN_DEADLINE_S` | `240` | Time limit per turn, in seconds, before the offline planner finishes it |

---

## 2. Example output

Structured output for all five claims is in [`outputs/`](outputs/): one JSON file per claim and a
ranked queue in [`prioritization.json`](outputs/prioritization.json). Each claim's file also lists
every tool the agent called (`tools_used`) and which model handled each turn (`driven_by`).

| Claim | Status | Next action | What the agent found |
|---|---|---|---|
| [CLM-001](outputs/CLM-001.json) | `complete` | finalize | All required documents present; all four fields agree (high confidence). Non-blocking warnings: the settlement doesn't subtract the deductible, and the adjuster's note mentions possible prior damage. |
| [CLM-002](outputs/CLM-002.json) | `incomplete` | message customer | Police report missing. The customer's reply promises it but doesn't include it, so the claim stays incomplete and the agent asks again. |
| [CLM-003](outputs/CLM-003.json) | `needs_review` | escalate | VIN conflict: the finance agreement says `…034182`, the police report and settlement say `…034127` (best guess). `settlement_breakdown_v2.pdf` replaces the earlier version, so the payout is $23,800. |
| [CLM-004](outputs/CLM-004.json) | `needs_review` | escalate | VIN `5YJ3E1EA7K` is only 10 characters in every document, and the insurance payout is still "TBD". |
| [CLM-005](outputs/CLM-005.json) | `needs_review` | escalate | Finance agreement missing, and the date of loss conflicts between documents (best guess: 2026-03-22). The loan balance comes from the settlement instead, at medium confidence. |

**Processing order:** CLM-001 → CLM-003 → CLM-002 → CLM-005 → CLM-004
(ready to finalize first; blocked on missing or invalid data last).

---

## 3. Approach and architecture

### How the problem is broken down

A total-loss claim is handled in eight steps. Each step has a clear owner: **plain code** where the
answer must be exact, and the **AI agent** where it needs judgement or language.

| Step | What happens | Who does it |
|---|---|---|
| 1. Document intake | List the claim's files, fingerprint them, and get the text out: PDF text directly, OCR for scans (with per-line confidence and tilt correction) | Code |
| 2. Classification | Decide what each document is from its *content*, not its filename. Code suggests a type from keywords; the agent makes the call | Agent |
| 3. Extraction | Pull out VIN, date of loss, insurance payout and loan balance using the form's layout, keeping the source file and confidence. Free text (handwritten notes, emails) is read by the agent, which must quote the exact words | Code (forms), Agent (free text) |
| 4. Validation | VIN is 17 valid characters, dates are real, amounts are numbers. Only unambiguous OCR slips are repaired | Code |
| 5. Cross-document consistency | Compare every value across documents. Flag disagreements, say which documents disagree, and give a best guess based on which document is the authority for that field. Spot duplicates and newer versions | Code |
| 6. Decision | Fixed rules set the status (`complete`, `incomplete`, `needs_review`) and the next action. The agent cannot override it | Code |
| 7. Customer communication | Write a clear message: what's missing, why, and answers to the customer's questions. A final check rejects messages that skip a missing document, ignore a question, or promise money | Agent writes, code checks |
| 8. Prioritization | Rank all claims so the team knows what to work on first | Code |

### Architecture

```
                   ┌──────────────────────────┐
  Claim folder ───►│        AI agent          │
  Customer reply   │  (Gemini → Groq → rule-  │
  or upload        │   based backup planner)  │
                   └───────┬──────────▲───────┘
                 picks a   │          │ results
                 tool      ▼          │ + hints
                   ┌──────────────────┴───────┐
                   │          Tools           │  read · extract · validate · compare ·
                   │  (thin wrappers around   │  reconcile · evaluate · submit
                   │   plain Python logic)    │
                   └───────┬──────────────────┘
                           │ reads / writes
                           ▼
                   ┌──────────────────────────┐
                   │       Claim state        │  documents, evidence for every value,
                   │  (one saved JSON/claim)  │  issues, status, conversation, tool log
                   └───────┬──────────────────┘
                           ▼
              Claim output JSON  +  prioritization.json
```

- **The agent** decides what to do next. It never touches facts directly.
- **The tools** are the only way to act. Each one wraps plain Python logic: OCR, layout extraction,
  validators, reconciliation, rules.
- **The claim state** is the single source of truth. Everything learned about a claim is stored
  there, with where it came from, not in the AI's conversation.

### How the agent chooses its next tool

The agent works in a loop: look at what it knows, call the tools it needs, read the results, repeat.
The turn ends when `submit_turn` accepts it. The agent isn't given a fixed order; it's given goals
and a sense of when each tool is worth using, and the evidence decides:

- A scan read with low confidence or missing fields → re-read it with enhanced OCR; if that still
  fails, ask the vision model.
- Two documents of the same type → compare them (duplicate or newer version?).
- A field seen in more than one document → reconcile it.
- A settlement with line items → check its arithmetic.
- A handwritten note or an email → read it and record what it says, with a quote.
- Nothing left to check → evaluate the claim, then submit.

Tools also send back hints (for example *"fields missing on a scan: try enhanced OCR"*), so each
result steers the next step. [Section 5](#5-how-the-agent-chooses-tools) covers this in
detail, with examples from the actual runs.

### Multi-turn: how customer replies update a claim

1. **First turn:** the agent reviews only the documents the claim arrived with. Customer replies in
   the folder are held back.
2. **Customer turn:** a reply (the sample reply file, a chat message, or a web UI message) is saved
   to the claim as a new document. Uploaded files are added as new documents too.
3. **The agent picks up where it left off.** It starts with a short summary of the claim's current
   state, not the whole history, then processes only what's new:
   - facts in the message (*"the date was the 22nd"*) are recorded with a quote, as weaker evidence
     that can back up or contradict the documents;
   - intentions are recorded: a promise to send a document, a document they can't find, a question;
   - a new upload is read, classified, extracted and compared like any other document.
4. **Everything is re-evaluated.** Reconciliation and the rules run again over the full, updated
   state, so the status can change.

A promise is not a document. If the customer says *"I'll send the police report next weekend"*, the
claim stays `incomplete` until the report actually arrives.

### Output and prioritization

**Per claim** (`outputs/CLM-00X.json`), the claim state is written out as:

- `status` and `status_reason`
- `extracted_fields`: value, confidence, source document and reason for each of the four fields
- `documents`: identified, missing, duplicates, superseded
- `issues`: each problem with its type, the documents involved and a suggested value
- `cross_document_checks`: vehicle, lender, names and policy numbers across documents
- `next_action`: finalize / message customer / escalate, the customer message, and an internal note
- `conversation`: what happened in each turn
- `tools_used`: every tool call, in order

**Across all claims** (`outputs/prioritization.json`), claims are ranked by how close each one is
to a payout that can be trusted:

| Tier | Meaning |
|---|---|
| 1 | Complete: finalize now |
| 2 | All documents in; only internal issues with a clear best-guess fix |
| 3 | Waiting on a document the customer has promised |
| 4 | Needs review and a document the customer can't easily get |
| 5 | Blocked on unknowns on our side, such as a payout still "TBD", plus invalid data |

Within a tier: fewer blocking issues first, then higher confidence in the decision, then the oldest
loss. Customer messages are sent straight away regardless of rank, because waiting on the customer
is the slowest part. For the five sample claims the order is **CLM-001 → CLM-003 → CLM-002 → CLM-005
→ CLM-004**.

---

## 4. Tool Design Rationale

The agent uses small, focused Python tools rather than one tool that handles the entire claim. Each
tool has a defined responsibility, returns structured results, and can be called when the agent
determines it is needed. This keeps document processing, evidence collection, validation, and
decision-making separate and traceable.

### 3.1 Getting at the Documents

#### `list_documents`

**What it does:** Lists the files available for a claim and computes a fingerprint for each file.

**Why it exists separately:** Document discovery and duplicate detection should happen before
extraction, without coupling file management to the extraction logic.

**When the agent calls it:** At the start of a claim and when new documents may have arrived, such as
after a customer upload.

**What it returns and how the agent uses it:** Each file's type, size, fingerprint, whether it has
already been read and typed, and the turn it arrived in, plus groups of byte-for-byte duplicates.
The agent uses this to decide which documents still need processing.

#### `read_document(file, mode)`

**What it does:** Reads a document using its PDF text layer or OCR for scanned pages. OCR reports
line-level confidence and handles slight page rotation. An enhanced mode attempts to recover text
from poor-quality scans.

**Why it exists separately:** Reading a document is distinct from interpreting its contents. Keeping
it separate also allows the agent to choose a more expensive reading method only when needed.

**When the agent calls it:** When a document has not yet been read, or when missing fields or low OCR
confidence justify enhanced OCR.

**What it returns and how the agent uses it:** Extracted text, OCR confidence information, and a
suggested document type. When OCR quality is mediocre, it also returns a hint to try enhanced OCR or
vision. The agent uses the text for extraction and classification, but does not treat the suggested
type as authoritative.

#### `vision_read(file, question)`

**What it does:** Uses a vision model to answer a specific question about a document image.

**Why it exists separately:** Visual interpretation is useful for handwriting and ambiguous marks,
but can be slower and less predictable than ordinary text extraction.

**When the agent calls it:** As a last resort when ordinary and enhanced OCR cannot reliably recover a
required value.

**What it returns and how the agent uses it:** A visual interpretation and transcript that can be
checked later. Any value recorded from this result must still be supported by verifiable evidence.

### 3.2 Turning Text into Facts

#### `extract_fields(file, doc_type)`

**What it does:** Records the document type chosen by the agent and extracts the VIN, date of loss,
insurance payout, and outstanding loan balance using document-specific layout rules.

**Why it exists separately:** Known forms have predictable layouts, so deterministic extraction is
easier to validate and trace than asking an LLM to interpret every field.

**When the agent calls it:** Once for every document, after identifying its type from its contents.
For types without a form layout (adjuster notes, customer messages, unknown documents), it extracts
nothing and tells the agent to read the text itself and use `record_observation`.

**What it returns and how the agent uses it:** Field values with source locations, extraction methods,
and confidence information. It also reports expected fields that were not found (with a hint to try
enhanced OCR on a scan) and a VIN in the narrative that differs from the labelled VIN. The agent uses
the values as candidate facts for validation and reconciliation.

#### `record_observation(file, field, value, quote)`

**What it does:** Records a field value interpreted from free text, such as an adjuster's note or
customer email.

**Why it exists separately:** Free text requires interpretation, unlike the fixed layouts handled by
`extract_fields`. Separating the two also makes the evidence requirements explicit.

**When the agent calls it:** When a relevant value appears in a note, email, or other content that
cannot be handled by the form extraction rules.

**What it returns and how the agent uses it:** A recorded observation if the supporting quote is
valid; otherwise, an error. The exact quote must be found in the source document or an accepted
vision transcript, and the quote must actually state the value being recorded. This prevents
unsupported LLM-generated values from becoming facts. Values from customer messages are stored as
low-authority evidence that can corroborate or contradict documents but never replace one.

#### `flag_concern(...)`

**What it does:** Records a potentially important issue that does not fit the four standard fields.

**Why it exists separately:** A claim can contain relevant risks that cannot be represented by
extracting a VIN, date, payout, or loan balance alone.

**When the agent calls it:** When a document raises a concern requiring reviewer attention, such as an
adjuster's note questioning prior vehicle damage.

**What it returns and how the agent uses it:** A recorded concern, with a supporting quote that is
verified against the document when one is given. The agent includes it in the review findings rather
than silently ignoring it or treating it as a verified conclusion. Concerns are warnings by default.

### 3.3 Checking and Reconciling the Facts

#### `validate_vin`

**What it does:** Checks VIN length and prohibited characters (I, O and Q, which are common OCR
confusions, with a suggested repair), and calculates the VIN check digit.

**Why it exists separately:** VIN validation is deterministic and should not depend on the LLM's
judgment.

**When the agent calls it:** When a VIN has been extracted or recorded and needs validation,
particularly one from free text or one that looks suspicious.

**What it returns and how the agent uses it:** Validation results, including the check-digit result.
The check digit is reported as a diagnostic signal, not used as an automatic rejection criterion,
because four of the five synthetic sample VINs fail it.

#### `compare_documents(a, b)`

**What it does:** Compares two documents to identify exact duplicates, content duplicates (for
example, a PDF and a scan of it), revisions, or conflicting versions.

**Why it exists separately:** Document-level comparison is different from comparing individual field
values. It prevents an outdated version from being treated as an equally current source.

**When the agent calls it:** When two documents appear to represent the same underlying document,
particularly when multiple versions are present.

**What it returns and how the agent uses it:** A comparison result indicating the relationship between
the documents. Duplicates and superseded versions are excluded from reconciliation, keeping the
better-quality or newer copy; conflicting versions stay active so the conflict surfaces in
reconciliation. For CLM-003, the newer settlement explicitly replaces the earlier version, so the
agent uses the current $23,800 payout rather than treating the old $24,100 figure as an unresolved
conflict.

#### `check_settlement_math`

**What it does:** Checks whether the settlement amounts are internally consistent: ACV − deductible −
prior damage = net payout and loan balance − net payout = gap.

**Why it exists separately:** Arithmetic is deterministic and should be checked independently of the
LLM's interpretation of the documents.

**When the agent calls it:** When a settlement breakdown with line items is present.

**What it returns and how the agent uses it:** Arithmetic findings, including expected and stated
amounts. Mismatches are warnings by default and can be made blocking through configuration. Missing
or nonnumeric values cannot be verified and must be handled by the relevant field-validation rules.

#### `reconcile(field)`

**What it does:** Compares all recorded candidate values for a field across the claim's documents.

**Why it exists separately:** Extracting a value and deciding which source to trust are different
tasks. Reconciliation applies consistent rules across all candidate readings.

**When the agent calls it:** When a field has multiple candidate readings or evidence from multiple
sources that needs to be compared.

**What it returns and how the agent uses it:** Agreement or conflict findings, source evidence,
confidence, and a suggested value when sufficiently supported. Source authority and read quality
influence the result: police reports carry more weight for dates, settlements for payouts, and
finance agreements for loan balances. Duplicates do not count as independent evidence, and customer
statements carry less weight. When neither side is clearly stronger, it returns "No guess" rather
than picking one. Unresolved conflicts remain flagged for human review, even when a best guess is
available.

#### `cross_check_documents`

**What it does:** Checks consistency across other shared identifiers and details, including vehicle
information, lender, owner and borrower names, policy numbers, and police report numbers.

**Why it exists separately:** A claim can have correct individual field values but still contain
mismatched documents or identities.

**When the agent calls it:** Once the relevant document information has been collected and there is
enough evidence to compare shared details.

**What it returns and how the agent uses it:** Cross-document consistency findings. The agent uses
discrepancies to flag possible misfiled documents or other issues requiring investigation, rather
than assuming that every document belongs to the same claim.

### 3.4 Deciding and Replying

#### `evaluate_claim`

**What it does:** Applies the claim rules to determine the status, outstanding issues, and
recommended next action.

**Why it exists separately:** Final status must be determined by explicit, reproducible rules rather
than by the LLM's own judgment.

**When the agent calls it:** After collecting the relevant evidence and running the necessary
validation and reconciliation checks, and again when new information changes the claim.

**What it returns and how the agent uses it:** A status of `complete`, `incomplete`, or
`needs_review`, along with findings and a recommended action. The model cannot directly set or
override the status through its response.

#### `note_customer_intent`

**What it does:** Records what a customer says beyond field values: a promise to provide a document
(with an expected date), a document they cannot provide, a question about the claim, or an
authorization they give.

**Why it exists separately:** A customer's intention or promise is not the same as documentary
evidence, and unanswered questions must not be lost between turns.

**When the agent calls it:** When processing a customer message that contains a commitment, question,
authorization, or other relevant intent.

**What it returns and how the agent uses it:** Structured customer-intent information, including
promises and questions. The agent uses it to track follow-up requirements and ensure that its reply
addresses the customer's actual message.

#### `submit_turn`

**What it does:** Validates and submits the completed agent turn.

**Why it exists separately:** Producing a response is not enough; the final status, action, evidence
processing, and customer communication must satisfy the workflow's requirements.

**When the agent calls it:** When it believes the turn is ready to finish.

**What it returns and how the agent uses it:** Acceptance or specific validation errors. Submission is
rejected if any submitted document has not been processed, the action conflicts with the status, a
missing document is not requested, a customer question goes unanswered, or the message reveals
internal findings, promises unauthorized outcomes, claims actions that were not taken (such as
contacting the lender), or offers substitutes for a required document. The agent uses the feedback
to correct the response and retry. On acceptance, account numbers in the customer message are masked
to the last four digits.

---

## 5. How the Agent Chooses Tools

### Conditional tool selection

The tools are exposed through a small `@tool` decorator that derives each tool's description from its
parameters and docstring. At each step, the model can choose from the available tools rather than
following a hard-coded sequence.

The loop in `claims_agent/agent.py` asks the model what to do, executes its requested tool calls,
returns the results, and repeats until `submit_turn` accepts the turn. Tool errors are returned to
the model so it can correct its arguments or choose another approach. Execution is bounded by a step
limit and a per-turn deadline; if either is reached before an accepted submission, the fallback
planner completes the turn from the evidence already recorded.

The model is encouraged to:

- Read documents before attempting to extract their fields.
- Use enhanced OCR when standard reading misses fields or produces weak results.
- Use vision only when OCR cannot reliably recover the required information.
- Compare documents when multiple files appear to be versions of the same document.
- Reconcile fields when multiple candidate readings need to be compared.
- Avoid repeating work when previously processed evidence remains valid.
- Run independent checks without unnecessary sequential calls.

Tool results can guide the next step. For example, a scan with missing fields returns a hint to try
enhanced OCR, a settlement with line items prompts the arithmetic check, and a document type with no
form layout prompts the agent to read it and record observations. The agent therefore adapts its
tool use to the evidence available in each claim.

### What the model decides versus what Python controls

The model chooses the order of work, decides each document's type, interprets free text, identifies
customer intent, and drafts customer-facing and reviewer-facing messages. Python tools validate
evidence, perform arithmetic and reconciliation, apply claim rules, and enforce submission
requirements.

This separation allows the agent to be flexible about how it investigates a claim without allowing
the model to invent verified facts or independently approve a claim.

### Evidence from the sample runs

The execution traces demonstrate that tool selection depends on the claim:

- **CLM-003:** Document comparison identifies a superseded settlement version.
- **CLM-004:** Vision reading is used for scanned documents with an incomplete VIN.
- **CLM-001:** A concern is recorded from the adjuster's note.
- **CLM-002 and CLM-005:** Customer replies are recorded and evaluated without unnecessarily
  re-reading previously processed documents.
- **Later uploads:** New files are added to the claim and the agent is told to read and process them;
  a file re-uploaded under the same name is kept separately so the two can be compared.

Each claim's `tools_used` output records the tools called during processing.

### Fallback behavior

When an LLM provider is unavailable or `LLM_PROVIDER=none` is configured, `FallbackPlanner` provides a
deterministic, rule-based path through the same tools. It is a fallback for outages and testing, not
the primary agent. It is more predictable but less flexible at interpreting unusual documents and
customer messages.

---

## 6. Key decisions and trade-offs

### 1. Python for facts, LLM for interpretation

**Decision:** Use Python-based layout rules to extract fields from known forms, with an LLM fallback
for unreadable scans, unfamiliar documents, and free text. Python validates values, reconciles
evidence, and determines claim status.

**Alternative:** Let the LLM extract all fields and assign statuses.

**Trade-off:** Form-specific rules require maintenance but provide more predictable, traceable
results. The LLM adds flexibility where deterministic extraction is insufficient. Every LLM-proposed
value must include verifiable supporting evidence.

### 2. Layout-aware OCR with vision fallback

**Decision:** Prefer PDF text extraction, followed by CPU-based RapidOCR for scanned documents. Retry
poor OCR with enhanced preprocessing before using a vision model.

**Alternative:** Send every page to a vision LLM.

**Trade-off:** Local OCR is inexpensive and works offline, but layout rules are tailored to known
forms and handwriting remains challenging. Vision fallback improves coverage at the cost of latency
and variability.

### 3. Field-specific source authority and conservative conflict resolution

**Decision:** Reconcile values using source authority, extraction quality, and independent
corroboration. Police reports are preferred for dates of loss, settlement breakdowns for payouts,
and finance agreements for loan balances. Duplicates and superseded versions do not count as
independent evidence.

**Alternative:** Use majority voting or automatically select the highest-scoring value.

**Trade-off:** Manually configured weights allow the agent to suggest a likely value, but they are
not statistically calibrated. Conflicts remain `needs_review` even when a preferred candidate exists.
This prioritizes safe human verification over potentially incorrect automatic resolution.

### 4. Deterministic status rules

**Decision:** Python rules determine claim status, with `needs_review` taking precedence over
`incomplete`, which takes precedence over `complete`. Warnings are distinguished from blocking
issues, and configuration flags control warning severity.

**Alternative:** Let the LLM decide the final status or block every warning.

**Trade-off:** Explicit rules make decisions reproducible and testable but require business logic to
be maintained in code. Settlement arithmetic mismatches are warnings by default; distinguishing
explained deductible differences from unexplained discrepancies remains an improvement area.

### 5. Custom agent loop instead of a framework

**Decision:** Implement a lightweight tool-calling loop in `agent.py`, with explicit submission
validation, iteration limits, and timeouts.

**Alternative:** Use LangChain or LangGraph to manage execution.

**Trade-off:** The custom loop is straightforward to debug and explain, but retries, provider
fallback, and tracing require additional maintenance. A framework may be worthwhile if the workflow
becomes more complex.

### 6. Persistent claim state with fresh model context

**Decision:** Provide a concise claim summary to the model on each turn and persist the full claim
state as JSON. Customer replies update existing evidence and trigger re-evaluation.

**Alternative:** Send the entire conversation history with every request.

**Trade-off:** Bounded context reduces token usage and supports session recovery, but important
information must be explicitly preserved in state to avoid losing conversational context.

### 7. Provider fallback with an offline planner

**Decision:** Use a shared OpenAI-compatible interface with a Gemini → Groq → rule-based planner
fallback chain.

**Alternative:** Depend on a single provider or require local model inference for all operations.

**Trade-off:** Provider fallback improves resilience to rate limits and outages, while the
deterministic planner can complete known workflows without further LLM calls. However, it is less
flexible with unfamiliar inputs, and different providers may behave differently. Local inference
remains configurable but is constrained by the development machine's hardware.

### Overall design principle

Use the LLM for interpretation and conditional tool selection, and Python for validation,
reconciliation, and final status decisions. Preserve evidence and uncertainty rather than allowing
an unsupported guess to finalize a claim.

---

## 7. What we'd do with more time

**Test accuracy on more claims.** Build a larger set of manually verified claims and measure how often
the agent extracts the correct values and assigns the correct status. This would also help determine
whether the current confidence levels are reliable.

**Handle financial discrepancies more carefully.** The current system reports settlement arithmetic
mismatches as warnings by default. We'd distinguish explainable differences, such as a deductible
that was not subtracted, from unexplained discrepancies that should block finalization and require
human review.

**Support unfamiliar documents.** Improve OCR for poor-quality scans and extend the fallback for
document layouts the current extraction rules don't recognize (today the agent reads such documents
as free text). If a value cannot be verified, the agent should continue to flag it for review rather
than guess.

**Make human review easier.** The UI already lists each issue with its best guess and shows document
previews. The next step is to show conflicting values side by side with the exact place they appear
in each source document, explain why the claim needs attention, and let the reviewer confirm or
correct the value. Store reviewer corrections so recurring errors can be identified and used to
improve future evaluations.

**Prepare for production use.** Replace JSON-based claim state with a database, add structured
logging and safe retries, and implement authentication, access controls, and appropriate
data-retention policies before processing real customer information.

"""HTTP API + live agent event stream (SSE) for the web UI."""
from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import service
from .agent import Agent
from .config import ROOT, settings
from .ingest import pdf_to_image
from .report import claim_output, prioritize

app = FastAPI(title="Total-Loss Claims Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _summary(cid: str) -> dict:
    st = service.load(cid)
    folder = settings.claims_dir / cid
    files = sorted(p.name for p in folder.iterdir() if p.is_file())
    s = {"claim_id": cid, "files": files, "processed": st is not None}
    if st and st.evaluation:
        ev = st.evaluation
        s.update(status=ev.status, turns=len(st.turns), pending_replies=st.pending_replies,
                 blocking=sum(1 for i in ev.issues if i.severity == "blocking"),
                 warnings=sum(1 for i in ev.issues if i.severity == "warning"),
                 claimant=_claimant(st), payout=ev.extracted_fields["insurance_payout"].value)
    return s


def _claimant(st) -> Optional[str]:
    return next((d.meta["person_name"] for d in st.documents.values() if d.meta.get("person_name")), None)


@app.get("/api/config")
def config():
    agent = Agent()
    return {"agent_mode": agent.mode, "provider": settings.provider, "fallbacks": list(settings.fallback_providers),
            "strict_settlement_math": settings.strict_settlement_math,
            "warnings_block_finalize": settings.warnings_block_finalize,
            "required_documents": list(settings.required_documents)}


@app.get("/api/claims")
def claims():
    return [_summary(c) for c in service.list_claim_ids()]


@app.get("/api/claims/{cid}")
def claim(cid: str):
    st = service.load(cid)
    if st is None:
        return {"claim_id": cid, "processed": False, **_summary(cid)}
    out = claim_output(st)
    out["processed"] = True
    out["pending_replies"] = st.pending_replies
    out["claimant"] = _claimant(st)
    out["evidence"] = {k: v.candidates for k, v in st.evaluation.extracted_fields.items()}
    out["documents_detail"] = [{"file": d.file, "kind": d.kind, "doc_type": d.doc_type.value, "status": d.status,
                                "related_to": d.related_to, "status_reason": d.status_reason,
                                "method": d.text_method, "ocr_confidence": d.ocr_confidence,
                                "added_in_turn": d.added_in_turn, "type_reason": d.type_reason,
                                "meta": {k: v for k, v in d.meta.items() if k != "vision_transcripts"}}
                               for d in st.documents.values()]
    out["concerns"] = [c.model_dump() for c in st.concerns]
    out["commitments"] = [c.model_dump() for c in st.commitments]
    out["tool_log"] = [c.model_dump() for c in st.tool_log]
    return out


def _resolve_file(cid: str, name: str) -> Path:
    for base in (settings.claims_dir / cid, settings.state_dir / cid):
        p = (base / name).resolve()
        if p.is_file() and base.resolve() in p.parents:
            return p
    raise HTTPException(404, "file not found")


@app.get("/api/claims/{cid}/files/{name}")
def file(cid: str, name: str):
    return FileResponse(_resolve_file(cid, name))


@app.get("/api/claims/{cid}/preview/{name}")
def preview(cid: str, name: str):
    """PNG preview for any document (PDFs rendered server-side)."""
    p = _resolve_file(cid, name)
    if p.suffix.lower() != ".pdf":
        return FileResponse(p)
    import cv2

    out = settings.cache_dir / f"preview_{cid}_{p.stem}.png"
    if not out.exists():
        cv2.imwrite(str(out), cv2.cvtColor(pdf_to_image(p), cv2.COLOR_RGB2BGR))
    return FileResponse(out)


def _stream(job: Callable[[Callable[[dict], None]], None]) -> StreamingResponse:
    q: queue.Queue = queue.Queue()

    def runner():
        try:
            job(q.put)
        except Exception as e:  # surface failures to the UI instead of hanging the stream
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            q.put(None)

    threading.Thread(target=runner, daemon=True).start()

    def gen():
        while True:
            e = q.get()
            if e is None:
                yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                return
            yield "data: " + json.dumps(e, default=str) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/claims/{cid}/process")
def process(cid: str):
    return _stream(lambda emit: service.run_initial(cid, emit=emit))


@app.post("/api/claims/{cid}/replay-reply")
def replay(cid: str):
    st = service.load(cid)
    if not st or not st.pending_replies:
        raise HTTPException(400, "no pending customer reply for this claim")
    reply = st.pending_replies[0]
    return _stream(lambda emit: service.customer_turn(cid, emit=emit, source_file=reply))


@app.post("/api/claims/{cid}/message")
async def message(cid: str, text: str = Form(""), files: list[UploadFile] = File(default=[])):
    tmp = Path(tempfile.mkdtemp())
    paths = []
    for f in files:
        safe = Path(f.filename or "upload").name
        p = tmp / safe
        with p.open("wb") as fh:
            shutil.copyfileobj(f.file, fh)
        paths.append(p)
    if not text.strip() and not paths:
        raise HTTPException(400, "empty message")
    return _stream(lambda emit: service.customer_turn(cid, text=text, uploads=paths, emit=emit))


@app.post("/api/claims/{cid}/reset")
def reset(cid: str):
    service.reset(cid)
    return {"ok": True}


@app.get("/api/prioritization")
def prioritization():
    states = [s for s in (service.load(c) for c in service.list_claim_ids()) if s and s.evaluation]
    return prioritize(states) if states else {"processing_order": [], "do_in_parallel_today": []}


_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")

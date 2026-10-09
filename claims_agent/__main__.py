"""CLI.

  python -m claims_agent run                 # all claims (initial review + replay customer replies) -> outputs/
  python -m claims_agent run CLM-003         # one claim
  python -m claims_agent chat CLM-002        # play the customer in the terminal
  python -m claims_agent serve               # API + web UI on http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys

from . import service
from .config import settings
from .report import claim_output, prioritize


def _printer(verbose: bool):
    def emit(e: dict):
        if e["type"] == "turn_start":
            print(f"\n== turn {e['turn']} ({e['kind']}) - {e['agent']}")
        elif e["type"] == "tool" and verbose:
            mark = "ok " if e["ok"] else "ERR"
            args = json.dumps(e["input"])[:90]
            print(f"   [{mark}] {e['tool']}({args})")
        elif e["type"] == "thought" and verbose:
            print(f"   ~ {e['text'][:200]}")
        elif e["type"] == "turn_end":
            print(f"   -> status={e['status']} action={e['next_action']['type']}  [by {e.get('driven_by')}]")
    return emit


def cmd_run(args):
    ids = args.claims or service.list_claim_ids()
    states = []
    for cid in ids:
        state = service.process_claim_with_replies(cid, emit=_printer(not args.quiet))
        out = claim_output(state)
        service.dump_json(out, settings.output_dir / f"{cid}.json")
        states.append(state)
    if not args.claims:
        pr = prioritize(states)
        service.dump_json(pr, settings.output_dir / "prioritization.json")
        print("\nRecommended processing order:")
        for r in pr["processing_order"]:
            print(f"  {r['rank']}. {r['claim_id']} [{r['status']}] {r['reason']}")
    print(f"\nWrote outputs to {settings.output_dir}")


def cmd_chat(args):
    cid = args.claim
    state = service.load(cid)
    if state is None or args.fresh:
        state = service.run_initial(cid, emit=_printer(args.verbose), hold_back_replies=True)
    last = state.turns[-1].next_action
    print(f"\nStatus: {state.evaluation.status}")
    print(f"\n--- Agent -> customer ---\n{last.message or '(no customer message; action=' + last.type + ')'}\n")
    print("You are the customer. Type a reply (attach files with: /attach path1 path2 ; /quit to exit).")
    pending: list = []
    from pathlib import Path
    while True:
        try:
            line = input("customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/attach"):
            pending += [Path(p) for p in line.split()[1:]]
            print(f"  attached: {[p.name for p in pending]} (send a message or empty line to submit)")
            continue
        if not line and not pending:
            continue
        state = service.customer_turn(cid, text=line, uploads=pending, emit=_printer(args.verbose))
        pending = []
        na = state.turns[-1].next_action
        print(f"\nStatus: {state.evaluation.status}  (action: {na.type})")
        print(f"\n--- Agent -> customer ---\n{na.message or '(no message)'}\n")


def cmd_serve(args):
    import uvicorn

    uvicorn.run("claims_agent.api:app", host=args.host, port=args.port, reload=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="claims_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("claims", nargs="*")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("chat")
    c.add_argument("claim")
    c.add_argument("--fresh", action="store_true")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(fn=cmd_chat)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.fn(args)


if __name__ == "__main__":
    main()

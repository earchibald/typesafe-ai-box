"""Run the Box experiment suite.

    uv run run.py                 # every experiment not yet run
    uv run run.py --only e1,e3    # some of them
    uv run run.py --force         # rerun and overwrite
    uv run run.py --mock          # no API calls: fabricated answers into results-mock/

Needs TYPESAFE_API_KEY in the environment or in a .env file beside this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from boxlab.api import Api, MockApi, load_key
from boxlab.experiments import EXPERIMENTS, Ctx

ROOT = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    out = ROOT / ("results-mock" if args.mock else "results")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    if args.mock:
        api = MockApi()
    else:
        key = load_key()
        if not key:
            print("TYPESAFE_API_KEY is not set (environment or .env).", file=sys.stderr)
            return 2
        api = Api(key)

    names = [n for n in args.only.split(",") if n] or list(EXPERIMENTS)
    ctx = Ctx(api, out)
    for name in names:
        path = out / "raw" / f"{name}.jsonl"
        if path.exists() and not args.force:
            print(f"{name}: already run, skipping (use --force to rerun)")
            continue
        t0 = time.time()
        records = EXPERIMENTS[name](ctx)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        ok = sum(r.get("status") == 200 for r in records)
        print(f"{name}: {len(records)} requests, {ok} ok, {time.time() - t0:.1f}s")
        if name != "e0" and ok == 0:
            first = records[0] if records else {}
            print(f"  every request failed; first error: {first.get('status')} {first.get('error')}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

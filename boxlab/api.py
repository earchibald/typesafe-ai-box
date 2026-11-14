"""Thin HTTP client for POST /v1/systemone with images, plus a mock for pipeline tests."""

from __future__ import annotations

import base64
import io
import os
import random
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"  # pinned: Box needs Jev 1.14, and aliases move
ROOT = Path(__file__).resolve().parent.parent


def load_key() -> str | None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TYPESAFE_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    return None


class Assets:
    """Collects the images of one request and hands out their reference strings."""

    def __init__(self) -> None:
        self.map: dict[str, dict] = {}

    def add(self, image: Image.Image) -> str:
        ref = f"img-{uuid.uuid4()}"
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "PNG")
        self.map[ref] = {
            "type": "image",
            "format": "png",
            "data_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        }
        return ref


class Api:
    def __init__(self, key: str) -> None:
        self.client = httpx.Client(timeout=90.0, headers={"Authorization": f"Bearer {key}"})

    def ask(
        self,
        state,
        questions: dict,
        assets: Assets | None = None,
        *,
        model: str = MODEL,
        images_header: bool = True,
        expect: dict | None = None,  # used only by MockApi
    ) -> dict:
        body = {"model": model, "state": state, "questions": questions}
        if assets and assets.map:
            body["assets"] = assets.map
        headers = {"TypeSafe-Danger-Zone-Images": "true"} if images_header else {}
        last: dict = {}
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                r = self.client.post(URL, json=body, headers=headers)
            except httpx.HTTPError as e:
                last = {"status": 0, "error": repr(e), "latency_s": time.perf_counter() - t0}
                time.sleep(min(30, 1.5 * 2**attempt))
                continue
            latency = time.perf_counter() - t0
            try:
                payload = r.json()
            except ValueError:
                payload = {"raw": r.text[:2000]}
            last = {
                "status": r.status_code,
                "latency_s": round(latency, 4),
                "request_id": r.headers.get("x-request-id") or r.headers.get("request-id"),
                "attempts": attempt + 1,
            }
            if r.status_code == 200:
                last.update(model=payload.get("model"), usage=payload.get("usage"), answers=payload.get("answers"))
                return last
            last["error"] = payload
            if r.status_code in (429, 500, 502, 503, 504, 529):
                wait = float(r.headers.get("retry-after") or 0) or min(30, 1.5 * 2**attempt)
                time.sleep(wait)
                continue
            return last
        return last


class MockApi:
    """Fabricates plausible answers from `expect`. For testing the pipeline only:
    its output is written under results-mock/ and never reported as a finding."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def ask(self, state, questions, assets=None, *, model=MODEL, images_header=True, expect=None) -> dict:
        answers = {}
        for qid, q in questions.items():
            e = (expect or {}).get(qid, {})
            if q["type"] == "box":
                gt = e.get("gt") or [0.3, 0.3, 0.6, 0.6]
                j = lambda v: min(1.0, max(0.0, v + self.rng.gauss(0, 0.02)))  # noqa: E731
                l, t, r, b = j(gt[0]), j(gt[1]), j(gt[2]), j(gt[3])
                answers[qid] = {
                    "type": "box",
                    "box": {"left": min(l, r), "top": min(t, b), "right": max(l, r) + 1e-3, "bottom": max(t, b) + 1e-3},
                    "confidence": round(self.rng.uniform(0.4, 0.99), 3),
                }
            elif q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": e.get("p", 0.5)}
            elif q["type"] == "choice":
                options = list(q["criteria"])
                pick = e.get("choice", options[0])
                answers[qid] = {
                    "type": "choice",
                    "choice": pick,
                    "confidence": 0.9,
                    "probabilities": {o: (0.9 if o == pick else 0.1 / max(1, len(options) - 1)) for o in options},
                }
        n_img = len(assets.map) if assets else 0
        return {
            "status": 200,
            "latency_s": round(self.rng.uniform(0.2, 0.6), 4),
            "request_id": "mock",
            "attempts": 1,
            "model": "mock",
            "usage": {"input_tokens": 100 + 1500 * n_img + 20 * len(questions), "output_tokens": 0},
            "answers": answers,
        }

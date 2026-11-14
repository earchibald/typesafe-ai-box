"""Grade the raw records and write results/summary.json plus overlay images.

    uv run analyze.py            # reads results/
    uv run analyze.py --mock     # reads results-mock/
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
HIT = 0.5  # IoU at or above this counts as a hit


def edges(ans: dict) -> list[float]:
    b = ans["box"]
    return [b["left"], b["top"], b["right"], b["bottom"]]


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: list[float], b: list[float]) -> float:
    inter = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def contains(outer: list[float], inner: list[float], pad: float = 0.0) -> bool:
    return (
        outer[0] - pad <= inner[0] and outer[1] - pad <= inner[1] and outer[2] + pad >= inner[2] and outer[3] + pad >= inner[3]
    )


def union_box(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(st.fmean(xs), 4) if xs else None


def median(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 4) if xs else None


def rate(flags) -> float | None:
    flags = list(flags)
    return round(sum(bool(f) for f in flags) / len(flags), 4) if flags else None


def quantile(xs, q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4)


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability that a random positive scores above a random negative."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else None


def load(out: Path) -> dict[str, list[dict]]:
    data = {}
    for path in sorted((out / "raw").glob("*.jsonl")):
        data[path.stem] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return data


def boxes(records: list[dict], want=lambda rec, qid, t: True):
    """Yield (record, qid, target, predicted edges, confidence, IoU) for graded Box answers."""
    for rec in records:
        if rec.get("status") != 200:
            continue
        for qid, t in rec["targets"].items():
            ans = (rec.get("answers") or {}).get(qid)
            if not ans or ans.get("type") != "box" or "gt" not in t or not want(rec, qid, t):
                continue
            e = edges(ans)
            yield rec, qid, t, e, ans.get("confidence"), iou(e, t["gt"])


def acc_block(rows: list[tuple]) -> dict:
    ious = [r[5] for r in rows]
    return {
        "n": len(rows),
        "mean_iou": mean(ious),
        "median_iou": median(ious),
        "hit_rate": rate(i >= HIT for i in ious),
        "iou_ge_0_75": rate(i >= 0.75 for i in ious),
        "iou_ge_0_9": rate(i >= 0.9 for i in ious),
        "mean_confidence": mean(r[4] for r in rows),
    }


def analyze(data: dict[str, list[dict]]) -> dict:
    s: dict = {"hit_threshold_iou": HIT}
    pooled: list[tuple[float, float, str]] = []  # (confidence, IoU, experiment)

    def pool(rows: list[tuple], exp: str) -> None:
        pooled.extend((r[4], r[5], exp) for r in rows if r[4] is not None)

    all_ok = [r for recs in data.values() for r in recs if r.get("status") == 200]
    s["overall"] = {
        "requests": sum(len(v) for v in data.values()),
        "requests_ok": len(all_ok),
        "models": sorted({r.get("model") for r in all_ok if r.get("model")}),
        "input_tokens_total": sum((r.get("usage") or {}).get("input_tokens", 0) for r in all_ok),
        "latency_median_s": median(r["latency_s"] for r in all_ok),
        "latency_p90_s": quantile([r["latency_s"] for r in all_ok], 0.9),
    }
    s["overall"]["cost_usd"] = round(s["overall"]["input_tokens_total"] * 0.042 / 1e6, 4)

    if "e0" in data:
        s["e0"] = [
            {
                "case": r["meta"]["case"],
                "documented": r["meta"]["documented"],
                "status": r.get("status"),
                "error": json.dumps(r.get("error"))[:300] if r.get("error") else None,
            }
            for r in data["e0"]
        ]

    if "e1" in data:
        rows = list(boxes(data["e1"]))
        pool(rows, "e1")
        by_shape = defaultdict(list)
        for r in rows:
            by_shape[r[2]["shape"]].append(r)
        errs = [abs(r[3][k] - r[2]["gt"][k]) for r in rows for k in range(4)]
        s["e1"] = {
            **acc_block(rows),
            "mean_edge_error": mean(errs),
            "mean_edge_error_px_at_1024": round(mean(errs) * 1024, 1) if errs else None,
            "by_shape": {k: acc_block(v) for k, v in sorted(by_shape.items())},
        }

    if "e2" in data:
        full = [r for r in data["e2"] if r["meta"]["mode"] == "full"]
        rows = list(boxes(full))
        pool(rows, "e2")
        by_d = defaultdict(list)
        for r in rows:
            by_d[r[0]["meta"]["diameter"]].append(r)
        nouls = defaultdict(list)
        for rec in full:
            v = ((rec.get("answers") or {}).get("visible") or {}).get("noul")
            if v is not None:
                nouls[rec["meta"]["diameter"]].append(v)
        tiles = [r for r in data["e2"] if r["meta"]["mode"] == "tile_choice" and r.get("status") == 200]
        tile_ok = {r["meta"]["parent"]: (r["answers"]["tile"]["choice"] == r["targets"]["tile"]["choice"]) for r in tiles}
        crop = defaultdict(list)
        for rec in data["e2"]:
            if rec["meta"]["mode"] != "crop_box" or rec.get("status") != 200:
                continue
            t, ans = rec["targets"]["target"], rec["answers"]["target"]
            qx, qy = {"top_left": (0, 0), "top_right": (1, 0), "bottom_left": (0, 1), "bottom_right": (1, 1)}[t["tile"]]
            e = edges(ans)
            mapped = [qx * 0.5 + e[0] / 2, qy * 0.5 + e[1] / 2, qx * 0.5 + e[2] / 2, qy * 0.5 + e[3] / 2]
            crop[rec["meta"]["diameter"]].append(iou(mapped, t["gt_full"]))
        s["e2"] = {
            "by_diameter_px": {
                str(d): {
                    **acc_block(v),
                    "area_share_of_image": round((d / 1024) ** 2, 6),
                    "presence_noul_mean": mean(nouls[d]),
                }
                for d, v in sorted(by_d.items())
            },
            "rescue": {
                "tile_choice_accuracy": rate(tile_ok.values()),
                "by_diameter_px": {
                    str(d): {"n": len(v), "mean_iou": mean(v), "hit_rate": rate(i >= HIT for i in v)} for d, v in sorted(crop.items())
                },
                "all": {"n": sum(len(v) for v in crop.values()), "mean_iou": mean(i for v in crop.values() for i in v), "hit_rate": rate(i >= HIT for v in crop.values() for i in v)},
            },
        }
        small = [r for r in rows if r[0]["meta"]["diameter"] <= 48]
        s["e2"]["full_image_small_targets"] = acc_block(small)

    if "e3" in data:
        rows = list(boxes(data["e3"]))
        pool(rows, "e3")
        conf_present = [r[4] for r in rows if r[4] is not None]
        conf_absent, landing, noul_p, noul_a = [], defaultdict(int), [], []
        by_kind = defaultdict(list)
        for rec in data["e3"]:
            if rec.get("status") != 200:
                continue
            a = rec["answers"]
            noul_p.append(a["present_noul"]["noul"])
            noul_a.append(a["absent_noul"]["noul"])
            ab, t = a["absent_box"], rec["targets"]["absent_box"]
            conf_absent.append(ab.get("confidence"))
            by_kind[rec["meta"]["kind"]].append(ab.get("confidence"))
            e = edges(ab)
            best = max(rec["meta"]["objects"], key=lambda o: iou(e, o["box"]))
            if iou(e, best["box"]) < HIT:
                landing["on no object"] += 1
            elif best["color"] == t["color"]:
                landing["on an object of the same colour"] += 1
            elif best["shape"] == t["shape"]:
                landing["on an object of the same shape"] += 1
            else:
                landing["on an unrelated object"] += 1
        conf_absent = [c for c in conf_absent if c is not None]
        s["e3"] = {
            "n_scenes": len(noul_p),
            "present_box": acc_block(rows),
            "box_confidence_present_mean": mean(conf_present),
            "box_confidence_absent_mean": mean(conf_absent),
            "box_confidence_absent_by_kind": {k: mean(v) for k, v in by_kind.items()},
            "box_confidence_auc_present_vs_absent": auc(conf_present, conf_absent),
            "noul_present_mean": mean(noul_p),
            "noul_absent_mean": mean(noul_a),
            "noul_auc": auc(noul_p, noul_a),
            "noul_accuracy_at_0_5": rate([p > 0.5 for p in noul_p] + [p <= 0.5 for p in noul_a]),
            "absent_box_lands": dict(landing),
            "box_confidence_absent_values": sorted(round(c, 3) for c in conf_absent),
            "box_confidence_present_values": sorted(round(c, 3) for c in conf_present),
        }

    if "e4" in data:
        amb, conf_amb = defaultdict(int), []
        fixed = defaultdict(list)
        wrong_twin = defaultdict(list)
        counts = []
        for rec in data["e4"]:
            if rec.get("status") != 200:
                continue
            a, t = rec["answers"], rec["targets"]
            e = edges(a["ambiguous"])
            il, ir = iou(e, t["ambiguous"]["left"]), iou(e, t["ambiguous"]["right"])
            iu = iou(e, union_box(t["ambiguous"]["left"], t["ambiguous"]["right"]))
            if max(il, ir) >= HIT:
                amb["the left one" if il >= ir else "the right one"] += 1
            elif iu >= HIT:
                amb["one box spanning both"] += 1
            else:
                amb["neither"] += 1
            conf_amb.append(a["ambiguous"].get("confidence"))
            for q in ("leftmost", "rightmost", "near_square"):
                e = edges(a[q])
                i = iou(e, t[q]["gt"])
                fixed[q].append((rec, q, t[q], e, a[q].get("confidence"), i))
                wrong_twin[q].append(iou(e, t[q]["other"]) >= HIT)
            counts.append(a["count"]["choice"] == "several")
        for q, rows in fixed.items():
            pool(rows, "e4")
        s["e4"] = {
            "n_scenes": len(counts),
            "ambiguous_box_lands_on": dict(amb),
            "ambiguous_mean_confidence": mean(conf_amb),
            "disambiguated": {q: {**acc_block(rows), "picked_the_other_twin": rate(wrong_twin[q])} for q, rows in fixed.items()},
            "count_choice_says_several": rate(counts),
        }

    if "e5" in data:
        rows = list(boxes(data["e5"]))
        pool(rows, "e5")
        by_field = defaultdict(list)
        for r in rows:
            by_field[r[2]["field"]].append(r)

        def field_block(v: list[tuple]) -> dict:
            return {
                **acc_block(v),
                "target_area_share": mean(area(r[2]["gt"]) for r in v),
                "contains_target": rate(contains(r[3], r[2]["gt"]) for r in v),
                "contains_target_with_2pct_pad": rate(contains(r[3], r[2]["gt"], 0.02) for r in v),
                "contains_target_with_5pct_pad": rate(contains(r[3], r[2]["gt"], 0.05) for r in v),
            }

        s["e5"] = {"all": field_block(rows), "by_field": {k: field_block(v) for k, v in by_field.items()}}

    if "e6" in data:
        recs = [r for r in data["e6"] if r.get("status") == 200]
        res = defaultdict(list)
        for row in boxes([r for r in recs if r["meta"]["part"] == "resolution"]):
            res[row[0]["meta"]["size"]].append(row)
        pool([r for v in res.values() for r in v], "e6")
        res_tokens = defaultdict(list)
        res_latency = defaultdict(list)
        for r in recs:
            if r["meta"]["part"] == "resolution":
                res_tokens[r["meta"]["size"]].append((r.get("usage") or {}).get("input_tokens"))
                res_latency[r["meta"]["size"]].append(r["latency_s"])
        aspect = defaultdict(list)
        for row in boxes([r for r in recs if r["meta"]["part"] == "aspect"]):
            aspect[row[0]["meta"]["aspect"]].append(row)
        pool([r for v in aspect.values() for r in v], "e6")
        aspect_tokens = defaultdict(list)
        for r in recs:
            if r["meta"]["part"] == "aspect":
                aspect_tokens[r["meta"]["aspect"]].append((r.get("usage") or {}).get("input_tokens"))
        nq = {}
        for r in recs:
            if r["meta"]["part"] == "questions":
                rows = list(boxes([r]))
                nq[str(r["meta"]["n_questions"])] = {
                    "input_tokens": (r.get("usage") or {}).get("input_tokens"),
                    "latency_s": r["latency_s"],
                    "mean_iou": mean(x[5] for x in rows),
                }
        repeats = [r for r in recs if r["meta"]["part"] == "repeat"]
        spread, conf_spread = [], []
        if repeats:
            for qid in repeats[0]["answers"]:
                es = [edges(r["answers"][qid]) for r in repeats]
                spread.append(max(max(e[k] for e in es) - min(e[k] for e in es) for k in range(4)))
                cs = [r["answers"][qid].get("confidence") or 0 for r in repeats]
                conf_spread.append(max(cs) - min(cs))
        # Least-squares line: input tokens against pixels, same questions at every size.
        pts = [(size * size, t) for size, ts in res_tokens.items() for t in ts if t is not None]
        fit = None
        if len(pts) >= 2:
            mx, my = st.fmean(p[0] for p in pts), st.fmean(p[1] for p in pts)
            den = sum((p[0] - mx) ** 2 for p in pts)
            if den:
                slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
                fit = {"tokens_per_megapixel": round(slope * 1e6, 1), "intercept_tokens": round(my - slope * mx, 1)}
        s["e6"] = {
            "by_square_size_px": {
                str(k): {**acc_block(v), "input_tokens": median(res_tokens[k]), "latency_median_s": median(res_latency[k])}
                for k, v in sorted(res.items())
            },
            "by_aspect": {k: {**acc_block(v), "input_tokens": median(aspect_tokens[k])} for k, v in aspect.items()},
            "tokens_vs_pixels_fit": fit,
            "by_question_count": dict(sorted(nq.items(), key=lambda kv: int(kv[0]))),
            "repeat": {
                "runs": len(repeats),
                "max_edge_spread": round(max(spread), 6) if spread else None,
                "max_confidence_spread": round(max(conf_spread), 6) if conf_spread else None,
            },
        }

    if "e7" in data:
        rows = list(boxes(data["e7"]))
        pool(rows, "e7")
        by_mode = defaultdict(list)
        for r in rows:
            by_mode[r[2]["mode"]].append(r)
        s["e7"] = {k: acc_block(v) for k, v in by_mode.items()}

    if pooled:
        bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0001)]
        s["confidence"] = {
            "n": len(pooled),
            "spearman_confidence_vs_iou": spearman([p[0] for p in pooled], [p[1] for p in pooled]),
            "auc_confidence_predicts_hit": auc([p[0] for p in pooled if p[1] >= HIT], [p[0] for p in pooled if p[1] < HIT]),
            "overall_hit_rate": rate(p[1] >= HIT for p in pooled),
            "bins": [
                {
                    "confidence": f"{lo:.2f} to {min(hi, 1.0):.2f}",
                    "n": len(b),
                    "mean_iou": mean(p[1] for p in b),
                    "hit_rate": rate(p[1] >= HIT for p in b),
                }
                for lo, hi in bins
                for b in [[p for p in pooled if lo <= p[0] < hi]]
            ],
            "points": [[round(p[0], 3), round(p[1], 3), p[2]] for p in pooled],
        }
    return s


def overlays(out: Path, data: dict[str, list[dict]]) -> int:
    """Draw ground truth (green) and the model's box (magenta) on the saved sample images."""
    dest = out / "overlays"
    dest.mkdir(exist_ok=True)
    n = 0
    for recs in data.values():
        for rec in recs:
            src = out / "samples" / f"{rec['id']}.jpg"
            if not src.exists() or rec.get("status") != 200:
                continue
            img = Image.open(src).convert("RGB")
            d = ImageDraw.Draw(img)
            w, h = img.size
            lw = max(2, w // 300)
            for qid, t in rec["targets"].items():
                for key in ("gt", "left", "right"):
                    if key in t and not (key == "gt" and "left" in t):
                        g = t[key]
                        d.rectangle([g[0] * w, g[1] * h, g[2] * w, g[3] * h], outline=(20, 150, 70), width=lw)
                ans = (rec.get("answers") or {}).get(qid)
                if ans and ans.get("type") == "box":
                    e = edges(ans)
                    d.rectangle([e[0] * w, e[1] * h, e[2] * w, e[3] * h], outline=(225, 30, 160), width=lw)
            img.save(dest / f"{rec['id']}.jpg", "JPEG", quality=88)
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()
    out = ROOT / ("results-mock" if args.mock else "results")
    data = load(out)
    summary = analyze(data)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    n = overlays(out, data)
    brief = {k: v for k, v in summary.items() if k != "confidence"}
    if "confidence" in summary:
        brief["confidence"] = {k: v for k, v in summary["confidence"].items() if k != "points"}
    print(json.dumps(brief, indent=2))
    print(f"\nwrote {out / 'summary.json'} and {n} overlays")


if __name__ == "__main__":
    main()

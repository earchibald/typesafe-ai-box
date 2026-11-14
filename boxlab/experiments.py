"""The experiment suite. Each function sends requests and returns one record per request.

A record is: {exp, id, meta, targets, status, latency_s, usage, answers, ...}.
`targets[qid]` holds what the analysis needs to grade that question (ground truth, role).
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from . import scenes as S
from .api import Assets

WORKERS = 4
SAMPLES_PER_EXPERIMENT = 3


def box_q(what: str, path: str = "image") -> dict:
    return {"type": "box", "instructions": f"{what} in `{path}`"}


def noul_q(what: str, path: str = "image") -> dict:
    return {"type": "noul", "instructions": f"Is there {what} in `{path}`?"}


def a_an(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


class Ctx:
    def __init__(self, api, out: Path) -> None:
        self.api, self.out = api, out
        self._saved: dict[str, int] = {}

    def sample(self, exp: str, rid: str, image: Image.Image) -> None:
        """Keep the first few images of each experiment for overlays."""
        n = self._saved.get(exp, 0)
        if n < SAMPLES_PER_EXPERIMENT:
            self._saved[exp] = n + 1
            d = self.out / "samples"
            d.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(d / f"{rid}.jpg", "JPEG", quality=88)

    def run(self, jobs: list[dict]) -> list[dict]:
        def one(job: dict) -> dict:
            resp = self.api.ask(job["state"], job["questions"], job.get("assets"), expect=job.get("expect"), **job.get("kw", {}))
            return {"exp": job["exp"], "id": job["id"], "meta": job.get("meta", {}), "targets": job.get("targets", {}), **resp}

        with ThreadPoolExecutor(WORKERS) as pool:
            return list(pool.map(one, jobs))


def _image_job(ctx: Ctx, exp: str, rid: str, image: Image.Image, questions: dict, targets: dict, meta: dict) -> dict:
    assets = Assets()
    ref = assets.add(image)
    ctx.sample(exp, rid, image)
    expect = {q: ({"gt": t["gt"]} if "gt" in t else {k: t[k] for k in ("p", "choice") if k in t}) for q, t in targets.items()}
    return {
        "exp": exp,
        "id": rid,
        "state": {"image": ref},
        "questions": questions,
        "assets": assets,
        "targets": targets,
        "meta": {**meta, "width": image.width, "height": image.height},
        "expect": expect,
    }


# --- E0: contract checks -----------------------------------------------------


def e0_contract(ctx: Ctx) -> list[dict]:
    """Do the documented limits and errors behave as documented?"""
    scene = S.shapes_scene(1000)
    name = scene.objects[0].name
    q = {"target": box_q(f"the {name}")}
    jobs = []

    def add(case: str, documented: str, image: Image.Image | None, **kw) -> None:
        job = _image_job(ctx, "e0", f"e0-{case}", image, q, {}, {"case": case, "documented": documented}) if image else None
        if job is None:
            job = {"exp": "e0", "id": f"e0-{case}", "questions": q, "meta": {"case": case, "documented": documented}}
            job["state"] = {"image": "A kitchen counter with a coffee mug next to the sink."}
        job["kw"] = kw
        jobs.append(job)

    add("valid_1024", "200", S.render(scene))
    add("min_size_280", "200", S.render(scene, 280 / 1024))
    add("no_images_header", "400", S.render(scene), images_header=False)
    add("too_small_279", "400", S.render(scene, 279 / 1024))
    add("too_large_1569", "413", S.render(scene, 1569 / 1024))
    add("box_on_jev_1_13", "error (Box needs Jev 1.14)", S.render(scene), model="jev-1.13.0")
    add("box_without_any_image", "error (Box needs an image)", None)
    # An asset that nothing references.
    stray = _image_job(ctx, "e0", "e0-unreferenced_asset", S.render(scene), q, {}, {"case": "unreferenced_asset", "documented": "400"})
    stray["assets"].add(S.render(S.shapes_scene(1001), 0.3))
    jobs.append(stray)
    return ctx.run(jobs)


# --- E1: baseline accuracy ---------------------------------------------------


def e1_baseline(ctx: Ctx, n_scenes: int = 12) -> list[dict]:
    """Five distinct shapes per scene, one Box question each, one request per scene."""
    jobs = []
    for i in range(n_scenes):
        scene = S.shapes_scene(100 + i)
        questions = {f"o{k}": box_q(f"the {o.name}") for k, o in enumerate(scene.objects)}
        targets = {f"o{k}": {"gt": o.box, "name": o.name, "shape": o.shape} for k, o in enumerate(scene.objects)}
        jobs.append(_image_job(ctx, "e1", f"e1-{i:02d}", S.render(scene), questions, targets, {"seed": scene.seed}))
    return ctx.run(jobs)


# --- E2: target size, and the crop rescue ------------------------------------

DIAMETERS = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256]
RESCUE_MAX = 48
QUADS = {"top_left": (0, 0), "top_right": (1, 0), "bottom_left": (0, 1), "bottom_right": (1, 1)}


def _size_scene(seed: int, diameter: int) -> tuple[S.Scene, str]:
    rng = random.Random(seed)
    scene = S.Scene(1024, 1024, seed)
    quad = rng.choice(list(QUADS))
    qx, qy = QUADS[quad]
    region = (qx * 0.5, qy * 0.5, qx * 0.5 + 0.5, qy * 0.5 + 0.5)
    scene.objects.append(S.Obj("red", "circle", S.place(scene, rng, diameter, region=region)))
    combos = [(c, s) for c in S.COLORS if c not in ("red", "pink", "orange") for s in S.SHAPES]
    for color, shape in rng.sample(combos, 4):
        scene.objects.append(S.Obj(color, shape, S.place(scene, rng, rng.uniform(90, 150))))
    return scene, quad


def e2_size(ctx: Ctx, reps: int = 3) -> list[dict]:
    """A red circle from 10 to 256 px wide in a 1024 px image. For the small ones,
    also the documented fix: find the tile with a Choice, then Box on that tile."""
    jobs, scenes = [], {}
    for d in DIAMETERS:
        for r in range(reps):
            rid = f"e2-d{d:03d}-{r}"
            scene, quad = _size_scene(2000 + d * 10 + r, d)
            scenes[rid] = (scene, quad)
            target = scene.objects[0]
            questions = {"target": box_q("the red circle"), "visible": noul_q("a red circle")}
            targets = {"target": {"gt": target.box, "name": target.name}, "visible": {"p": 1.0, "present": True}}
            jobs.append(_image_job(ctx, "e2", rid, S.render(scene), questions, targets, {"diameter": d, "mode": "full"}))
    records = ctx.run(jobs)

    # Rescue step 1: which quadrant of a 2x master holds the target?
    step1 = []
    for rid, (scene, quad) in scenes.items():
        d = int(rid.split("-")[1][1:])
        if d > RESCUE_MAX:
            continue
        master = S.render(scene, 2.0)
        assets, tiles = Assets(), {}
        for name, (qx, qy) in QUADS.items():
            tiles[name] = assets.add(master.crop((qx * 1024, qy * 1024, qx * 1024 + 1024, qy * 1024 + 1024)))
        criteria = {name: f"The red circle is in `tiles.{name}`." for name in QUADS}
        criteria["none"] = "None of the tiles contains a red circle."
        step1.append(
            {
                "exp": "e2",
                "id": f"{rid}-tile",
                "state": {"tiles": tiles},
                "questions": {"tile": {"type": "choice", "instructions": "Which tile in `tiles` contains the red circle?", "criteria": criteria}},
                "assets": assets,
                "targets": {"tile": {"choice": quad}},
                "meta": {"diameter": d, "mode": "tile_choice", "parent": rid},
                "expect": {"tile": {"choice": quad}},
            }
        )
    tile_records = ctx.run(step1)
    records += tile_records

    # Rescue step 2: Box on the chosen tile; the analysis maps it back to the full image.
    step2 = []
    for rec in tile_records:
        picked = ((rec.get("answers") or {}).get("tile") or {}).get("choice")
        if picked not in QUADS:
            continue
        rid = rec["meta"]["parent"]
        scene, quad = scenes[rid]
        qx, qy = QUADS[picked]
        tile = S.render(scene, 2.0).crop((qx * 1024, qy * 1024, qx * 1024 + 1024, qy * 1024 + 1024))
        g = scene.objects[0].box
        gt_tile = [(g[0] - qx * 0.5) * 2, (g[1] - qy * 0.5) * 2, (g[2] - qx * 0.5) * 2, (g[3] - qy * 0.5) * 2]
        targets = {"target": {"gt": gt_tile, "gt_full": g, "tile": picked, "tile_correct": picked == quad}}
        job = _image_job(ctx, "e2", f"{rid}-crop", tile, {"target": box_q("the red circle")}, targets, {"diameter": rec["meta"]["diameter"], "mode": "crop_box", "parent": rid})
        step2.append(job)
    return records + ctx.run(step2)


# --- E3: absent targets ------------------------------------------------------


def e3_absent(ctx: Ctx, n_scenes: int = 20) -> list[dict]:
    """Box always answers. Ask for one thing that is there and one that is not,
    each with a presence Noul, and compare confidence and where the box lands."""
    jobs = []
    for i in range(n_scenes):
        rng = random.Random(3000 + i)
        colors = rng.sample(list(S.COLORS), 5)
        shapes = rng.sample(S.SHAPES, 3)
        scene = S.shapes_scene(3000 + i, colors=colors, shapes=shapes)
        used = {(o.color, o.shape) for o in scene.objects}
        used_c, used_s = {c for c, _ in used}, {s for _, s in used}
        kind = "near" if i % 2 == 0 else "far"
        if kind == "near":  # colour and shape both appear, but never together
            pool = [(c, s) for c in used_c for s in used_s if (c, s) not in used]
        else:  # neither the colour nor the shape appears anywhere
            pool = [(c, s) for c in S.COLORS if c not in used_c for s in S.SHAPES if s not in used_s]
        ac, ash = rng.choice(sorted(pool))
        present = rng.choice(scene.objects)
        absent = f"{ac} {ash}"
        questions = {
            "present_box": box_q(f"the {present.name}"),
            "present_noul": noul_q(a_an(present.name)),
            "absent_box": box_q(f"the {absent}"),
            "absent_noul": noul_q(a_an(absent)),
        }
        objects = [{"name": o.name, "color": o.color, "shape": o.shape, "box": o.box} for o in scene.objects]
        targets = {
            "present_box": {"gt": present.box, "name": present.name},
            "present_noul": {"p": 0.97, "present": True},
            "absent_box": {"name": absent, "color": ac, "shape": ash, "absent": True},
            "absent_noul": {"p": 0.04, "present": False},
        }
        jobs.append(_image_job(ctx, "e3", f"e3-{i:02d}", S.render(scene), questions, targets, {"kind": kind, "objects": objects}))
    return ctx.run(jobs)


# --- E4: two of the same thing -----------------------------------------------


def _twins_scene(seed: int) -> tuple[S.Scene, str]:
    for attempt in range(200):
        rng = random.Random(seed * 1000 + attempt)
        scene = S.Scene(1024, 1024, seed)
        d = rng.uniform(120, 180)
        left = S.Obj("red", "circle", S.place(scene, rng, d, region=(0.02, 0.05, 0.42, 0.95)))
        scene.objects.append(left)
        right = S.Obj("red", "circle", S.place(scene, rng, d, region=(0.58, 0.05, 0.98, 0.95)))
        scene.objects.append(right)
        side = rng.choice(["left", "right"])
        region = (0.0, 0.0, 0.5, 1.0) if side == "left" else (0.5, 0.0, 1.0, 1.0)
        square = S.Obj("blue", "square", S.place(scene, rng, rng.uniform(90, 130), region=region))
        scene.objects.append(square)

        def dist(a: S.Obj, b: S.Obj) -> float:
            return math.hypot((a.box[0] + a.box[2] - b.box[0] - b.box[2]) / 2, (a.box[1] + a.box[3] - b.box[1] - b.box[3]) / 2)

        near, far = (left, right) if side == "left" else (right, left)
        if dist(square, near) > 0.6 * dist(square, far):
            continue
        for color, shape in rng.sample([("green", "triangle"), ("yellow", "star"), ("purple", "square"), ("black", "triangle")], 2):
            scene.objects.append(S.Obj(color, shape, S.place(scene, rng, rng.uniform(90, 140))))
        return scene, side
    raise RuntimeError("no twins layout")


def e4_twins(ctx: Ctx, n_scenes: int = 10) -> list[dict]:
    """Two identical red circles. What does an ambiguous Box do, and do words fix it?"""
    jobs = []
    for i in range(n_scenes):
        scene, side = _twins_scene(4000 + i)
        left, right = scene.objects[0].box, scene.objects[1].box
        questions = {
            "ambiguous": box_q("the red circle"),
            "leftmost": box_q("the leftmost red circle"),
            "rightmost": box_q("the rightmost red circle"),
            "near_square": box_q("the red circle nearest the blue square"),
            "count": {
                "type": "choice",
                "instructions": "How many red circles are in `image`?",
                "criteria": {"none": "No red circle.", "one": "Exactly one red circle.", "several": "Two or more red circles."},
            },
        }
        targets = {
            "ambiguous": {"left": left, "right": right, "gt": left},
            "leftmost": {"gt": left, "other": right},
            "rightmost": {"gt": right, "other": left},
            "near_square": {"gt": left if side == "left" else right, "other": right if side == "left" else left},
            "count": {"choice": "several"},
        }
        jobs.append(_image_job(ctx, "e4", f"e4-{i:02d}", S.render(scene), questions, targets, {"near_side": side}))
    return ctx.run(jobs)


# --- E5: fields on a document ------------------------------------------------


def e5_document(ctx: Ctx, n_docs: int = 8) -> list[dict]:
    """Seven named fields on a rendered invoice, from a small text value to the whole table."""
    jobs = []
    for i in range(n_docs):
        image, gt = S.invoice(5000 + i)
        questions = {k: box_q(v, "invoice") for k, v in S.INVOICE_QUESTIONS.items()}
        targets = {k: {"gt": gt[k], "field": k} for k in questions}
        job = _image_job(ctx, "e5", f"e5-{i:02d}", image, questions, targets, {"seed": 5000 + i})
        job["state"] = {"invoice": job["state"]["image"]}
        jobs.append(job)
    return ctx.run(jobs)


# --- E6: resolution, aspect ratio, tokens, repeatability ---------------------

SQUARES = [280, 392, 560, 784, 1120, 1568]


def e6_operational(ctx: Ctx) -> list[dict]:
    jobs = []

    def scene_job(rid: str, scene: S.Scene, scale: float, meta: dict, n_questions: int | None = None) -> dict:
        n = n_questions or len(scene.objects)
        questions = {f"q{k}": box_q(f"the {scene.objects[k % len(scene.objects)].name}") for k in range(n)}
        targets = {f"q{k}": {"gt": scene.objects[k % len(scene.objects)].box} for k in range(n)}
        return _image_job(ctx, "e6", rid, S.render(scene, scale), questions, targets, meta)

    for s in range(3):  # the same three layouts at six sizes
        scene = S.shapes_scene(6000 + s)
        for size in SQUARES:
            jobs.append(scene_job(f"e6-res-{size:04d}-{s}", scene, size / 1024, {"part": "resolution", "size": size}))
    for s in range(3):  # wide and tall layouts
        jobs.append(scene_job(f"e6-wide-{s}", S.shapes_scene(6100 + s, width=1568, height=560), 1.0, {"part": "aspect", "aspect": "1568x560"}))
        jobs.append(scene_job(f"e6-tall-{s}", S.shapes_scene(6200 + s, width=560, height=1568), 1.0, {"part": "aspect", "aspect": "560x1568"}))
    base = S.shapes_scene(6300)
    for n in (1, 5, 10, 20):  # how tokens and latency grow with questions on one image
        jobs.append(scene_job(f"e6-nq-{n:02d}", base, 1.0, {"part": "questions", "n_questions": n}, n))
    records = ctx.run(jobs)
    for r in range(5):  # the identical request five times, one after another
        records += ctx.run([scene_job(f"e6-repeat-{r}", base, 1.0, {"part": "repeat", "run": r})])
    return records


# --- E7: the target as a picture ---------------------------------------------


def e7_reference(ctx: Ctx, n_scenes: int = 8) -> list[dict]:
    """Find a patterned tile from a reference picture, and from a description in words."""
    jobs = []
    for i in range(n_scenes):
        scene = S.tiles_scene(7000 + i)
        rng = random.Random(7000 + i)
        picks = rng.sample(range(len(scene.objects)), 2)
        assets = Assets()
        image = S.render(scene)
        ctx.sample("e7", f"e7-{i:02d}", image)
        ref = assets.add(image)
        questions, targets, expect = {}, {}, {}
        for k, idx in enumerate(picks):
            o = scene.objects[idx]
            pic = assets.add(S.tile_reference(o, 7000 + i * 10 + k))
            questions[f"picture{k}"] = {"type": "box", "instructions": f"the tile in `image` that matches this picture: {pic}"}
            questions[f"words{k}"] = box_q(S.tile_words(o))
            for q, mode in ((f"picture{k}", "picture"), (f"words{k}", "words")):
                targets[q] = {"gt": o.box, "mode": mode, "name": S.tile_words(o)}
                expect[q] = {"gt": o.box}
        jobs.append(
            {
                "exp": "e7",
                "id": f"e7-{i:02d}",
                "state": {"image": ref},
                "questions": questions,
                "assets": assets,
                "targets": targets,
                "meta": {"width": image.width, "height": image.height},
                "expect": expect,
            }
        )
    return ctx.run(jobs)


EXPERIMENTS = {
    "e0": e0_contract,
    "e1": e1_baseline,
    "e2": e2_size,
    "e3": e3_absent,
    "e4": e4_twins,
    "e5": e5_document,
    "e6": e6_operational,
    "e7": e7_reference,
}

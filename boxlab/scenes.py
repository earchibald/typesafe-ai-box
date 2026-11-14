"""Synthetic test images with exact ground-truth boxes.

Every object is drawn to fill a known rectangle, so the ground truth is the
rectangle itself. Boxes are stored as fractions of the image: [left, top, right, bottom].
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from PIL import Image, ImageChops, ImageDraw, ImageFont

COLORS = {
    "red": (214, 40, 40),
    "blue": (38, 84, 214),
    "green": (34, 150, 70),
    "yellow": (240, 200, 20),
    "purple": (130, 60, 180),
    "orange": (240, 130, 30),
    "black": (25, 25, 28),
    "pink": (240, 120, 180),
}
SHAPES = ["circle", "square", "triangle", "star"]
SS = 3  # supersampling factor for smooth edges


@dataclass
class Obj:
    color: str
    shape: str
    box: list[float]  # fractions: left, top, right, bottom
    pattern: str | None = None
    color2: str | None = None

    @property
    def name(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass
class Scene:
    width: int
    height: int
    seed: int
    objects: list[Obj] = field(default_factory=list)


def _background(w: int, h: int, seed: int) -> Image.Image:
    """Light paper-like background: a soft gradient plus faint noise."""
    rng = random.Random(seed)
    top = tuple(rng.randint(232, 246) for _ in range(3))
    bot = tuple(rng.randint(214, 232) for _ in range(3))
    grad = Image.linear_gradient("L").resize((w, h))
    img = Image.composite(Image.new("RGB", (w, h), bot), Image.new("RGB", (w, h), top), grad)
    noise = Image.effect_noise((w, h), 9).convert("RGB")
    return ImageChops.blend(img, noise, 0.06)


def _star_points(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = 1.0 if i % 2 == 0 else 0.42
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    return [
        (x0 + (px - minx) / (maxx - minx) * (x1 - x0), y0 + (py - miny) / (maxy - miny) * (y1 - y0))
        for px, py in pts
    ]


def _draw_pattern(draw: ImageDraw.ImageDraw, o: Obj, x0: float, y0: float, x1: float, y1: float) -> None:
    c1, c2 = COLORS[o.color], COLORS[o.color2 or "black"]
    draw.rectangle([x0, y0, x1, y1], fill=c1)
    w, h = x1 - x0, y1 - y0
    n = 6
    if o.pattern == "horizontal stripes":
        for i in range(1, n, 2):
            draw.rectangle([x0, y0 + h * i / n, x1, y0 + h * (i + 1) / n], fill=c2)
    elif o.pattern == "vertical stripes":
        for i in range(1, n, 2):
            draw.rectangle([x0 + w * i / n, y0, x0 + w * (i + 1) / n, y1], fill=c2)
    elif o.pattern == "polka dots":
        for i in range(3):
            for j in range(3):
                cx, cy = x0 + w * (i + 0.5) / 3, y0 + h * (j + 0.5) / 3
                r = w / 10
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c2)
    elif o.pattern == "checkerboard":
        m = 4
        for i in range(m):
            for j in range(m):
                if (i + j) % 2:
                    draw.rectangle(
                        [x0 + w * i / m, y0 + h * j / m, x0 + w * (i + 1) / m, y0 + h * (j + 1) / m], fill=c2
                    )


def render(scene: Scene, scale: float = 1.0) -> Image.Image:
    """Render the scene at `scale` times its nominal size."""
    w, h = round(scene.width * scale), round(scene.height * scale)
    big = _background(w, h, scene.seed).resize((w * SS, h * SS), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(big)
    for o in scene.objects:
        x0, y0, x1, y1 = (
            o.box[0] * w * SS,
            o.box[1] * h * SS,
            o.box[2] * w * SS - 1,
            o.box[3] * h * SS - 1,
        )
        fill = COLORS[o.color]
        if o.pattern:
            _draw_pattern(draw, o, x0, y0, x1, y1)
        elif o.shape == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=fill)
        elif o.shape == "square":
            draw.rectangle([x0, y0, x1, y1], fill=fill)
        elif o.shape == "triangle":
            draw.polygon([((x0 + x1) / 2, y0), (x0, y1), (x1, y1)], fill=fill)
        elif o.shape == "star":
            draw.polygon(_star_points(x0, y0, x1, y1), fill=fill)
    return big.resize((w, h), Image.Resampling.LANCZOS)


def _overlaps(a: list[float], b: list[float], gap: float) -> bool:
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def place(
    scene: Scene,
    rng: random.Random,
    size_px: float,
    *,
    region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    gap_px: float = 24,
    margin_px: float = 16,
) -> list[float]:
    """Find a free square of `size_px` (nominal pixels) inside `region` (fractions)."""
    fw, fh = size_px / scene.width, size_px / scene.height
    mx, my = margin_px / scene.width, margin_px / scene.height
    gap = gap_px / min(scene.width, scene.height)
    for _ in range(4000):
        lo_x, hi_x = region[0] + mx, region[2] - mx - fw
        lo_y, hi_y = region[1] + my, region[3] - my - fh
        if hi_x < lo_x or hi_y < lo_y:
            break
        x, y = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
        box = [x, y, x + fw, y + fh]
        if not any(_overlaps(box, o.box, gap) for o in scene.objects):
            return box
    raise RuntimeError("could not place object")


def shapes_scene(
    seed: int,
    n: int = 5,
    width: int = 1024,
    height: int = 1024,
    colors: list[str] | None = None,
    shapes: list[str] | None = None,
    size_range: tuple[float, float] = (0.09, 0.24),
) -> Scene:
    """`n` objects, each a unique colour and shape pair. Sizes are fractions of the short side."""
    rng = random.Random(seed)
    scene = Scene(width, height, seed)
    combos = [(c, s) for c in (colors or list(COLORS)) for s in (shapes or SHAPES)]
    rng.shuffle(combos)
    short = min(width, height)
    for color, shape in combos[:n]:
        size = rng.uniform(*size_range) * short
        scene.objects.append(Obj(color, shape, place(scene, rng, size)))
    return scene


PATTERNS = ["horizontal stripes", "vertical stripes", "polka dots", "checkerboard"]
PAIRS = [("blue", "yellow"), ("red", "black"), ("green", "pink"), ("purple", "orange"), ("yellow", "black")]


def tiles_scene(seed: int, n: int = 5, width: int = 1024, height: int = 1024) -> Scene:
    """Patterned square tiles, each a unique pattern and colour pair."""
    rng = random.Random(seed)
    scene = Scene(width, height, seed)
    combos = [(p, pair) for p in PATTERNS for pair in PAIRS]
    rng.shuffle(combos)
    for pattern, (c1, c2) in combos[:n]:
        size = rng.uniform(0.13, 0.22) * min(width, height)
        scene.objects.append(Obj(c1, "tile", place(scene, rng, size), pattern=pattern, color2=c2))
    return scene


def tile_reference(o: Obj, seed: int) -> Image.Image:
    """The tile on its own, as a 320 px reference picture."""
    ref = Scene(320, 320, seed + 9000)
    ref.objects.append(Obj(o.color, "tile", [0.16, 0.16, 0.84, 0.84], pattern=o.pattern, color2=o.color2))
    return render(ref)


def tile_words(o: Obj) -> str:
    return f"the square tile with {o.color} and {o.color2} {o.pattern}"


# --- Documents -------------------------------------------------------------


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default(size)


VENDORS = ["Northwind Supply Co.", "Harbor & Finch Ltd.", "Blue Mesa Tools", "Calder Paper Works", "Juniper Freight"]
CLIENTS = ["Avery Lindqvist", "Okafor Dental Group", "Marlowe Studio", "Tessel Robotics", "Pine Street Bakery"]
ITEMS = [
    "Shipping labels, roll of 500",
    "Packing tape, 6 pack",
    "Corrugated box, medium",
    "Thermal printer ribbon",
    "Pallet wrap, 18 inch",
    "Bubble mailer, size 4",
    "Desk scale, 30 kg",
    "Barcode scanner cable",
]


def invoice(seed: int, width: int = 1100, height: int = 1424) -> tuple[Image.Image, dict[str, list[float]]]:
    """A rendered invoice and the ground-truth box of each named field (fractions)."""
    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), (252, 251, 247))
    d = ImageDraw.Draw(img)
    gt: dict[str, list[float]] = {}

    def frac(b: tuple[float, float, float, float]) -> list[float]:
        return [b[0] / width, b[1] / height, b[2] / width, b[3] / height]

    def text(xy: tuple[float, float], s: str, font: ImageFont.FreeTypeFont, fill=(30, 30, 34), anchor: str = "la"):
        d.text(xy, s, font=font, fill=fill, anchor=anchor)
        return d.textbbox(xy, s, font=font, anchor=anchor)

    logo_left = rng.random() < 0.5
    m = 70
    # Logo: a coloured mark, top left or top right.
    lw, lh = 150, 110
    lx = m if logo_left else width - m - lw
    ly = 60 + rng.randint(0, 20)
    accent = COLORS[rng.choice(["blue", "green", "purple", "orange"])]
    d.rounded_rectangle([lx, ly, lx + lw, ly + lh], radius=18, fill=accent)
    d.ellipse([lx + 22, ly + 22, lx + 88, ly + 88], fill=(252, 251, 247))
    d.polygon([(lx + 84, ly + 90), (lx + 132, ly + 90), (lx + 108, ly + 30)], fill=(252, 251, 247))
    gt["logo"] = frac((lx, ly, lx + lw, ly + lh))

    vendor = rng.choice(VENDORS)
    vx = width - m if logo_left else m
    anchor = "ra" if logo_left else "la"
    text((vx, ly + 6), vendor, _font(34, True), anchor=anchor)
    text((vx, ly + 52), f"{rng.randint(10, 990)} Wharf Road, Unit {rng.randint(1, 40)}", _font(20), (90, 90, 96), anchor)
    text((vx, ly + 80), "accounts@example.com", _font(20), (90, 90, 96), anchor)

    y = ly + lh + 60
    text((m, y), "INVOICE", _font(46, True))
    number = f"INV-{rng.randint(10000, 99999)}"
    date = f"{rng.randint(1, 28):02d} {rng.choice(['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])} 2026"
    due = f"{rng.randint(1, 28):02d} Dec 2026"
    rx = width - m
    label_f, value_f = _font(22), _font(22, True)
    for i, (label, value, key) in enumerate(
        [("Invoice No:", number, "invoice_number"), ("Date:", date, "invoice_date"), ("Due:", due, None)]
    ):
        yy = y + i * 34
        vb = text((rx, yy), value, value_f, anchor="ra")
        text((vb[0] - 14, yy), label, label_f, (90, 90, 96), "ra")
        if key:
            gt[key] = frac(vb)

    y += 130
    text((m, y), "Bill to", _font(20), (90, 90, 96))
    b0 = text((m, y + 30), rng.choice(CLIENTS), _font(24, True))
    b1 = text((m, y + 62), f"{rng.randint(1, 400)} Larch Avenue", _font(22))
    b2 = text((m, y + 92), f"Portland, OR 97{rng.randint(201, 299)}", _font(22))
    gt["billing_address"] = frac((min(b0[0], b1[0], b2[0]), b0[1], max(b0[2], b1[2], b2[2]), b2[3]))

    # Line items table.
    y += 170
    cols = [m, 640, 780, width - m]
    table_top = y
    d.rectangle([m, y, width - m, y + 44], fill=(230, 229, 222))
    hf = _font(20, True)
    text((cols[0] + 12, y + 11), "Description", hf)
    text((cols[1] + 100, y + 11), "Qty", hf, anchor="ra")
    text((cols[2] + 100, y + 11), "Unit", hf, anchor="ra")
    text((cols[3] - 12, y + 11), "Amount", hf, anchor="ra")
    y += 44
    subtotal = 0.0
    rf = _font(21)
    for item in rng.sample(ITEMS, rng.randint(3, 7)):
        qty, unit = rng.randint(1, 12), rng.choice([4.5, 7.25, 12.0, 18.4, 26.0, 64.9])
        amount = qty * unit
        subtotal += amount
        text((cols[0] + 12, y + 12), item, rf)
        text((cols[1] + 100, y + 12), str(qty), rf, anchor="ra")
        text((cols[2] + 100, y + 12), f"{unit:,.2f}", rf, anchor="ra")
        text((cols[3] - 12, y + 12), f"{amount:,.2f}", rf, anchor="ra")
        y += 46
        d.line([m, y, width - m, y], fill=(214, 213, 206), width=1)
    gt["line_items_table"] = frac((m, table_top, width - m, y))

    tax = round(subtotal * 0.08, 2)
    total = subtotal + tax
    y += 28
    for label, value in [("Subtotal", subtotal), ("Tax (8%)", tax)]:
        text((cols[2] + 100, y), label, _font(21), (90, 90, 96), "ra")
        text((cols[3] - 12, y), f"${value:,.2f}", _font(21), anchor="ra")
        y += 36
    y += 8
    d.line([cols[1], y, width - m, y], fill=(30, 30, 34), width=2)
    y += 16
    text((cols[2] + 100, y), "TOTAL DUE", _font(24, True), anchor="ra")
    tb = text((cols[3] - 12, y), f"${total:,.2f}", _font(28, True), anchor="ra")
    gt["total_due"] = frac(tb)

    # Signature: a handwritten-looking scribble above a rule.
    sig_left = rng.random() < 0.5
    sw, sh = 300, 90
    sx = m + 20 if sig_left else width - m - sw - 20
    sy = min(y + 150, height - 220)
    pts = []
    for i in range(60):
        t = i / 59
        pts.append(
            (
                sx + t * sw,
                sy + sh / 2 + math.sin(t * rng.uniform(16, 22)) * sh * 0.42 * (1 - 0.5 * t) + rng.uniform(-3, 3),
            )
        )
    d.line(pts, fill=(20, 30, 110), width=3, joint="curve")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    gt["signature"] = frac((min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2))
    d.line([sx - 10, sy + sh + 14, sx + sw + 10, sy + sh + 14], fill=(90, 90, 96), width=1)
    text((sx - 10, sy + sh + 22), "Authorised signature", _font(18), (90, 90, 96))
    text((m, height - 70), "Payment within 30 days. Thank you for your business.", _font(18), (120, 120, 126))
    return img, gt


INVOICE_QUESTIONS = {
    "invoice_number": "the invoice number value (the number itself, not its label)",
    "invoice_date": "the invoice date value (the date itself, not its label, and not the due date)",
    "total_due": "the total amount due (the final total figure, not the subtotal or the tax)",
    "logo": "the company logo",
    "signature": "the handwritten signature",
    "line_items_table": "the line items table, including its header row",
    "billing_address": "the bill-to name and address block",
}

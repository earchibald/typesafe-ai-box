"""Voice podcast/script.txt with the macOS `say` command and stitch it with ffmpeg.

    uv run podcast/build.py

Each line of the script is "A: ..." or "B: ...". Needs macOS and ffmpeg.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICES = {"A": "Samantha", "B": "Daniel"}
RATE = "172"  # words per minute
GAP = 0.38  # seconds of silence between turns
OUT = HERE / "typesafe-box-podcast.mp3"


def main() -> None:
    turns = []
    for line in (HERE / "script.txt").read_text().splitlines():
        if line.strip():
            who, _, text = line.partition(":")
            turns.append((who.strip(), text.strip()))
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        gap = t / "gap.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", str(GAP), gap],
            check=True,
        )
        parts = []
        for i, (who, text) in enumerate(turns):
            aiff, wav = t / f"{i:03d}.aiff", t / f"{i:03d}.wav"
            subprocess.run(["say", "-v", VOICES[who], "-r", RATE, "-o", str(aiff), text], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "22050", "-ac", "1", str(wav)], check=True)
            parts += [wav, gap]
        listing = t / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-ac", "1", "-b:a", "96k",
                "-metadata", "title=Testing TypeSafe's Box", "-metadata", "artist=boxlab",
                str(OUT),
            ],
            check=True,
        )
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(OUT)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"{OUT.name}: {float(dur) / 60:.2f} minutes, {len(turns)} turns, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

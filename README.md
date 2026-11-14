# boxlab

A small experiment suite for TypeSafe's [Box](https://docs.typesafe.ai/primitives/box) question type
(Jev 1.14): where is this thing in the image?

Every test image is drawn by the suite, so the true box of every target is known exactly.

## Run it

```sh
echo 'TYPESAFE_API_KEY=...' > .env   # or export it; .env is gitignored
uv run run.py                        # all experiments, about 160 requests
uv run analyze.py                    # writes results/summary.json and results/overlays/
```

`uv run run.py --only e1,e3` runs a subset, `--force` reruns, and `--mock` exercises the
pipeline with fabricated answers (written to `results-mock/`, never to `results/`).

## Experiments

| | Experiment | Question |
| --- | --- | --- |
| e0 | Contract checks | Do the documented limits and errors hold? |
| e1 | Baseline accuracy | How tight is the box on an easy, unambiguous target? |
| e2 | Target size | When are small targets missed, and does crop-then-Box recover them? |
| e3 | Absent targets | What does Box return for something that is not there? Does a Noul catch it? |
| e4 | Twins | Where does an ambiguous box land, and do words fix it? |
| e5 | Document fields | Is a box on a page tight enough to crop a field? |
| e6 | Operations | Image size, aspect ratio, question count, tokens, latency, repeatability |
| e7 | Picture as target | Does a reference picture work as well as a description? |

Scoring is intersection over union (IoU) against the true box; 0.5 or more is a hit.

## Layout

- `boxlab/scenes.py`: scene and invoice generators with ground truth
- `boxlab/api.py`: HTTP client for `POST /v1/systemone` (base64 images, retries), and the mock
- `boxlab/experiments.py`: the eight experiments
- `run.py`, `analyze.py`: runner and grading
- `results/raw/*.jsonl`: one record per request, with targets, answers, usage and latency

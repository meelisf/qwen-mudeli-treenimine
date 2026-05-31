#!/usr/bin/env python3
"""
OCR mudeli hindamine ground-truth vastu (CER ja WER)

Käivitamine:
  python scripts/eval_model.py models/qwen3.5-ocr-lora
  python scripts/eval_model.py models/qwen3.5-ocr-markup-20260526
  python scripts/eval_model.py --all          # kõik data/test/ alamkaustad

GT failid: data/test/ground-truth/<pildi-nimi>.txt
Mudeli väljund: data/test/<mudeli-nimi>/<pildi-nimi>.txt

pip install editdistance
"""

import sys
import editdistance
from pathlib import Path

GT_DIR = Path("data/test/ground-truth")


def cer(gt: str, hyp: str) -> float:
    gt = gt.strip()
    hyp = hyp.strip()
    if not gt:
        return 0.0 if not hyp else 1.0
    return editdistance.eval(gt, hyp) / len(gt)


def wer(gt: str, hyp: str) -> float:
    gt_words = gt.strip().split()
    hyp_words = hyp.strip().split()
    if not gt_words:
        return 0.0 if not hyp_words else 1.0
    return editdistance.eval(gt_words, hyp_words) / len(gt_words)


def eval_model(model_output_dir: Path) -> dict:
    results = {}
    gt_files = sorted(GT_DIR.glob("*.txt"))

    for gt_file in gt_files:
        gt_text = gt_file.read_text(encoding="utf-8").strip()
        if not gt_text:
            continue

        hyp_file = model_output_dir / gt_file.name
        if not hyp_file.exists():
            results[gt_file.stem] = {"cer": None, "wer": None, "status": "puudub"}
            continue

        hyp_text = hyp_file.read_text(encoding="utf-8").strip()
        results[gt_file.stem] = {
            "cer": cer(gt_text, hyp_text),
            "wer": wer(gt_text, hyp_text),
            "status": "ok",
        }

    return results


def print_results(model_name: str, results: dict):
    print(f"\n{'='*60}")
    print(f"Mudel: {model_name}")
    print(f"{'='*60}")
    print(f"{'Fail':<45} {'CER':>6}  {'WER':>6}  {'Märkus'}")
    print(f"{'-'*60}")

    cer_vals = []
    wer_vals = []

    for stem, r in sorted(results.items()):
        if r["status"] == "puudub":
            print(f"  {stem:<43} {'—':>6}  {'—':>6}  (väljund puudub)")
            continue
        cer_pct = r["cer"] * 100
        wer_pct = r["wer"] * 100
        cer_vals.append(r["cer"])
        wer_vals.append(r["wer"])
        print(f"  {stem:<43} {cer_pct:>5.1f}%  {wer_pct:>5.1f}%")

    if cer_vals:
        avg_cer = sum(cer_vals) / len(cer_vals) * 100
        avg_wer = sum(wer_vals) / len(wer_vals) * 100
        print(f"{'-'*60}")
        print(f"  {'Keskmine':<43} {avg_cer:>5.1f}%  {avg_wer:>5.1f}%  ({len(cer_vals)} faili)")
    else:
        print("  (ei leitud ühtegi võrreldavat faili)")


def find_model_dirs():
    test_dir = Path("data/test")
    return [
        d for d in sorted(test_dir.iterdir())
        if d.is_dir() and d.name != "ground-truth"
    ]


# ---------------------------------------------------------------------------
# Argumendid
# ---------------------------------------------------------------------------

args = sys.argv[1:]

if not GT_DIR.exists():
    print(f"Viga: GT kaust puudub: {GT_DIR}")
    sys.exit(1)

if "--all" in args:
    model_dirs = find_model_dirs()
    if not model_dirs:
        print("Ei leitud ühtegi mudeli väljundkausta data/test/ alt.")
        sys.exit(1)
    for d in model_dirs:
        print_results(d.name, eval_model(d))
elif args:
    model_path = Path(args[0])
    model_name = model_path.name
    output_dir = Path("data/test") / model_name
    if not output_dir.exists():
        print(f"Viga: mudeli väljundkaust puudub: {output_dir}")
        print(f"  Käivita kõigepealt: python scripts/test_model.py --model {model_path}")
        sys.exit(1)
    print_results(model_name, eval_model(output_dir))
else:
    print(__doc__)
    sys.exit(0)

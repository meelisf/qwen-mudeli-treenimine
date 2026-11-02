"""
Koondab data/raw/ alamkataloogidest jpg+txt paarid
data/processed/images/ kausta ja loob metadata.csv.

Ignoreerib: _thumbs/, .json, .backup.*, .zip
"""
import os
import csv
import shutil

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
IMAGES_DIR = os.path.join(OUT_DIR, "images")
CSV_PATH = os.path.join(OUT_DIR, "metadata.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)

rows = []

for doc_dir in sorted(os.listdir(RAW_DIR)):
    doc_path = os.path.join(RAW_DIR, doc_dir)
    if not os.path.isdir(doc_path):
        continue
    if doc_dir.startswith("_"):
        continue

    for fname in sorted(os.listdir(doc_path)):
        if not fname.endswith(".jpg"):
            continue

        jpg_src = os.path.join(doc_path, fname)
        txt_src = os.path.join(doc_path, fname.replace(".jpg", ".txt"))

        if not os.path.exists(txt_src):
            print(f"  HOIATUS: txt puudub - {txt_src}")
            continue

        transcription = open(txt_src, encoding="utf-8").read().strip()
        if not transcription:
            print(f"  HOIATUS: tühi transkriptsioon - {txt_src}")
            continue

        dest_name = f"{doc_dir}__{fname}"
        jpg_dest = os.path.join(IMAGES_DIR, dest_name)
        shutil.copy2(jpg_src, jpg_dest)

        rows.append({"failinimi": f"images/{dest_name}", "transkriptsioon": transcription})

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["failinimi", "transkriptsioon"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nValmis: {len(rows)} lehekülge -> {CSV_PATH}")

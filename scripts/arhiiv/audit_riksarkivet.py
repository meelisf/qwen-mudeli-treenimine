#!/usr/bin/env python3
"""Riksarkiveti _seg allikate tühjade TextLine'ide audit (2026-10-04).

Sama viga mis xix-is (vt tuhjad_read_audit.py): `parse_pagexml` jätab
teksti(ta) TextLine'i vaikselt vahele. Riksarkiveti failinimed on andmestikus
lõigatud 60 märgini (lehe ID võib kaduda) → treeningrida seotakse XML-iga
TEKSTI järgi: XML parsitakse sama `parse_pagexml`-iga ja tulemust võrreldakse
metadata.csv tekstiga.

Laadib alla AINULT `data/page_xmls/*.tar.gz` (pilte mitte), vahemällu
data/raw_xml/riksarkivet/.
Väljund: data/kurrent_xix_audit/tuhjad_read_riksarkivet.csv
Käivitus: venv/bin/python scripts/audit_riksarkivet.py
"""
import csv, re, sys, tarfile
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = sys.argv[:1]                       # build-skript loeb argv-d impordil
from build_riksarkivet_dataset import parse_pagexml                  # noqa: E402

ALLIKAD = {                                   # allikas v4-s → HF dataset
    "svea_hovratt_seg": "svea_hovratt_seg",
    "trolldomskommissionen_seg": "trolldomskommissionen_seg",
    "krigshovrattens_seg": "krigshovrattens_dombocker_seg",
    "bergskollegium_rel_seg": "bergskollegium_relationer_och_skrivelser_seg",
    "bergskollegium_adv_seg": "bergskollegium_advokatfiskalskontoret_seg",
    "gota_hovratt_seg": "gota_hovratt_seg",
    "jonkopings_seg": "jonkopings_radhusratts_och_magistrat_seg",
}
META = Path("data/kurrent_v4/metadata.csv")
CACHE = Path("data/raw_xml/riksarkivet")
OUT = Path("data/kurrent_xix_audit/tuhjad_read_riksarkivet.csv")
LINE = re.compile(r"<TextLine\b.*?</TextLine>", re.S)
WORD = re.compile(r"<Word\b.*?</Word>", re.S)
UNI = re.compile(r"<TextEquiv[^>]*>\s*<Unicode>(.*?)</Unicode>", re.S)

csv.field_size_limit(sys.maxsize)


def tuhje(xml):
    n = e = 0
    for tl in LINE.findall(xml):
        u = UNI.findall(WORD.sub("", tl))
        n += 1
        e += not (u and u[-1].strip())
    return n, e


def main():
    token = get_token()
    api = HfApi()
    rows = defaultdict(dict)                  # allikas → tekst → failinimi
    for r in csv.DictReader(open(META, encoding="utf-8")):
        if r["allikas"] in ALLIKAD:
            rows[r["allikas"]][r["transkriptsioon"]] = r["failinimi"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["allikas", "dataset", "xml", "ridu", "tuhje", "treeningus"])
        for allikas, ds in ALLIKAD.items():
            repo = f"Riksarkivet/{ds}"
            tars = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", token=token)
                          if f.startswith("data/page_xmls/") and f.endswith(".tar.gz"))
            leitud, st = set(), Counter()
            for t in tars:
                p = hf_hub_download(repo, t, repo_type="dataset", token=token, cache_dir=CACHE)
                with tarfile.open(p, "r:gz") as tf:
                    for m in tf.getmembers():
                        if not m.name.endswith(".xml"):
                            continue
                        x = tf.extractfile(m).read().decode("utf-8", errors="replace")
                        n, e = tuhje(x)
                        text = parse_pagexml(x)
                        trn = rows[allikas].get(text, "") if text else ""
                        if trn:
                            leitud.add(trn)
                            st["≥2 tühja" if e >= 2 else "0–1 tühja"] += 1
                        w.writerow([allikas, ds, m.name, n, e, trn])
            print(f"{allikas}: tar {len(tars)}, v4-s {len(rows[allikas])}, seotud {len(leitud)}  {dict(st)}", flush=True)


if __name__ == "__main__":
    main()

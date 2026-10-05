"""GT-kontrolli kandidaatide järjekorrast sõltumatu sõelumine (2026-10-05).

gt_kontroll.py CER on ridade järjekorrale tundlik: mitmeveeruline leht,
marginaalia või tabel annab kõrge CER-i ka siis, kui sisu klapib. Siin
võrreldakse GT ja mudeli väljundi SÕNAHULKA (recall = GT sõnadest väljundis,
precision = väljundi sõnadest GT-s) ja klassifitseeritakse:

  sisu klapib    rec ≥ .75 ja prec ≥ .75   → jääb (järjekord / raske käsi)
  GT puudulik    prec < .6, rec ≥ .6        → välja (GT katab osa lehest)
  osaline GT     rec ≥ .95, prec < .75      → välja
  erinev         rec < .6 ja prec < .6      → välja (enamasti toores HTR-GT;
                                              mõni mudeli hallutsinatsioon kaob kaasa)
  loop           mudeli rike, GT korras     → jääb
  vutt_horedad   ALATI jääb — tahtlikult peaaegu tühjad lehed

Väljund: <gt_kontroll kaust>/sonakate.csv ja valja.csv.
Käivitus: venv/bin/python scripts/gt_sonakate.py
"""
import collections
import csv
import re
import unicodedata

D = "data/kurrent/gt_kontroll/kurrent-20261002-Q8_0/"
gt = {r["failinimi"].split("/")[-1]: r["transkriptsioon"]
      for r in csv.DictReader(open("data/kurrent/metadata.csv"))}
kand = [r for r in csv.DictReader(open(D + "kandidaadid.csv")) if r["failinimi"] in gt]


def sonad(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("¬\n", "").replace("-\n", "").replace("ſ", "s")
    return collections.Counter(re.findall(r"\w+", t))


def klass(allikas, rec, prec, liik):
    if liik == "loop":
        return "loop"
    if allikas == "vutt_horedad":
        return "kaitstud"
    if rec >= 0.75 and prec >= 0.75:
        return "sisu klapib"
    if rec < 0.6 and prec < 0.6:
        return "erinev"
    if prec < 0.6:
        return "GT puudulik"
    if rec >= 0.95 and prec < 0.75:
        return "osaline GT"
    return "piiripealne"


VALJA = {"erinev", "GT puudulik", "osaline GT"}
read = []
for r in kand:
    f = r["failinimi"]
    o = open(D + "valjundid/" + f.rsplit(".", 1)[0] + ".txt").read()
    G, O = sonad(gt[f]), sonad(o)
    yhine = sum((G & O).values())
    rec = yhine / max(1, sum(G.values()))
    prec = yhine / max(1, sum(O.values()))
    read.append([r["allikas"], f, float(r["cer_norm"]) * 100, rec, prec, r["liik"],
                 klass(r["allikas"], rec, prec, r["liik"])])

with open(D + "sonakate.csv", "w") as fh:
    w = csv.writer(fh)
    w.writerow(["allikas", "failinimi", "cer", "recall", "precision", "liik", "klass"])
    w.writerows(read)
with open(D + "valja.csv", "w") as fh:
    w = csv.writer(fh)
    w.writerow(["failinimi", "allikas", "pohjus"])
    w.writerows([x[1], x[0], x[6]] for x in read if x[6] in VALJA)

kokku = collections.Counter(x[6] for x in read)
print(f"Kandidaate metadata.csv-s: {len(read)}")
for k, n in kokku.most_common():
    print(f"  {k:12} {n}")
print(f"Välja: {sum(kokku[k] for k in VALJA)}  → {D}valja.csv")

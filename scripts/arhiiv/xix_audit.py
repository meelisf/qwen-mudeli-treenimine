"""kurrent-xix mitte-Zürichi projektide audit: dateerimine, tühjad lehed, duplikaadid.

Väljund:
  data/kurrent_xix_audit/pages.csv     — üks rida lehe kohta
  data/kurrent_xix_audit/projects.csv  — projekti kokkuvõte
  data/kurrent_xix_audit/dups.txt      — duplikaatgrupid (teksti- ja pildiräsi järgi)
  data/kurrent_xix_audit/samples/      — 2 näidispilti projekti kohta (pisipilt)
"""
import collections, csv, glob, hashlib, io, os, re, statistics
import pyarrow.parquet as pq
from PIL import Image

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--dh-unibe--image-text_kurrent-xix/snapshots/*/data/train"))[0]
OUT = "data/kurrent_xix_audit"
os.makedirs(OUT + "/samples", exist_ok=True)

LINE = re.compile(r"<TextLine\b.*?</TextLine>", re.S)
WORD = re.compile(r"<Word\b.*?</Word>", re.S)
UNI = re.compile(r"<TextEquiv[^>]*>\s*<Unicode>(.*?)</Unicode>", re.S)
YEAR = re.compile(r"(?<!\d)(1[6-9]\d\d)(?!\d)")


def lines(xml):
    out = []
    for tl in LINE.findall(xml):
        u = UNI.findall(WORD.sub("", tl))      # rea enda TextEquiv, mitte sõnade oma
        if u and u[-1].strip():
            out.append(u[-1].strip())
    return out


def norm_text(ls):
    return re.sub(r"\s+", " ", " ".join(ls)).strip().lower()


projects = sorted(p for p in os.listdir(SNAP) if not p.startswith("MM_"))
pages_f = open(OUT + "/pages.csv", "w", newline="")
pw = csv.writer(pages_f)
pw.writerow(["project", "filename", "n_lines", "n_chars", "years", "w", "h", "text_sha", "img_sha"])
summ = []
by_text = collections.defaultdict(list)
by_img = collections.defaultdict(list)

for proj in projects:
    n = 0; nl = []; years = []; sizes = collections.Counter(); samples = 0
    for f in sorted(glob.glob(f"{SNAP}/{proj}/*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16, columns=["xml_content", "filename", "image"]):
            for r in batch.to_pylist():
                n += 1
                ls = lines(r["xml_content"] or "")
                txt = norm_text(ls)
                b = r["image"]["bytes"]
                isha = hashlib.sha1(b).hexdigest()[:16]
                tsha = hashlib.sha1(txt.encode()).hexdigest()[:16] if len(ls) >= 5 else ""
                try:
                    im = Image.open(io.BytesIO(b)); w, h = im.size
                except Exception:
                    im = None; w = h = 0
                ys = [int(y) for y in YEAR.findall(" ".join(ls))]
                years += ys
                nl.append(len(ls))
                sizes["rõht" if w > h else "püst"] += 1
                pw.writerow([proj, r["filename"], len(ls), len(txt), " ".join(map(str, ys[:10])), w, h, tsha, isha])
                if tsha:
                    by_text[tsha].append(proj)
                by_img[isha].append(proj)
                if im is not None and len(ls) >= 5 and samples < 2 and n % 5 == 0:
                    im.thumbnail((900, 900))
                    im.convert("RGB").save(f"{OUT}/samples/{proj[:60]}_{samples}.jpg", quality=75)
                    samples += 1
    nonempty = sum(1 for x in nl if x >= 5)
    nimes = re.findall(r"(1[6-9]\d\d)", proj)
    y_med = int(statistics.median(years)) if years else None
    summ.append(dict(project=proj, pages=n, nonempty=nonempty,
                     lines_median=statistics.median(nl) if nl else 0,
                     year_in_name=nimes[0] if nimes else "",
                     year_text_median=y_med,
                     year_text_p10=sorted(years)[len(years) // 10] if years else None,
                     year_text_p90=sorted(years)[len(years) * 9 // 10] if years else None,
                     n_year_mentions=len(years), orient=dict(sizes)))
    print(proj, n, nonempty, y_med, flush=True)

pages_f.close()
with open(OUT + "/projects.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0]))
    w.writeheader(); w.writerows(summ)

# Duplikaatgrupid: projektipaarid, mis jagavad lehti
with open(OUT + "/dups.txt", "w") as f:
    for name, idx in (("TEKST", by_text), ("PILT", by_img)):
        pair = collections.Counter()
        for projs in idx.values():
            u = sorted(set(projs))
            if len(projs) > 1:
                if len(u) == 1:
                    pair[(u[0], u[0])] += len(projs) - 1        # projektisisene kordus
                for i in range(len(u)):
                    for j in range(i + 1, len(u)):
                        pair[(u[i], u[j])] += 1
        f.write(f"== {name}: projektipaarid jagatud lehtede arvuga\n")
        for (a, b), c in pair.most_common():
            f.write(f"{c:6d}  {a}  <->  {b}\n")
        uniq = len(idx)
        f.write(f"   unikaalseid {name.lower()}-räsisid: {uniq}\n\n")
print("VALMIS")

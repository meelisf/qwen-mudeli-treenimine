# Qwen3.5-9B OCR – juhised

Ajalooliste dokumentide OCR (ladina, kreeka, fraktuur, Kurrent käsikiri)
koos VUTT XML markup'iga. Suhtlus ja kommentaarid **eesti keeles**.

## Loe kõigepealt SPIKKER.md

**`SPIKKER.md` on täielik juhend** – käivitamine, torujuhe, andmete
puhastusahel, mudeli testimine ja aktiveerimine, temperatuuri jälgimine,
pärast-reebooti seadistused. Ära dubleeri seda infot siia; kui midagi
muutub, uuenda SPIKKER-it.

Andmestike sisukirjeldused: `docs/kurrent-andmestikud.md`,
`docs/truki-andmestikud.md`, `docs/masina-parameetrid.md`.

## Keskkond

Venv on `venv/`, pakid paigaldatakse **uv-ga**:

```bash
VIRTUAL_ENV=venv venv/bin/uv pip install <pakk>
```

Python 3.12, torch cu128 (RTX 5090). Pikk treening käib **tmux-is**.

## Kõvad reeglid – neid on lihtne valesti teha

- **Käsikirju ei tohi ajada trükimudeliga** ega vastupidi. Käsikiri
  trükimudelis jookseb tokenilakke – ~8 minutit raisku lehe kohta.
  Testpildid on lahus: `data/test/print/` ja `data/test/hand/`.
- **Treenitud checkpointist laadimisel ei kutsuta `get_peft_model()`
  uuesti** – LoRA adapterid on juba küljes.
- **Inferentsil bf16, mitte 4-bit.** 4-bit on treeningu mälusääst,
  inferentsil ainult aeglustab (mõõdetud: bf16 1.5x kiirem).
- **Image processor'i `longest_edge` on KOGU pikslite arv**, mitte serva
  pikkus. Vaikeväärtus 16M px tähendab OOM-i; kasuta 5 120 000.
  Valem: visuaaltokeneid ≈ pikslid / 1024.
- **Enne treeningut peata ocr-service** (`sudo systemctl stop ocr-service`),
  muidu GPU mälu ei jätku.

## Git

- `.env` sisaldab HF_TOKEN-it – **ei lähe kunagi repositooriumi**.
- `data/kurrent/*.csv` (41 MB) ja pildid on gitignore's: ilma piltideta
  pole CSV-dest kasu.
- SSH võti on parooliga; pärast reebooti tuleb see agenti laadida, muidu
  `git push` ja `vutt_sync.py` ei tööta. Käsk on SPIKKER-is.

## Andmete puhastus – oluline taust

`build_vutt_dataset.py` ja `train_markup.py` **muudavad transkriptsiooni
automaatselt** (märgendite lahtipakkimine, ristuva pesastuse parandus,
tühjade märgendite eemaldus). Treeningandmed ei ole seega bait-bait samad
mis VUTT-is. Ahela kirjeldus on SPIKKER-is, sammu 2 all.

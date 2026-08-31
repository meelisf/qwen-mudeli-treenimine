# Kurrendi treeningu öine järelahel (30./31.08.2026)

**Ülestähendus, et töö ei sõltuks ühestki elavast Claude-sessioonist.**
Kui sessioon on surnud või sa loed seda hommikul: tulemused on kettal,
midagi ei ole pooleli jäänud ootama inimest.

Ajur: `oo_eval.sh` (scratchpadis, vt allpool), käivitatud 30.08 16:23
`setsid`-iga lahti haagituna — elab üle tmux-akna sulgemise. Ta **ei aktiveeri
midagi tootmises** ega puutu systemd-teenuseid.

## Mis järjekorras käib

| # | Samm | Eeldatav aeg |
|---|---|---|
| 0 | Ootab treeningu pid 3121315 lõppu, siis 2 min GPU vabanemiseks | ~02:20 |
| 1 | `merge_lora` → `convert_hf_to_gguf --no-nextn` → `--mmproj` → `llama-quantize Q8_0 28` | ~02:25, 25 min |
| 2 | `llama-server` **pordil 8099** (`--image-max-tokens 5000`) + `eval_kurrent.py --endpoint` | ~02:50, 5 min |
| 3 | `eval_kurrent.py` transformers bf16 lõppmudelil | ~03:00, 25 min |
| 4 | `eval_kurrent.py` 1. epohhi adapteril (`models/kurrent-20260829-epohh1`) | ~03:25, 25 min |

Port 8099 on meelega vaba port, et mitte segada `llama-server-hand`-i (8081),
kui see vahepeal käima pannakse.

## Miks kaks hindamist samale mudelile

- **Samm 3 (transformers bf16)** on ainus aus võrdlus baseline'iga: vana mudel
  `qwen3.5-ocr-kurrent-20260602` mõõdeti 26.08 samamoodi, CER 13,9 %
  (mediaan 6,6 %), 2 loopi. Vt mälu `kurrent-holdout-baseline`.
- **Samm 2 (GGUF Q8_0)** on see, mis päriselt teenusesse läheks — ja
  **esimene Kurrendi mõõtmine `--image-max-tokens 5000` lipuga**. 27.08
  mootorivõrdlus tehti ilma selleta, ehk pilt kärbiti vaikselt 4096 tokenini;
  see number vajaski üle mõõtmist.

## Kus asjad on

```
data/kurrent/eval/kurrent-20260829-Q8_0/      # samm 2 + results.csv + run.json
data/kurrent/eval/qwen3.5-ocr-kurrent-20260829/   # samm 3
data/kurrent/eval/kurrent-20260829-epohh1/    # samm 4
models/qwen3.5-ocr-kurrent-20260829/          # LoRA adapter (treeningu väljund)
models/gguf/kurrent-20260829-Q8_0.gguf + mmproj-kurrent-20260829-F16.gguf
models/merged/qwen3.5-ocr-kurrent-20260829-bf16/  # vaheaste, võib kustutada
models/gguf/kurrent-20260829-BF16.gguf            # vaheaste, võib kustutada
```

Logid (scratchpad, kaob reboodiga — tõsta välja, kui vaja säilitada):
`/tmp/claude-1000/-home-mf-Dokumendid-LLM-qwen3-5/a0c7f425-daa4-463b-99a3-51c9b070a217/scratchpad/`
→ `oo_eval.log` (sammumarkerid), `gguf.log`,
`eval_gguf.log`, `eval_bf16.log`, `eval_ep1.log`, `server.log`.

## Kui midagi kukkus

`grep MARKER oo_eval.log` näitab, kui kaugele jõuti. Iga samm on eraldi
käivitatav, retseptid on SPIKKER-is; ükski hilisem samm ei sõltu teisest peale
selle, et GGUF-i vaja sammuks 2.

Kontroll, kas ajur veel elab: `pgrep -af oo_eval.sh` (pid 3207267).

## Mis jääb INIMESE otsustada — ahel neid EI tee

1. **Mudeli aktiveerimine.** `llama-server-hand.service` tee uuendus +
   `MODEL_CONFIGS["hand"]` failis `kataloogi-jalgimine-ja-ocr.py`. Nõuab sudot.
2. **Kolm teenust on 29.08-st maas** (`ocr-service`, `llama-server-print`,
   `llama-server-hand`) — treening vajas GPU-d. Hommikul tuleb need käsitsi
   üles panna, muidu VUTT-i OCR ei tööta.
3. **Silmaga vaatamine.** Numbrid ei ole otsus; CER mõõdab vastavust
   arhiivikorpuse tavadele, mitte kasulikkust VUTT-is.
4. **Vahefailide kustutamine** (~35 GB, `models/merged/*-bf16` +
   `*-BF16.gguf`) — taastuvad adapterist mõne minutiga.
5. **`data/kurrent/metadata.csv` kosmeetiline parandus** — 2 039 real puudub
   kolmas veerg `allikas`. Nüüd on ohutu, treening on läbi.

## Mida tulemustest vaadata

- **Need 23 lehte, mida vana mudel treeningul ei näinud** (aaeb, hanse,
  dresdner_1665, senatsprotokolle) — ainus koht, kus võrdlus on tõeliselt aus.
  Vanal mudelil neil CER keskm 22,6 %, mediaan 8,6 %.
- **`ratio` veerg** — loobid. Vanal 2 lehte üle 1,4; `senatsp_UAT_047_19_017`
  ratio 2,93. Kui uus mudel selle ära parandab, on senatsprotokolle'i
  lisamine (229 lk) end ära tasunud.
- **Epohh 1 vs 2** — 20260602 puhul andis teine epohh vähe. Kui vahe on väike,
  saab järgmise jooksu poole lühemaks.

**Ajuri koopia repos:** `docs/arhiiv/kurrent-20260829-oine-ahel.sh` (scratchpad kaob reboodiga).

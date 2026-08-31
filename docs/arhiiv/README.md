# Arhiiv

Siin on lõpetatud uurimused ja täidetud plaanid. **Neid ei uuendata.** Osa
järeldusi on hiljem ümber lükatud — kehtiv seis on alati `docs/SEIS.md`.

Faile hoitakse alles mõõtmisandmete ja tõendiahelate pärast, mitte järelduste
pärast.

| fail | mis see oli | miks arhiivis |
|---|---|---|
| `llamacpp-juurdlus-20260827.md` | marginaalide kadumise juurdlus llama.cpp-s, 512 rida mõõtmisi | **Põhijäreldus („trükipoolel kasutuskõlbmatu") on 28.08 ümber lükatud** – põhjus oli 4096-tokeniline pildipiir. Vt SEIS §2.1 |
| `plaan-trukipool-jargmine-treening.md` | trükipoole plaan v2, 940 rida | sammud 1, 2, 8, 9 tehtud; lahtised on SEIS §5-s. Sisaldab endiselt ainsat teoste-kaupa märgendi-inventuuri |
| `marginaalid-silmaga-vaadata.md` | 27.08 GGUF-jooksu silmaga-vaatamise nimekiri | käib vana mudeli väljundi kohta, mis pole enam tootmises |
| `marginaalia-ankur-briifing.md` | ankrutega marginaaliformaadi katse (juuni) | katse 2, ebaõnnestus; praegune formaat on inline `<m>` |
| `9b-edasiarendus-ja-27b-strateegia.md` | 27B strateegia (juuli) | 27B suund on maha võetud, GGUF-id kustutatud 27.08 |
| `plaan.md`, `finetune-qwen-3-5.md` | algsed märtsi plaanid ja seadistusjuhend | asendatud `SPIKKER.md`-ga |
| `treening-ja-inferentsi-koodi-ulevaade-20260828.md` | 28.08 koodiülevaade: kus treening- ja inferentsikood mudeliga ei sobitu | leiud on üle viidud SEIS §2.6 ja §5.3–5.7. **Punkt 7 (`D. D. D.` valehäire) on ümber lükatud** — vt SEIS §2.7 |
| `gemini-3.7-flash-markused-20260828.md` | Gemini 3.7 Flashi seisuanalüüs | ~90 % SEIS-i ja ülaltoodud ülevaate ümbersõnastus; ainus uus leid (loobituvastuse kaks lahku jooksnud koopiat) on SEIS §5.6-s. Kordas ka ümber lükatud `D. D. D.` väidet |
| `markup-katvus-20260828-tulemused.md` | märgenduskatvuse katse täisraport (Menii `<m>` 70 → 207) | katse tehtud ja mudel tootmises 29.08-st; kokkuvõte SEIS §2.3 |
| `kurrent-20260829-oine-ahel.md` + `.sh` | 30./31.08 öine järelahel: ootas treeningu lõppu, tegi GGUF-i ja kolm hindamisjooksu | ahel läbitud 31.08 03:36, kõik sammud exit=0; tulemused `docs/kurrent-20260829-tulemused.md` |
| `kurrent-strateegia.md` | Kurrendi strateegia ja 20260602 diagnoos (juuni) | lehearv („16 579") on vale, õige on 12 712; **diagnoos ise KEHTIB endiselt** — herrnhutlaste lühendid ja eesti kohanimed, kinnitust saanud 31.08 Kambja lehel, vt SEIS §3.4 |
| `kataloogi-jalgimine-ja-ocr-20260107.py`, `ocr-service-20260107.service` | jaanuarikuised koopiad `docs/`-i all | **lahknesid päris failidest** — elav teenus on repo juurkataloogis ja `/etc/systemd/system/`-is; koopiad olid lõks |
| `näidis-lehekylje-treenimine.py` | novembri 2025 näidisskript | asendatud `scripts/train*.py`-ga |
| `katkised-lehed-20260721.txt` | juuli tööjärg katkiste lehtedega | tööjärg tehtud/aegunud |

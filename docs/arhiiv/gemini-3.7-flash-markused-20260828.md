# Qwen3.5-9B OCR: Seisuanalüüs ja soovitused

**Koostaja:** Gemini 3.7 Flash  
**Kuupäev:** 28.08.2026  
**Lähtedokument:** `docs/SEIS.md` ja seotud koodibaas  
**Fookusmudel:** `unsloth/Qwen3.5-9B` (peenhäälestus Unsloth FastVisionModel + inferents llama.cpp GGUF/mmproj)

---

## 1. Ülevaade ja kinnitatud faktid

Läbiviidud analüüsi ja otsingutega on kinnitatud järgmised `docs/SEIS.md` põhiseisukohad:

1. **Produktsiooniahel ja llama.cpp pariteet:**
   - Trüki- ja käsikirjamudelid jooksevad edukalt `llama-server` peal (pordid 8080 ja 8081).
   - `--image-max-tokens 5000` + `fit_to_grid()` + PNG lahendas vaikiva pildikärpe probleemi (`clip.cpp: set_limit_image_tokens(8, 4096)`), tagades 5x kiirema inferentsi ilma marginaale kaotamata.
2. **Marginaalide ja kursiivi seos:**
   - 58,6% VUTT toorestest marginaalidest on tegelikult kursiivis (`<m><i>...</i></m>`). Varasem `strip_italics_in_marginalia()` eemaldas need ekslikult, andes mudelile vastuolulise signaali.
3. **Menii vealiik:**
   - Uus mudel loeb Menii ääreveeru tekstist ~87% välja, kuid ei lisa `<m>` märgendit (tekst ilmub kas `<i>` sees või sildita). See ei ole nägemis- ega eraldusvõime viga, vaid märgendusotsuse/positsioonikalduvuse küsimus.
4. **Tokenieelarve on ohutu, kuid piiri lähedal:**
   - Halvim jada pikkus on ~7 849 – 7 999 tokenit 8 192-st. Kärbet ei toimu, kuid vaba puhver on ~200–340 tokenit.
5. **20-leheline GT-holdout on küllastunud:**
   - Tulemus `<m>` 170/185 ja CER 1,9–2,0% tähendab, et praegune holdout ei võimalda enam uusi peenemaid edusamme mõõta. Peamiseks sihtmõõduks on saanud Menii sond (`scripts/menii_probe.py`).

---

## 2. Kiired koodi- ja teenuseparandused (GPU-vabad, kohene realiseerimine)

Need parandused ei vaja GPU treeningut, vaid kõrvaldavad tootmises (`ocr-service`) esinevad vaiksed vead ja valehäired:

### 2.1. `finish_reason == "length"` kontroll HTTP-kliendis
- **Fail:** `kataloogi-jalgimine-ja-ocr.py` (funktsioon `process_batch_http`, read ~583–615)
- **Probleem:** Kui Qwen3.5-9B jõuab 4096 tokeni laeni (nt ülipikk lehekülg või osaline kordus), tagastab llama-server `finish_reason: "length"`. Praegune kood ignoreerib seda välja ja salvestab pooliku transkriptsiooni vaikselt eduka `.txt` failina.
- **Lahendus:** Kontrollida serveri vastust:
  ```python
  choice = vastus["choices"][0]
  if choice.get("finish_reason") == "length":
      write_err_marker(txt_path, Exception("Väljund kärbitud (max_tokens 4096 piir täis)"), KAT_MUDEL)
      return None
  ```

### 2.2. Kordusloopide tuvastuse ühtlustamine ja `D. D. D.` valehäire
- **Failid:** `scripts/loop_detect.py` ja `kataloogi-jalgimine-ja-ocr.py`
- **Probleem:** 
  - `scripts/loop_detect.py`-s on `LOOP_MIN_REPS = 3`. Ladinakeelne pühendusvormel `D. D. D.` (*Dat, Dicat, Dedicat*) klassifitseeritakse ekslikult loopiks (periood 1, kordusi 3), mille tõttu leht saab `.err` märgendi ja läheb kasutajale kaduma.
  - Teenuse sees olev `LoopStopper` kasutab vanemat piiri `LOOP_MAX_PERIOD = 20`, mis ei püüa 26-sõnalist tuvastatud loopi.
- **Lahendus:** 
  - Muuta `find_tail_loop` korduste lävi perioodist sõltuvaks:
    - Periood 1–2 sõna: `min_reps >= 8` (hoiab ära `D. D. D.` ja lühikesed loetelud).
    - Periood $\ge 3$ sõna: `min_reps >= 3` (püüab mitmerealised kordused).
  - Kaotada koodi duplitseerimine: suunata `kataloogi-jalgimine-ja-ocr.py` kasutama otse `scripts/loop_detect.py` funktsiooni `is_looped()`.

### 2.3. Re-OCR kandidaatide nimekirja külmutamine
- **Fail:** `scripts/reocr_vutt.py`
- **Probleem:** Dünaamiline päring (`--since 2026-07-22`) kasvatab valimit (143 $\rightarrow$ 147 lehte), mistõttu mudelite ajalooline võrdlus ei ole matemaatiliselt stabiilne.
- **Lahendus:** Salvestada praegused 147 kandidaati fikseeritud faili (nt `data/vutt/reocr/frozen_candidates_147.txt`) ja lisada skriptile selle faili sisselugemise tugi.

---

## 3. Qwen3.5-9B treeningu optimeerimised (järgmiseks jooksuks)

### 3.1. `train_on_responses_only=True` Unsloth Vision kollatoris
- **Fail:** `scripts/train_markup.py` (rida ~279)
- **Mõju Qwen3.5-9B peenhäälestusele:** 
  - `UnslothVisionDataCollator` maskib vaikimisi ainult pildi- ja täitetokenid. 813-tokenine süsteemijuhis (`INSTRUCTION`) läheb täies mahus lossi arvutusse.
  - Kuna keskmise lehe transkriptsioon on ~400 tokenit, kulub praegu **~67% treeninggradiendist ja arvutusvõimsusest konstantse ingliskeelse juhise uuesti-ennustamisele**.
- **Soovitus:** Seadistada kollatoris sihipärane maskimine:
  ```python
  data_collator = UnslothVisionDataCollator(
      model,
      tokenizer,
      resize="max",
      max_seq_length=8192,
      train_on_responses_only=True,
      instruction_part="<|im_start|>user\n",
      response_part="<|im_start|>assistant\n",
  )
  ```
  See suunab 100% õppimisvõimest lehe tegelikule sisule, XML-märgenditele ja `<|im_end|>` lõpetamissignaalile.

### 3.2. `--keep-m-italics` A/B käivitamine
- Käivitada puhtalt baasilt (`unsloth/Qwen3.5-9B`) isoleeritud treening `--keep-m-italics` lipuga, et kõrvaldada 58,6% marginaalide kunstlik lahknevus reaalse skaneeringu tüpograafiaga.

### 3.3. Märgendamata lehtede filtreerimine / korrastamine
- `data/lehekyljed/metadata_markup.csv` 1500 lehest on 780 lehte täiesti ilma märgenduseta (kuigi lehtedel esineb marginaale). See treenib mudelit marginaale ignoreerima.
- Enne täismahus jooksu filtreerida mittemärgendatud lehed markup-treeningust välja või viia läbi nende kiire märgistamine VUTT-is (sh Menii 58 "Toores" lehte).

### 3.4. Tühjade/hõredate lehtede näited trükikomplekti
- Lisada trükiandmestikku tühje ja illustratsioonidega lehti markeriga `[tühi lehekülg]`, kopeerides Kurrendi poolel juba edukalt toiminud metoodikat (`vutt_horedad`).

### 3.5. `¬` (U+00AC) normaliseerimine
- Normaliseerida CSV-s leiduvad 8 971 `¬` märki kriipsuks (`-`), viies treeningandmed täielikku kooskõlla juhisega (`INSTRUCTION`).

### 3.6. Piltide eelgenereerimine `fit_to_grid` LANCZOS-iga
- Salvestada treeningandmestiku pildid kettale juba valmis 32-kordsel võrel (`fit_to_grid`, LANCZOS).
- See eemaldab epohhide ajal toorpiltide CPU-põhise BICUBIC skaleerimise kulu (~300 ms/lk) ja tagab 100% geomeetrilise ja pikslitaseme pariteedi llama.cpp inferentsiga.

---

## 4. Qwen3.5-9B arhitektuur ja hindamisstrateegia

1. **Reasoning / `<think>` tokenite range kontroll:**
   - Qwen3.5 mudelitel on natiivne arutlusvõimekus. OCR-ülesande puhul on assistendi prefiksiks treenitud `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
   - Inferentsis tuleb tagada, et `enable_thinking=False` püsib nii llama-serveri päringutes kui ka Unslothi otseses väljakutses, vältides mudeli eksimist pikkadesse tühjadesse mõtteahelatesse.

2. **Laiendatud holdout-komplekt (50–100 lehte):**
   - Kuna praegune 20-leheline holdout on saavutanud lae (170/185 `<m>`), tuleb luua uus laiendatud testkomplekt.
   - Komplekt peab sisaldama sihilikult raskeid lehti: marginaalitihedaid teoseid (Menii jt), mitmeveerulisi küljendusi, segakirju (antiikva, fraktuur, kreeka) ja tühje lehti.
   - Integreerida `scripts/menii_probe.py` automaatseks osaks `scripts/eval_print.py` hindamisraamistikus.

---

## 5. Kokkuvõtlik tegevuskava

| Jrk | Tegevus | Fail / Koht | Eesmärk |
|---|---|---|---|
| 1 | `finish_reason == "length"` kontroll | `kataloogi-jalgimine-ja-ocr.py` | Vältida kärbitud lehtede vaikset salvestamist |
| 2 | `loop_detect.py` lävi ja ühendamine | `scripts/loop_detect.py` | Kaotada `D. D. D.` valehäire, ühtlustada loogika |
| 3 | Re-OCR valimi külmutamine | `scripts/reocr_vutt.py` | Tagada ajalooline võrreldavus |
| 4 | `train_on_responses_only=True` | `scripts/train_markup.py` | Suunata 100% lossist transkriptsioonile ja märgenditele |
| 5 | `--keep-m-italics` A/B jooks | `scripts/train_markup.py` | Likvideerida 58,6% marginaalide tüpograafiline vastuolu |
| 6 | Tühjade lehtede lisamine ja `¬` fix | `data/vutt/`, `convert_marginalia.py` | Tagada juhise ja andmete 100% kooskõla |
| 7 | Laiendatud holdout + Menii sond | `scripts/eval_print.py` | Mõõta mudeli tegelikku võimekust väljaspool küllastunud holdouti |

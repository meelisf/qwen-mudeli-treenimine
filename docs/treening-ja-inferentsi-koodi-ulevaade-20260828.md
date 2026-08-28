# Treening- ja inferentsikoodi ülevaade

**Kui sügavalt meie kood sobitub mudeliga ja kus on järelejäänud võidud.**
Koostatud 28.08.2026, tellimuse peale: *„vaata peale treenimise ja inferentsi
koodile, ära midagi muuda, anna nõu, kuidas paremini sobituda mudelile."*

> **Uurimismeetod:** koodi lugemine + CPU-poolseid mõõtmisi (tokenizer, PIL).
> GPU-d ei puudutatud, ühtegi faili ei muudetud. Kõrvuti jooksnud
> `print-base-r64-20260827` treening (samm ~541/646) jäi puutumata.

Materjal, mida läbi vaadati:

| osa | failid |
|---|---|
| treening | `scripts/train_markup.py`, `scripts/train_kurrent.py`, `scripts/imaging.py`, `scripts/prompt.py`, `convert_marginalia.py` (puhastusahel) |
| inferents | `kataloogi-jalgimine-ja-ocr.py`, `scripts/loop_detect.py`, `scripts/eval_kurrent.py`, `scripts/merge_lora.py` |
| baaskomponendid | `unsloth.trainer.UnslothVisionDataCollator` (lähtekood), `chat_template.jinja` (mudeli kaustas) |
| taustadokumendid | `docs/arhiiv/llamacpp-juurdlus-20260827.md`, `docs/arhiiv/plaan-trukipool-jargmine-treening.md`, `SPIKKER.md` |

Palju allpoolkirjatust on juba plaanis
(`docs/arhiiv/plaan-trukipool-jargmine-treening.md`) — igal juhul juures täpsustus,
**kas ja kuhu see praegusesse `print-base-r64-20260827` jooksu ei mahtunud**.

---

## Kokkuvõte ühes tabelis

| # | leid | mõju | kulu | olek |
|---|---|---|---|---|
| 1 | loss konstantsel juhisel — ~26–67 % treenitud tokenitest on 813-tokenine `INSTRUCTION` | suur | A/B ~2 h + kood | avastus, plaanis puudus |
| 2 | tokenitööjõu varu on **50 tokenit**, mitte 105 | risk (vaikne kärpimine) | CPU-mõõtmine | uus mõõtmine |
| 3 | trükikomplektis **0** tühja/hõredat lehte; juhis lubab `[tühi lehekülg]` | keskmine | VUTT-is märgendada | SPIKKER-is teada, ei täitnud |
| 4 | `¬` (8 971 esinemist selles jooksus) juhises dokumenteerimata | väike, süstemaatiline | ~1 h koodi | plaani samm 2, ei jõudnud |
| 5 | treening BICUBIC (protsessor) vs tootmine LANCZOS (`fit_to_grid`) | teadlik nihke, kasuliku suunaga | andmestiku uuesti ehitada | osalt juurdluses |
| 6 | `finish_reason == "length"` jääb kontrollimata — kärbitud leht kirjutatakse tervena | keskmine | ~10 rida | uus |
| 7 | `loop_detect.py` `min_reps=3` — juurdluse oma `D. D. D.`-valehäire elab edasi | keskmine (.err = kadunud leht) | ~5 rida | uus |
| 8 | pisiasjad: weight_decay erinevus, surnud kood, `save_strategy` | kosmeetika | — | — |

---

## Mis on juba hästi — puudutada ei tasu

Need kontrolliti läbi, sest just siin on treening/inferents-nihked tavalised:

1. **Chat template on treeningus ja inferentsis täpsus-tokenini sama.**
   `chat_template.jinja` renderdab assistendi pöörde treeningul kujul
   `<|im_start|>assistant\n<think>\n\n</think>\n\n` + transkriptsioon
   (tühja `reasoning_content`-iga langeb mall just sellele harule) ja
   inferentsi prompt (`enable_thinking=False`) lõpeb täpselt sama stringiga.
   Ehk juurdluse „mõtlemisrežiimi kõrvallei" põhjus on treeningus nähtav:
   tühje think-blokke õpitakse, mitte ei taluta.
2. **Promptijärjekord = treeningjärjekord.** `text` enne `image` mõlemas otsas;
   juurdlus (kliendi-eksperiment) kinnitas, et just see järjekord on parim.
3. **Pildieelarve on mõlemas otsas 5 120 000 px** (`imaging.MAX_PIXELS`, üks
   allikas) ja kollatoris `resize="max"` — pilti ei kärbita kollatoris enne
   protsessorit, `longest_edge`-i semantika (KOGU pikslite arv) on õigesti
   arvestatud.
4. **Deterministlik inferents:** `do_sample=False` / `temperature 0` mõlemal
   mootoril, samplerid ilusti puuduvad (juurdlus: DRY ja repeat_penalty mõlemad
   lükatud tagasi).
5. **Holdout'id, checkpointid, seed 3407, `random_state=3407`** —
   korratavuse pinnad on paigas. `train_markup.py` ei seadista
   `save_strategy`'t, aga transformersi vaikeväärtus (`save_steps=500`)
   päästis selle jooksu juures: `checkpoint-500` on olemas.

---

## 1. Suurim järelejäänud võit: loss konstantsel juhisel

`UnslothVisionDataCollator` maskib ainult padding- ja pilditokenid.
**Kõik muu treenitakse — sealhulgas 813-tokenine `INSTRUCTION`** (mõõdetud
tokenizeriga). Numbritega praeguse komplekti peal:

| | visuaal (maskitud) | juhis (treenitakse) | transkriptsioon (treenitakse) | **juhise osakaal lossist** |
|---|---|---|---|---|
| mediaanleht | ~4 700 | 813 | ~400 | **~67 %** |
| suurim leht | 5 015 | 813 | 2 274 | 26 % |

Ehk **umbes pool kogu treeninggradiendist käib hetkel konstantse ingliskeelse
juhise uuesti-ennustamise peale**, mitte transkriptsiooni ega lõpetamise
õppimise peale. Kollator toetab seda valmis:
`train_on_responses_only=True` + `instruction_part="<|im_start|>user\n"` +
`response_part="<|im_start|>assistant\n"`. Siis jääb lossile ainult
transkriptsioon + tühje think-blokid + `<|im_end|>` — just need, mis õpetavad
transkribeerima ja lõpetama.

**Ettevaatus:** üks test-batch läbi enne täisjooksu (kontroll, et labels
jäävad õigesse kohta ja pilditokenid on maskitud). Ja A/B — sama andmestik,
sama seed, ainult lipp erinev — trükijooks on mõõdetult ~2 h, see on odav
katse. Kui võitu ei ole, on ka see väärtuslik teadmine: praegune kujund
töötab ju kõvade numbritega.

## 2. Tokenitööjõu kliff on lähemal kui plaanis kirjas

Plaan mõõtis (ainult `data/vutt`, 1113 lk): max 8 087, varu 105 tokenit.
Praegune jooks võttis juurde `data/lehekyljed` — mõõtsin **kogu 2 598-näitelise
komplekti** (geomeetria valem + tokenizer + malli kulud, CPU):

| | kokku (tokenid) |
|---|---|
| mediaan | ~6 070 |
| p90 | ~7 340 |
| p99 | ~7 670 |
| **max** | **~8 140** |
| `max_seq_length` | 8 192 |

Praegu **ei kärbita ühtegi näidet** — aga varu on ~50 tokenit. Üksainus
Menii-laadse lehe laadi andmelisandil lõikub vaikselt koos oma `<|im_end|>`-iga:
mudel saab õppetunni, et pikk leht „lõpeb" ilma lõpetamiseta. See on täpselt
see vaikne vealiik, mida juurdlus kirjeldab („vaikeväärtus, mida keegi ei
näita, on ohtlikum kui viga").

**Nõuanne:** enne iga andmestiku muutmist korrata seda CPU-mõõtmist (ei vaja
GPU-d, ~5 min). Kui piir täitub: tõsta `max_length` (proovida testjooksuga —
algne 8192 valiti OOM-i tõttu) või pidada pikimad lehed teadlikult välja.
Kumbki ei ole praegu kiire — aga arv 105 tuleb asendada arvuga 50.

## 3. Tühjad ja hõredad lehed on trükikomplektis endiselt null

Kontrollitud 28.08:

| fail | `[tühi lehekülg]` ridu |
|---|---|
| `data/vutt/metadata.csv` (1 113) | **0** |
| `data/lehekyljed/metadata_markup.csv` (1 500) | **0** (15 tühja rida jäetakse laadimisel välja) |

Juhis (`INSTRUCTION`) lubab markerit ja annab selle jaoks kolm reeglit — aga
mudel pole elus ühtegi näidet näinud. Tegemist on täpselt sama lahknevusega,
mille Kurrendi poolel juba parandasite (`vutt_horedad`, 23+3 lehte), ja mille
kohta SPIKKER ütleb: *„Trükipool (`data/vutt`) on tegemata."* Kuni see nii on,
on trükimudeli tühja lehe käitumine toetatud ainult prompti sõnade peale.

Odavaim tee (SPIKKER-is juba kirjas): VUTT-is märgendada →
`build_vutt_dataset.py --type print --max-chars 100 --out data/vutt --append
--allikas vutt_tyhjad` → järgmine trükijooks korjab need automaatselt.

## 4. `¬` (U+00AC) on selles jooksus sees, juhises ei ole

Plaani samm 2 jäi tegemata ja jõudis seega ka sellesse jooksu:

| | esinemisi |
|---|---|
| `data/lehekyljed/metadata_markup.csv` (mõõdetud 28.08) | **8 971** |
| `data/vutt/metadata.csv` (plaani mõõtmine) | 3 961 |

`INSTRUCTION` tunneb ainult `-` ja `⸗`. Mudel õpib, et 12 teoses on rea lõpus
märk, mida juhis ei maini. Mitte katastroof (teose sees on transkribeerijad
järjekindlad), aga „juhis ↔ andmestik" kooskõla on katki ja VUTT-i poole
voolav väljund kannab edasi märki, mida keegi juhendina enam kirja ei pea.
Parandus (`¬` → `-` `clean_markup`-is) on plaanis — järgmine jooks.

## 5. Pildi filter: nihke suund on kasulik, aga nihke on

Mõõdetud kettalt (valim):

| | pildi suurused |
|---|---|
| `data/vutt/images` | 13,8–29,5 MP — **täisresolutsioonis**, ei ole `--resize`-ga ehitatud |
| `data/lehekyljed/images` | 1,6–15,3 MP, segane |

Seega treeningul skaleerib protsessor iga epohhi **BICUBIC-iga** (ja maksab
selle eest ~310 ms/lehe), aga tootmises saab llama.cpp `fit_to_grid`
**LANCZOS**-PNGsid. Juurdluse mõõtmine ütleb, et LANCZOS-i suund on isegi
BICUBIC-treenitud mudelile kasulik (`<m>` 200 → 216) — ehk praegune nihe on
**kasuliku suunaga**, aga formaalselt on see nihe.

Kui tahetakse täpset pariteeti (ja kiirust), siis järgmise andmestiku
ehitamisel kasutada `prepare_image`-is **grid-joondatud** skaleeringut:
`fit_to_grid`, mitte `fit_to_budget`. Erinevus on õhuke: `fit_to_budget`
ümardab ainult eelarvele, protsessor seejärel 32-kordsele võrele — tulemus
võib harvadel juhtudel ühe võresammu võrra `fit_to_grid`-ist lahku minna.
Juurdluse omadus-test leidis sarnaseid äärjuhtumeid 4/1240. Ühe
skaleerimiskoha põhimõte jääb: *AINULT treeningandmetel on valida* —
tootmises otsustab klient.

## 6. Inferents: `finish_reason` jääb kontrollimata

`process_batch_http` küsib `max_tokens: 4096`, aga ei vaata vastuses
`finish_reason` välja. Leht, mis jookseb väljundilaki (juurdlus: 6/143
jooksis, neist 1 päriselt liiga pikk leht, mitte loop), kirjutatakse
**vaikselt tervena** `.txt`-ina — `is_looped` teda ei näe, sest ta EI ole
loop. Ülejäänud 4 olid loopid ja saavad nüüd `.err`-i; päriselt liiga pikk
leht jääb tähelepanuta.

Vastuses on väli olemas: `finish_reason == "length"` puhul `.err`-märgend
(kategooria `mudel`) või vähemalt logirida. Unsloth'i tee peal vastab sellele
`LoopStopper`i tokeniloendur (jäädvustab `genereeritud` stopi hetkel). Tegemist
on juurdluse „kaks erinevat 4096-piiri" tabeli teise poole sulgemisega
kliendipoolel.

## 7. Loop-detektor: juurdluse oma järeldus on koodis rakendamata

`loop_detect.py` (HTTP-tee valmisväljundi kontroll) on kalibreeritud ja hea:
max_period 30, märgitasandi kontroll, viimase sõna viskamine. **Aga
`LOOP_MIN_REPS = 3`.** Juurdlus mõõtis: lävi ≥8 kordust lühikeste perioodide
(1–2 sõna) puhul hoiab kõik 4 päris loopi — (1, 1365), (1, 809), (1, 9),
(26, 48) — ja välistab `D. D. D.` pühendusvormeli (periood 1, 3 kordust),
mille 1 189 inimese kinnitatud teksti peal märgiti valehäireks.

Seega iga trükileht, mis lõpeb korrektse `D. D. D.`-ga, saab HTTP-tee peal
`.err`-i ja **läheb kaotsi kasutajale** — teenuse `LoopStopper`i (20 sõna /
16 kordust) puhul sama ei juhtu, sest ta peatab ainult, ei märgi. Parandus on
period-sõltuv lävi, ~5 rida, ja pärast seda saab kaks algoritmi koopiad
(teenuses vs `loop_detect.py`) lõpuks kokku viia — seda soovitab juba
`loop_detect.py` enda dokumentatsioon.

## 8. Pisiasjad (kirja panemiseks, mitte tegemiseks)

- `weight_decay=0.001` (markup) vs `0.01` (kurrent) — järjepidevuse küsimus,
  mitte viga.
- `_setup_tokenizer`'i replace `"enable_thinking=True"` → `"False"` on surnud
  kood: mallis pole seda literaali (seal seisab `enable_thinking is false`) ja
  `get_chat_template` annab lipu niikuinii otseselt edasi.
- `strip_output` lõikab markeri `"assistant\n"` järel — teoreetiliselt võib
  transkriptsiooni sees sama jada ära lõigata; Qwen-mall kasutab
  `<|im_start|>assistant`, ehk praktiliselt ei juhtu.
- `save_strategy="epoch"` (nagu `train_kurrent.py`-s) oleks deterministlikum
  kui vaikeväärtusele tuginemine.
- Teenuse `process_batch` (unsloth-i varutee) ei kutsu `fit_to_grid`-i —
  teadlik, dokumenteeritud seis. Mitte lülitada unsloth-ile tagasi ilma
  SPIKKER-i tablood „Millal vaja fit_to_budget"-ita.
- Kurrent-teenuse juhise parandus (`KURRENT_INSTRUCTION`) — mõõdetud mõju
  0,1 pp, aga **teha eraldi sammuna**, nagu kokku lepitud, et mootorivahetus
  jääks ainsaks muutujaks.

---

## Mis sellest jooksust (print-base-r64-20260827) peale hakata

1. **Käima lõpetada ja hinnata marginaalikeskset.** Plaan 0b: `<m>` on ainus
   sine qua non — hindamine peab andma `<m>` recall'i ja `<m>`-sisu CER-i
   143 külmutatud „Toores" lehel + 20 holdout-lehel, mitte ainult
   lehe-koond-CER-i. Menii (1635-1) lehed on siin peamõõt.
2. **Blokeerija:** `reocr_vutt.py` on hetkel llama-server-only — trükipoole
   mõõtmine transformersi ahelal vajab backend'i (plaani punkt 5). Ilma
   selleta jääb „uus vs vana" võrdlus taas silmamõõduks.
3. **Järgmise trükijooksu eeltingimused selles järjekorras:** tühjade lehtede
   märgendamine (3) → `¬` normaliseerimine (4) → tokenieelarve uus mõõtmine
   (2) → soovi korral `train_on_responses_only` A/B (1) → grid-joondatud
   andmestik (5).

**Juhtmõtet pidi:** kõik kolm suurimat leidumust on sama laadi kui juurdluse
omad — mitte mudeli vead, vaid ahela vaiksed eeldused (kes lossi saab, kui
pikk leht tohib olla, mis juhisega ja mis märgiga andmetel). Need on ka kõik
mõõdetavad enne treenimist, enamasti ilma GPU-ta.

---

*Meetod ja kõrvalehoid: midagi ei muudetud; GPU-d ei puudutatud. Tokeniarvutused
`AutoTokenizer` (models/qwen3.5-ocr-markup-20260722) + geomeetria valem
`(H/32)×(W/32)` eelarvega 5 120 000 px; malli kulu hinnatud +813 (juhis) +
~40 (wrapper) tokenit. Kõik numbrid korratavad samade käskudega.*

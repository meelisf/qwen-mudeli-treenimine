# Qwen 3.5 Fine-Tuning Plaan (OCR)

See dokument kirjeldab samm-sammulist tegevuskava Qwen 3.5 Vision-Language mudeli peenhäälestamiseks ajalooliste tekstide transkribeerimiseks.

## 1. Keskkonna ja kataloogide loomine
- [ ] Luua puhas kataloogistruktuur:
  - `data/raw/` - Algupärased andmed.
  - `data/processed/` - Treeninguks valmis pildid ja `metadata.csv`.
  - `scripts/` - Treening- ja ettevalmistusskriptid.
  - `models/` - Treeningu väljundid (adapterid ja GGUF-id).
- [ ] Kontrollida ja uuendada `requirements.txt` vastavalt Qwen 3.5 vajadustele (Unsloth 2025+ tugi).

## 2. Andmete ettevalmistus (Kriitiline samm)

- [ ] **Puhastamine**: 
  - Kontrollida reavahetuste ja erimärkide (ſ, æ, jne) säilimist, teisendada .
  - Poolituskriipsud: asendada failides esinevad `¬` poolitusmärgid õigete `-` (antiikva), `⸗` (fraktuur) märkidega
  - Ligatuurid: æ, œ jäävad, typograafilised (st, ff, fi, fl) kirjutatakse lahku
  - Umlaudid teisendada: `uͤ` → `ü`, `oͤ` → `ö`, `aͤ` → `ä`
  - Erikujud: `ſ` (pikk s), `ß` jäävad
  - Kaldkiri: `*tekst*`, rasvane: `**tekst**`
  - Koodivahetus (fraktuur antiikvas vms): `~tekst~`
  - Ääremärkused: `[[m: sisu]]`
  - Signatuurimärgid: lehe lõpus, nt `A 3`
  - Leheküljepiir (topeltleheküljel): `--lk--`
  - Maha tõmmatud leheküljed: ei transkribeerita
- [ ] **Teisendamine**: Viia andmed Qwen 3.5 jaoks sobivasse "messages" formaati.
- [ ] **Valideerimine**: Kontrollida, et kõik pildid on kättesaadavad ja failiteed on korrektsed.

## 3. Mudeli seadistamine
- [ ] **Mudeli valik**: `unsloth/Qwen3.5-9B` (või vastav Instruct versioon).
- [ ] **LoRA seadistus**: 
  - Rank (r=16 või 32).
  - Sihtmärgid: Vision ja Language kihid (nii Attention kui MLP).
- [ ] **Prompt Engineering**: Koostada optimeeritud süsteemne juhis (Instruction), mis suunab mudelit hoidma ajaloolist truudust.

## 4. Treening
- [ ] Seadistada `SFTTrainer` koos `UnslothVisionDataCollator`-iga.
- [ ] Määrata optimaalne `max_seq_length` (Qwen 3.5 toetab väga suurt konteksti, kuid hoiame mälukasutuse kontrolli all).
- [ ] Teostada kontroll-treening (nt 10-20 sammu), et veenduda mälukasutuse ja lossi kahanemise korrektsuses.

## 5. Valideerimine ja Eksport
- [ ] **Inference test**: Testida mudelit piltidega, mida treeningus ei kasutatud.
- [ ] **Võrdlus**: Võrrelda tulemusi vana Qwen3-VL-8B mudeliga.
- [ ] **GGUF eksport**: Konverteerida tulemus GGUF formaati (Q4_K_M, Q8_0) kasutamiseks llama.cpp-ga.

---
*Märkused: Qwen 3.5 on hübriidarhitektuuriga (standardne + lineaarne attention), mis peaks olema märgatavalt kiirem ja täpsem kui eelmine põlvkond.*

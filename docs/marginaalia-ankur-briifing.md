# Kontekst: marginaalia ankrutega formaat

## Mis olukord on

Me treenime Qwen 3.5-9B multimodaalset mudelit ajalooliste dokumentide transkribeerimiseks (VUTT projekt). Treeningandmed on pildid + referentstranskriptsioonid VUTT-i `.txt` failidest.

Praegune transkriptsiooniformaat paneb ääremärkused (marginaaliad) **inline põhiteksti sisse**, iga marginaalia rida eraldi `<m>` tähisega:

```
gleichfalls Exempeln / alß Zoroaſter, Moſes Cretenſis, Circe,
<m>Chryſoſt.</m>
<m>tom: 3. in</m>
<m>Evang: Io-</m>
<m>hannis ho-</m>
... (veel 11 rida) ...
<m>Deut. 18.</m>
Medea: Im gleichen die vnheiligen Väter...
```

## Miks see probleem on

Pikk marginaalia (kuni 15 rida) katkestab põhiteksti sekventsiaalse lugemise. Mudel peab:
1. Lugema põhiteksti kuni "Circe,"
2. Hüppama lehe vasakusse serva
3. Transkribeerima 15 rida väiksemat teksti
4. Hüppama tagasi põhiteksti täpselt samale kohale

See "edasi-tagasi hüppamine" koormab mudeli tähelepanumehhanismi. Praktikas: mudel kipub pikki marginaaliad osaliselt unustama või segamini ajama.

## Lahendus: kahe-etapiline ankrutega formaat

Uus treeningandmete formaat lahutab põhiteksti ja marginaalid:

```
gleichfalls Exempeln / alß Zoroaſter, Moſes Cretenſis, Circe, <m_ref id="1"/>
Medea: Im gleichen die vnheiligen Väter die Bapſte zu Rom/
...
Daher hat GOtt der HErr in ſeinem Wort außdrücklich
ſo ſoltu nicht lernen thun die grewel dieſer Völcker/ <m_ref id="2"/>
...
HERREN ein Grewel / vnd vmb ſolcher Grewel <m_ref id="3"/>
willen vertreibt ſie der HERR...

[MARGINAALID]
<m id="1">Chryſoſt.
tom: 3. in
Evang: Io-
hannis ho-
mil. 1. tra-
dit, Pytha-
goram
Philoſo-
phum Ma-
giam no-
viſſe. Vide
Auguſtin:
2. tractat:
lib 1.c. 3.
Deut. 18.</m>

<m id="2">Vide Oſa-
um Magnũ
de populis
ſeptentrio-
nalibus, &
eorum Ma-
gia quã eti
am Aretius
refert p. 2.
probl. f.
595.</m>

<m id="3">Lut: 19. &
20.</m>
```

**Põhimõte:**
- 1. jooks: mudel loeb põhiteksti järjest, paneb `<m_ref id="N"/>` ankru täpselt sinna, kus marginaalia visuaalselt algab (vertikaalne joondus pildil)
- 2. jooks: mudel transkribeerib kõik marginaaliad järjest `[MARGINAALID]` sektsiooni, blokkidena — neid saab lugeda ülalt alla ilma põhitekstiga edasi-tagasi hüppamata
- **Reavahetused marginaalide sees säilivad** — need on olulised (poolituskriipsud jms)
- Positsiooninfo säilib ankru kaudu (teame, millise lõiguga iga marginaalia seostub)

## Mis on juba tehtud

Repos (`scripts/`) on kolm muudatust:

1. **`convert_marginalia.py`** — konversiooniskript
   - Üksikfail: `python3 scripts/convert_marginalia.py --test page.txt` (eelvaade)
   - Kirjutab faili: `python3 scripts/convert_marginalia.py page.txt`
   - Loogika: järjestikused `<m>rida</m>` read → üks blokk; ankur lisatakse eelmise põhiteksti rea lõppu

2. **`prompt.py`** — INSTRUCTION uuendatud, kirjeldab kahe-etapilist lähenemist

3. **`build_vutt_dataset.py`** — rakendab `convert_marginalia()` automaatselt iga lehekülje peal CSV ehitamisel

## Järgmised sammud

1. `git pull` et saada uued skriptid
2. `python3 scripts/vutt_sync.py` — sünkroniseeri VUTT andmed (kui pole värske)
3. `python3 scripts/build_vutt_dataset.py` — ehita uus CSV ankrutega formaadis
4. Treeni mudel **esimese checkpoindi peale** (`train_markup.py --base models/qwen3.5-ocr-lora-backup-...`)
5. Testi keeruliste lehtedega (pikad marginaalid) — vaata kas ankrud ja blokkide piirid vastavad

Kui tulemus on hea, mõeldakse edasi:
- Kogu VUTT korpuse konverteerimine ankrutega formaati
- Visuaalne esitus VUTT-is (marginaalid joonealustena, mitte inline)

## Olulised piirangud

- **Hübriidvariant (lühikesed inline, pikad ankrutega) on välja lükatud** — ebaühtlus segaks mudelit rohkem kui ühtsus
- **ID-de järjekord on kriitiline** — N-nda `<m_ref id="N"/>` peab vastama N-ndale `<m id="N">` blokile; valesti loendatud blokid on suurim riskikoht
- **JSON formaat lükati tagasi** — JSON-süntaksi hoidmine ja OCR samaaegselt koormab mudelit kahekordse ülesandega; XML-laiendus on loomulikum juba õpitud märgenduste kontekstis

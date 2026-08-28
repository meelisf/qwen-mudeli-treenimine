# Kontekst: marginaalia ankrutega formaat

## Mis olukord on

Me treenime Qwen 3.5-9B multimodaalset mudelit ajalooliste dokumentide transkribeerimiseks (VUTT projekt). Treeningandmed on pildid + referentstranskriptsioonid VUTT-i `.txt` failidest.

Praegune VUTTis kasutatav inimeste märgendatud transkriptsiooniformaat paneb ääremärkused (marginaaliad) **inline põhiteksti sisse**, iga marginaalia rida eraldi `<m>` tähisega:

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

<marginalia>
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
</marginalia>
```

**Põhimõte:**
- 1. jooks: mudel loeb põhiteksti järjest, paneb `<m_ref id="N"/>` ankru täpselt sinna, kus marginaalia visuaalselt algab (vertikaalne joondus pildil)
- 2. jooks: mudel transkribeerib kõik marginaaliad järjest `<marginalia>` sektsiooni, blokkidena — neid saab lugeda ülalt alla ilma põhitekstiga edasi-tagasi hüppamata
- **Reavahetused marginaalide sees säilivad** — need on olulised (poolituskriipsud jms)
- Positsiooninfo säilib ankru kaudu (teame, millise lõiguga iga marginaalia seostub)


## Olulised piirangud

- **Hübriidvariant (lühikesed inline, pikad ankrutega) on välja lükatud** — ebaühtlus segaks mudelit rohkem kui ühtsus
- **ID-de järjekord on kriitiline** — N-nda `<m_ref id="N"/>` peab vastama N-ndale `<m id="N">` blokile; valesti loendatud blokid on suurim riskikoht
- **JSON formaat lükati tagasi** — JSON-süntaksi hoidmine ja OCR samaaegselt koormab mudelit kahekordse ülesandega; XML-laiendus on loomulikum juba õpitud märgenduste kontekstis

---

## convert_marginalia.py uue versiooni lähtepunkt

Git-ajaloos on ankruversioon commit `1841460` all, aga sellel on viga: töötleb teksti rida-realt ja eeldab iga `<m>...</m>` ühel real. Mitmerealine blokk (`<m>rida1\nrida2</m>`) läheb katki.

Hilisem (praegune) versioon lahendas multiline probleemi `_normalize_multiline_m`-ga – aga see jagab ühe bloki mitmeks, mis ankruversiooni jaoks ei sobi (kõik ühe marginaali read peavad minema ühe `<m id="N">` alla).

**Õige lahendus:** multiline-teadlik parser, mis jälgib kas ollakse avatud `<m>` sees:

```python
def convert(text: str) -> str:
    lines = text.split("\n")
    result_lines = []
    marginalia_blocks = []  # list of str (ühe bloki kogu sisu)
    current_block_lines = None  # None = pole blokis; list = kogume blokki

    def flush_block():
        if current_block_lines is None:
            return
        block_id = len(marginalia_blocks) + 1
        marginalia_blocks.append("\n".join(current_block_lines))
        if result_lines:
            result_lines[-1] = result_lines[-1].rstrip() + f' <m_ref id="{block_id}"/>'

    for line in lines:
        stripped = line.strip()

        if current_block_lines is not None:
            # Oleme avatud bloki sees
            if stripped.endswith("</m>"):
                # Blokk lõpeb siin
                current_block_lines.append(stripped[:-4])  # eemalda </m>
                flush_block()
                current_block_lines = None
            else:
                current_block_lines.append(stripped)
        elif stripped.startswith("<m>") and stripped.endswith("</m>"):
            # Üherealine blokk
            flush_block()
            current_block_lines = [stripped[3:-4]]  # eemalda <m> ja </m>
            flush_block()
            current_block_lines = None
        elif stripped.startswith("<m>"):
            # Mitmerealise bloki algus
            flush_block()
            current_block_lines = [stripped[3:]]  # eemalda <m>
        else:
            # Põhiteksti rida
            result_lines.append(line)

    flush_block()

    if not marginalia_blocks:
        return text

    m_section = "\n\n".join(
        f'<m id="{i+1}">{c}</m>' for i, c in enumerate(marginalia_blocks)
    )
    return "\n".join(result_lines).rstrip() + "\n\n<marginalia>\n" + m_section + "\n</marginalia>"
```

Märkused:
- `<pb/>` topeltleheküljed vajavad eraldi käsitlust (vt praeguse versiooni `parts = re.split(r"(<pb/>)", text)` loogika)
- Inline `<m>...</m>` segatud reas (tekst ümber) vajab lisaks `_INLINE_M` regex asendust nagu vanas versioonis
- Formaat: `[MARGINAALID]` asemel `<marginalia>...</marginalia>` on XML-konsistentsem

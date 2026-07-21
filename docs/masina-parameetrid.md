# Masina parameetrid

## Komponendid (ostudokumentide põhjal)

| Komponent | Mudel | Detailid |
|-----------|-------|---------|
| Protsessor | Intel Core i7-14700F | 2.1GHz, 20 tuuma (8P+12E), BX8071514700F |
| Jahutus | DeepCool AK620 | R-AK620-BKNNMT-G |
| Emaplaat | Gigabyte B760 GAMING X WIFI6E | DDR5, PCIe Gen 5 |
| Mälu | G.Skill DDR5 64GB | 2×32GB, 6000MHz CL36, F5-6000J3636F32GX2-R |
| SSD | Samsung 990 EVO Plus 2TB | M.2 PCIe 4.0, MZ-V9S2T0BW |
| Korpus | be quiet! Silent Base 802 | EATX, BGW39 |
| GPU | Gigabyte RTX 5090 Windforce OC 32GB | GV-N5090WF3OC-32GD |
| PSU | Gigabyte UD1000GM PG5 1000W | ATX 3.0, 80+ Gold, native 12V-2x6, GP-UD1000GMPG5 |

## GPU võimsuspiirid

- GPU TDP: 575W
- Vaikimisi power limit: 575W
- **Treeninguks seatud limiit: 450W** (`sudo nvidia-smi -pl 450`)
  - Põhjus: ~8h treeningutel sustained 559W tekitas muret kaablite/PSU kohta
  - 450W on konservatiivne ja ohutu valik
- Power limit lähtestub rebooti peal (persistence mode väljas)

## Süsteemi kogutarbimine treeningu ajal

| Komponent | ~Tarbimine |
|-----------|-----------|
| RTX 5090 (450W limiidiga) | ~450W |
| i7-14700F (koormusel) | ~100W |
| Emaplaat + RAM + SSD | ~50W |
| **Kokku** | **~600W** |

600W / 1000W PSU = 60% koormus – optimaalne efektiivsuse tsoon.

## PSU ohutusinfo

- **ATX 3.0** sertifikaat: projekteeritud uue põlvkonna GPU transientkoormuste jaoks
- **Native 16-pin (12V-2x6)** väljund – ei kasuta adapterit (ohutum kui 4×8-pin adapter)
- 12V-2x6 nimivõimsus: 600W – 450W limiidiga on hea varu

## UPS

| | Väärtus |
|--|--|
| Mudel | CyberPower PR1500ELCD (ID: 506696) |
| Nimivõimsus | 1500VA / **1350W** (power factor 0.9) |
| Tüüp | Line-interactive, pure sine wave |

**Koormusanalüüs (mõõdetud NUT-iga):**
- Praegune koormus: **42%** × 1350W ≈ 567W seinast
- UPS max 1350W → varu ~783W (väga mugav)
- Aku täis (100%), runtime praegusel koormusel ~33 min
- Ilma power limit-ita (575W GPU) oleks koormus ~55% – ikka OK

**Jälgimine Linuxis:**
```bash
# NUT (Network UPS Tools) – soovituslik
sudo apt install nut
# CyberPower pwrstat alternatiiv
which pwrstat
```

## GPU konnektor

Gigabyte RTX 5090 WF OC kasutab **12V-2x6** konnektorit (uuem, parandatud versioon 12VHPWR-st).
PSU UD1000GM PG5 on natiivse 12V-2x6 väljundiga – adapter pole vajalik.

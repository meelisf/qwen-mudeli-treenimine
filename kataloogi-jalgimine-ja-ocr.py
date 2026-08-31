# -*- coding: utf-8 -*-
"""
AUTO-OCR kataloogi jälgimise teenus – Qwen3.5-9B peenhäälestatud mudel

Jälgib kataloogi /home/mf/Dokumendid/LLM/AUTO-OCR rekursiivselt.
Iga pildi (.jpg/.png/...) jaoks, millel puudub kõrvalolev .txt fail,
käivitatakse OCR ja tulemus salvestatakse sama nimega .txt faili.
PDF-id pakitakse kõigepealt piltide kaustaks lahti.

Teenuse haldamine:
  sudo systemctl start ocr-service
  sudo systemctl stop ocr-service
  sudo systemctl status ocr-service
  journalctl -u ocr-service -f
"""

import os
import sys
import time
import logging
import signal
import torch
import gc
import re
import unicodedata
import shutil
from typing import Optional
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage
from pdf2image import convert_from_path
from transformers import StoppingCriteria
from unsloth import FastVisionModel
from natsort import natsorted
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from prompt import INSTRUCTION
from imaging import MAX_PIXELS, fit_to_grid
from loop_detect import is_looped

# --- 0. LOGIMINE JA SIGNAALID ---

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "ocr-service.log"

class FlushFileHandler(logging.FileHandler):
    """FileHandler, mis flushibi iga kirje järel – logid jõuavad kohe faili."""
    def emit(self, record):
        super().emit(record)
        self.flush()

root_logger = logging.getLogger()
root_logger.handlers = []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        FlushFileHandler(str(LOG_FILE), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"Sain signaali {signum}, peatan...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Vigaste PDF-ide meeldejätmine (vältimaks korduvaid ebaõnnestunud katseid)
failed_pdfs = set()

# --- 1. SEADISTUS ---

BASE_OCR_KAUST = "/home/mf/Dokumendid/LLM/AUTO-OCR"

# Iga tüübi jaoks eraldi alamkaust ja mudel
MODEL_CONFIGS = {
    "print": "models/qwen3.5-ocr-print-base-r64-mi-vl-20260828",
    "hand":  "models/qwen3.5-ocr-kurrent-20260829",
}

#: Mootor tüübi kaupa: "unsloth" (kohapeal GPU-l) või "llamacpp" (HTTP server).
#:
#: llama.cpp on Kurrendil mõõdetult PARITEEDIS ja 4,2x kiirem (CER 8,8 % vs
#: 8,7 %, GPU 12,7 vs 25,2 GB) – vt docs/arhiiv/llamacpp-juurdlus-20260827.md.
#:
#: TRÜKIPOOL: varasem hinnang „kasutuskõlbmatu, kaotab peene ääreveeru" on
#: 28.08.2026 ÜMBER LÜKATUD. Põhjus oli llama.cpp vaikne 4096-tokeniline
#: pildipiir, mitte mootor; `--image-max-tokens 5000` + PNG + fit_to_grid
#: annavad 20-lehesel GT-holdout'il transformersiga pariteedi (CER 1,9 % vs
#: 2,0 %, `<m>` 169 vs 170) ja 5x kiiruse (6,2 vs 30,8 s/lk). Mõõtmine:
#: scripts/eval_print.py, andmed data/vutt/eval/.
#:
#: AKTIVEERIMINE: vt SPIKKER.md "Käsikirjapool llama.cpp peale". Server peab
#: käima ENNE teenuse käivitamist ja GPU-l ei ole ruumi mõlemale mootorile
#: korraga – seetõttu vabastatakse unslothi mudel HTTP-tüübi ajaks.
ENGINE_CONFIGS = {
    "print": "llamacpp",
    "hand":  "llamacpp",
}

#: Iga mudel vajab OMA serverit – üks llama-server hoiab ühte mudelit.
#: Mõlemad mahuvad korraga GPU-le (~12,5 GB kumbki, RTX 5090 32,6 GB).
LLAMACPP_ENDPOINTS = {
    "print": "http://127.0.0.1:8080",
    "hand":  "http://127.0.0.1:8081",
}
LLAMACPP_TIMEOUT = 900

BATCH_SIZE = 4

PDF_DPI = 300

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Instruktsioon – identne treenimisega (train.py / train_stage2.py)
# Vana instruktsioon (markdown formaat, enne VUTT XML-ile \u00FCleminekut):
# INSTRUCTION_OLD = """You are an expert OCR assistant for historical documents.
#
# Instructions:
# 1. Transcribe the entire page from the provided image.
# 2. Preserve original line breaks and hyphenation:
#    - Antiqua hyphenation: - (regular hyphen), e.g. coa-cervare
#    - Fraktur/Gothic hyphenation: \u2E17 (double hyphen), e.g. Ge\u2E17witter
# 3. Do not translate; keep the original language (Latin, Greek, German, Estonian, etc.).
# 4. Ligatures:
#    - \u00E6, \u00C6, \u0153, \u0152 \u2013 transcribe exactly as they are
#    - st, ff, fi, fl and other typographic ligatures \u2013 write out as separate letters
# 5. Umlauts and diacritics:
#    - \u00F6, \u00E4, \u00FC, \u00F5 \u2013 always use modern form
#    - u\u0364, o\u0364, a\u0364 (letter + superscript e) \u2013 transcribe as \u00FC, \u00F6, \u00E4
#    - \u00E5, \u00C5 (Swedish) \u2013 keep as is
#    - \u0169, \u00F1, \u00F5 \u2013 keep as is (tilde preserved)
# 6. Special characters:
#    - \u017F (long s) \u2013 transcribe as \u017F
#    - \u00DF (double s) \u2013 transcribe as \u00DF
# 7. Abbreviations:
#    - que abbreviation (\uA757 etc.) \u2013 write as q;
#    - -us abbreviation (\uA770) \u2013 may be expanded
# 8. Signature marks (quire numbers): place at the very end, e.g. A 3
# 9. Page breaks: if the image contains a double-page spread, mark the page break between pages with --lk--.
#
# Return only the exact transcription as plain text."""

# --- 2. MUDELI LAADIMINE ---

logger.info("=== Käivitan In-Place OCR Teenuse (Qwen3.5-9B) ===")
logger.info(f"Baaskaust: {BASE_OCR_KAUST}")
logger.info(f"Mudelid: {MODEL_CONFIGS}")
logger.info(f"Mootorid: {ENGINE_CONFIGS}")

for _mt, _eng in ENGINE_CONFIGS.items():
    if _eng != "llamacpp":
        continue
    _ep = LLAMACPP_ENDPOINTS[_mt]
    # Server peab käima ENNE teenust ja tema pildieelarve peab vastama meie
    # omale. llama.cpp vaikepiir on 4096 visuaaltokenit, meil 5 120 000/1024 =
    # 5000; ilma --image-max-tokens 5000-ta kärbitakse pilt VAIKSELT ja peen
    # kiri kaob (docs/arhiiv/llamacpp-juurdlus-20260827.md, Põhjus 1).
    import json as _json, urllib.request as _url
    try:
        with _url.urlopen(f"{_ep}/health", timeout=10) as _r:
            _r.read()
        _keha = _json.dumps({"model": "x", "max_tokens": 1, "temperature": 0,
                             "messages": [{"role": "user", "content": "x"}]}).encode()
        _req = _url.Request(f"{_ep}/v1/chat/completions", data=_keha,
                            headers={"Content-Type": "application/json"})
        with _url.urlopen(_req, timeout=60) as _r:
            _json.loads(_r.read())
        logger.info(f"llama-server [{_mt}] vastab: {_ep}")
    except Exception as _e:
        logger.error(f"llama-server [{_mt}] ei vasta ({_ep}): {_e}")
        logger.error("Käivita mõlemad serverid enne teenust – vt SPIKKER.md.")
        sys.exit(1)
if "llamacpp" in ENGINE_CONFIGS.values():
    logger.warning("KONTROLLI, et serverid on käivitatud lipuga "
                   "--image-max-tokens 5000 – vaikepiir 4096 kärbib pildi vaikselt.")
logger.info(f"Logi fail: {LOG_FILE}")

if not torch.cuda.is_available():
    logger.error("CUDA puudub!")
    raise RuntimeError("CUDA puudub!")

# Laisk mudeli haldus — laadime ainult kui vaja, vahetame tüübivahel
_current_model_type = None  # type: Optional[str]
model = None
tokenizer = None


def _setup_tokenizer(tok):
    # NB! Kui andmestik ehitatakse --resize lipuga, ei piisa sellest üksi:
    # siis peab ka siin enne protsessorit kutsuma imaging.fit_to_budget(),
    # muidu treenib mudel LANCZOS-pilte ja näeb inferentsil BICUBIC-pilte.
    tok.image_processor.size = {
        "longest_edge": MAX_PIXELS,
        "shortest_edge": tok.image_processor.size.get("shortest_edge", 65536),
    }
    if tok.chat_template and "enable_thinking" in tok.chat_template:
        tok.chat_template = tok.chat_template.replace(
            "enable_thinking=True", "enable_thinking=False"
        )
    return tok


def ensure_model(model_type: str):
    """Laadib mudeli kui pole laetud või tüüp on muutunud.

    HTTP-mootoriga tüübi puhul mudelit ei laeta ja seni laetud unslothi mudel
    VABASTATAKSE: llama-server hoiab ise ~12,7 GB ja mõlemale korraga GPU-l
    ruumi ei ole.
    """
    global _current_model_type, model, tokenizer

    if ENGINE_CONFIGS.get(model_type) == "llamacpp":
        if model is not None:
            logger.info(f"Vabastan unslothi mudeli ({_current_model_type}) – "
                        f"{model_type} kasutab llama-serverit")
            model = None
            tokenizer = None
            _current_model_type = None
            gc.collect()
            torch.cuda.empty_cache()
        return

    if _current_model_type == model_type:
        return

    if model is not None:
        logger.info(f"Vabastab mudeli '{_current_model_type}'...")
        del model
        del tokenizer
        model = None
        tokenizer = None
        _current_model_type = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Mudel vabastatud.")

    model_path = MODEL_CONFIGS[model_type]
    logger.info(f"Laen mudelit '{model_type}': {model_path} ...")
    # bf16, mitte 4-bit: 4-bit on treeningu mälusääst, inferentsil ainult aeglustab
    # (mõõdetud 21.07.2026: bf16 1.5x kiirem; batch 4 tipp 24.6 GB / 32.6 GB,
    #  ei kasva genereerimise pikkusega, sest GatedDeltaNet olek on konstantne)
    m, tok = FastVisionModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    tok = _setup_tokenizer(tok)
    FastVisionModel.for_inference(m)
    model = m
    tokenizer = tok
    _current_model_type = model_type
    logger.info(f"Mudel '{model_type}' laetud.")

# --- 3. ABIFUNKTSIOONID ---

def get_instruction(model_type: str) -> str:
    """Juhis materjalitüübi kaupa.

    HOIATUS – teadaolev lahknevus, mida EI TOHI koos mootorivahetusega parandada:
    teenus on algusest saati saatnud MÕLEMALE tüübile `INSTRUCTION`-i, kuigi
    käsikirjamudel on treenitud `KURRENT_INSTRUCTION`-iga. Kogu Kurrendi
    pariteedimõõtmine (docs/arhiiv/llamacpp-juurdlus-20260827.md) tehti seevastu
    KURRENT_INSTRUCTION-iga, ehk mõõdetud konfiguratsioon ei ole see, mida
    teenus praegu kasutab.

    Siin hoitakse tahtlikult PRAEGUST käitumist, et mootorivahetus jääks ainsaks
    muutujaks. Vahe tuleb enne juhise parandamist ära mõõta:
        venv/bin/python scripts/eval_kurrent.py --prompt print <mudel>
    ja võrrelda vaikimisi (kurrent) jooksuga.
    """
    return INSTRUCTION


def get_chat_template():
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION},
            {"type": "image"},
        ]}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=False,
    )

def strip_output(text: str) -> str:
    """
    Eemaldab mudeli väljundist süsteemi artefaktid:
    - <think>...</think> plokid (Qwen3.5 reasoning tokenid)
    - assistendi markerid
    - markdown koodiplokid
    """
    # Eemalda <think>...</think> plokid (sh tühjad)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Lõika kõik, mis enne assistendi vastust
    for marker in ["</assistant>", "<|assistant|>", "<|im_start|>assistant", "assistant\n"]:
        if marker in text:
            text = text.split(marker, 1)[-1]
    # Eemalda markdown koodiplokid kui peaks esinema
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()

def sanitize_filename(name: str) -> str:
    """
    Teisendab failinime ASCII-sõbralikuks:
    - Normaliseerib unicode (ä→a, ß→ss, jne)
    - Asendab tühikud alakriipsuga
    - Eemaldab erimärgid
    """
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.replace("ß", "ss")
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-.]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def wait_for_file_stable(file_path, check_interval=2, stable_count=2):
    """
    Ootab kuni fail on stabiilne (suurus ei muutu).
    Kasulik suurte failide kopeerimise ootamiseks.
    """
    path = Path(file_path)
    if not path.exists():
        return False

    last_size = -1
    stable_checks = 0

    while stable_checks < stable_count:
        if shutdown_requested:
            return False
        try:
            current_size = path.stat().st_size
        except OSError:
            return False
        if current_size == last_size:
            stable_checks += 1
        else:
            stable_checks = 0
            last_size = current_size
        if stable_checks < stable_count:
            time.sleep(check_interval)

    return True

def expand_pdf(pdf_path):
    """
    Pakib PDF lahti samanimelisse kausta.
    Nt: raamat.pdf → kaust raamat/ → raamat_pg_001.jpg, raamat_pg_002.jpg, ...
    Vigased PDF-id teisaldatakse VIGASED/ kausta.
    """
    pdf_path = Path(pdf_path)

    if str(pdf_path) in failed_pdfs:
        return

    if not wait_for_file_stable(pdf_path):
        logger.info(f"PDF {pdf_path.name} pole veel stabiilne, jätan vahele...")
        return

    safe_name = sanitize_filename(pdf_path.stem)
    output_dir = pdf_path.parent / safe_name

    if output_dir.exists() and any(output_dir.iterdir()):
        return

    logger.info(f"Leidsin PDFi: {pdf_path.name}. Pakin lahti kausta: {output_dir.name}...")
    output_dir.mkdir(exist_ok=True)

    try:
        images = convert_from_path(str(pdf_path), dpi=PDF_DPI, fmt="jpg")
        for i, img in enumerate(images):
            fname = output_dir / f"{safe_name}_pg_{i+1:03d}.jpg"
            img.save(fname, "JPEG", quality=95)
        logger.info(f"PDF lahti pakitud: {len(images)} lehte.")
    except Exception as e:
        logger.error(f"Viga PDF lahtipakkimisel {pdf_path}: {e}")
        failed_pdfs.add(str(pdf_path))
        vigased_dir = Path(BASE_OCR_KAUST) / "VIGASED"
        vigased_dir.mkdir(exist_ok=True)
        try:
            dest = vigased_dir / pdf_path.name
            shutil.move(str(pdf_path), str(dest))
            logger.info(f"Vigane PDF teisaldatud: {dest}")
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
        except Exception as move_err:
            logger.error(f"Ei suutnud vigast PDF-i teisaldada: {move_err}")

# --- KORDUSLOOPI PEATAMINE (VUTT #227) ---
# Mudel satub vahel kordusesse ja genereerib laeni — 4096 tokenit prahti ühe lehe
# kohta, mis on ~15x aeglasem kui terve leht ja hoiab kogu järjekorda kinni.
# Mõõdetud juhtum 2026-08-24: 'Johan ton Crickebo' x452, 99% lehest, 1367 tokenit.
#
# Detektor töötab SÕNADE, mitte token'ite tasemel. Token'ite tasemel oleks periood
# ebastabiilne: BPE lõhub 'Crickebo' mitmeks tükiks ja reavahetused lähevad kaasa,
# nii et sõnaperiood 3 on token'ites 7-9. Sõnapõhine on ühtlasi TÄPSELT sama reegel,
# mis VUTT-i ocr_loop_audit.find_repeat_loop — üks algoritm mõlemas otsas.
#
# Läved on mõõdetud VUTT-i korpusel (21 747 lehte, 2026-08-09): pikim järjestikune
# kordus p95 = 2, p99,5 = 959 — kaks selgelt eraldi populatsiooni, seega läve täpne
# koht on ebaoluline. Peatamine on rangem kui tuvastus (16 vs 10 kordust), sest
# valepositiivi hind on siin kaotatud transkriptsioon, mitte üleliigne hoiatus.
# Periood 5 -> 20 (2026-08-25). Töö 5qdpq4 (47 lk, 20 min) kaotas kaks partiid
# loopidele, mida periood 5 ei näinud: lk 45 'Bruks Dagh / För år D:r Lax' = 6
# sõna x 315 kordust, lk 21 mitmerealine plokk = 17 sõna x 87 kordust. Mõlemad
# genereerisid laeni (4096 tokenit) ja hoidsid kogu partiid 5,5 min kinni, kus
# terve partii teeb ~45 s. MITMEREALINE korduv plokk (tabeli päis + rida) on
# eraldi populatsioon, mida 2026-08-09 mõõtmine ei tabanud — see mõõtis lühikesi
# perioode. Pikem periood ei saa anda ROHKEM kordusi kui lühem, seega on 20
# lühikese perioodi range ülemhulk, mitte selle asendus.
LOOP_MAX_PERIOD = 20      # SÕNADES; 'A B A B' tüüpi loope on 94 juhtu 250-st
LOOP_MIN_REPS = 16        # sügaval tühjas vahemikus kahe populatsiooni vahel
# Saba PEAB mahutama max_period x min_reps sõna, muidu jääb pikk periood ikka
# tabamata: 20 x 16 = 320 sõna, mis varauusaegse ortograafia BPE-tükeldusega on
# ~900-1200 tokenit. 512 mahutas ainult perioodi 5.
LOOP_TAIL_TOKENS = 1536   # dekodeeritav saba; 20 sõna x 16 kordust mahub kindlalt
# Intervall on ALGARV, mitte 16: kontrollisamm ei tohi jaguda korduse
# token-pikkusega. Mõõdetud 2026-08-24: '1/2' on 4 tokenit, intervalliga 16
# maandus iga kontroll täpselt samas faasis (alati sõna keskel) ja 1011 kordust
# jäi 256 kontrolli jooksul tuvastamata.
LOOP_CHECK_EVERY = 13     # sammu


class KordusLoop(Exception):
    """Genereerimine peatati, sest väljund oli kordusloopis."""


def find_tail_loop(sonad, max_period=LOOP_MAX_PERIOD, min_reps=LOOP_MIN_REPS):
    """Kas sõnajärjendi LÕPP on korduv tsükkel? Tagastab (periood, kordused) või None.

    Vaatab ainult saba: loop tuvastatakse siis, kui ta parajasti KESTAB. Varem
    lõppenud kordus (nt loetelu 'I. II. III.') ei tohi genereerimist peatada.
    """
    n = len(sonad)
    for period in range(1, max_period + 1):
        if n < period * min_reps:
            continue
        muster = sonad[n - period:]
        reps = 1
        i = n - 2 * period
        while i >= 0 and sonad[i:i + period] == muster:
            reps += 1
            i -= period
        if reps >= min_reps:
            return period, reps
    return None


class LoopStopper(StoppingCriteria):
    """Peatab kordusesse jäänud RIVI, jättes terved read edasi genereerima.

    prompt_len on kohustuslik: prompt sisaldab pildi kohatäite-tokeneid tuhandeid
    kordi järjest ja ilma lõikamiseta tuvastaks detektor kohe võltsloopi.
    """

    def __init__(self, prompt_len):
        self.prompt_len = prompt_len
        self.looped = {}      # rea indeks -> (periood, kordused, tokeneid)
        self.steps = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.steps += 1
        stop = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for rida in self.looped:
            stop[rida] = True
        genereeritud = input_ids.shape[1] - self.prompt_len
        if self.steps % LOOP_CHECK_EVERY or genereeritud <= 0:
            return stop

        algus = max(self.prompt_len, input_ids.shape[1] - LOOP_TAIL_TOKENS)
        sabad = tokenizer.batch_decode(input_ids[:, algus:], skip_special_tokens=True)
        for rida, saba in enumerate(sabad):
            if rida in self.looped:
                continue
            # Viimane "sõna" on peaaegu alati poolik (kontroll langeb keset
            # tokenit) ja lõhuks saba-tsükli — viskame ta ära.
            leid = find_tail_loop(saba.split()[:-1])
            if leid:
                self.looped[rida] = (leid[0], leid[1], genereeritud)
                stop[rida] = True
                logger.warning(
                    "Kordusloop reas {}: periood {} sõna, {} kordust — peatan "
                    "genereerimise {} tokeni järel".format(
                        rida, leid[0], leid[1], genereeritud))
        return stop


# .err märgendi KATEGOORIAD. Esimene väli failis, sest tellija otsus sõltub
# vea liigist, mitte sõnumist:
#   pilt     — skaneeringut ei saa avada (katki, 0 baiti, vale formaat).
#              Lehte EI SAA käsitsi transkribeerida: pilti ennast ei ole.
#   mudel    — pilt on korras, mudel ei andnud kasutatavat teksti (kordusloop,
#              CUDA OOM, tühi leht). Leht ON imporditav ja täidetav käsitsi.
#   kirjutus — tekst valmis, aga .txt kirjutus ebaõnnestus (nt kadunud kataloog).
#              Tulemus on olemas, aga kadunud — kordus tasub ära, tühjana import mitte.
KAT_PILT = "pilt"
KAT_MUDEL = "mudel"
KAT_KIRJUTUS = "kirjutus"


def write_err_marker(txt_path, exc, kategooria):
    """Kirjutab lehe kõrvale .err märgendi, et tellija saaks vea kohe kätte.

    Ilma selleta ei jää ebaõnnestunud lehest failisüsteemi ühtki jälge: VUTT
    näeb ainult ".txt on olemas / ei ole" ja ootab 12 h absoluuttaimerini.

    Sisu kuju: `{kategooria}: {ErandiTüüp}: {sõnum}` — kategooria on ESIMENE,
    sest tellija otsus (kas lehte saab tühjana importida) sõltub vea liigist.

    Kirjutus ise on best-effort — kataloog võib olla katkestamise järel kadunud
    (VUTT ADR 0024) ja see EI TOHI olla uus krahhiallikas.
    """
    err_path = Path(txt_path).with_suffix(".err")
    msg = "{}: {}: {}".format(kategooria, type(exc).__name__, exc)
    try:
        err_path.write_text(msg[:500] + "\n", encoding="utf-8")
        logger.error("Vea märgend {}: {}".format(err_path.name, msg[:200]))
    except Exception as e:
        logger.error("Ei suutnud .err märgendit kirjutada {}: {}".format(err_path, e))


def process_batch_http(batch_items, model_type):
    """Sama töö llama-serveri kaudu. Sama .err semantika mis unslothi rajal.

    Kolm asja, mis on mõõdetud ja mida ei tohi tagasi keerata
    (docs/arhiiv/llamacpp-juurdlus-20260827.md):

    1. **`imaging.fit_to_grid()`** – pilt viiakse täpselt sellele patch-võrele,
       mida Qwen3.5 protsessor valiks. llama.cpp ümardab ise, aga ilma
       antialiasinguta, ja see hävitab õhukesed tähed.
    2. **PNG, mitte JPEG** – lähtefail on juba JPEG; teine põlvkond sööb
       peene kirja ära.
    3. Server vajab **`--image-max-tokens 5000`**; vaikepiir on 4096 ja pilt
       kärbitakse vaikselt. Seda kontrollitakse käivitamisel.

    Loopi ei saa siin genereerimise ajal peatada (serveri API ei paku
    `StoppingCriteria`-t), seega tuvastatakse ta valminud väljundist
    `loop_detect.is_looped()`-iga – mõõdetult sama hea.
    """
    if not batch_items:
        return

    import base64, io, json, urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def uks(item):
        img_path, txt_path = item
        try:
            with PILImage.open(img_path) as im:
                pilt = fit_to_grid(im.convert("RGB"))
            buf = io.BytesIO()
            pilt.save(buf, "PNG", optimize=False)
        except Exception as e:
            logger.error(f"Viga pildi avamisel {img_path}: {e}")
            write_err_marker(txt_path, e, KAT_PILT)
            return None
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        keha = json.dumps({
            "model": model_type,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": get_instruction(model_type)},
                {"type": "image_url", "image_url": {"url": uri}},
            ]}],
            "max_tokens": 4096,
            "temperature": 0,
            # Mõtlemine on selle peenhäälestuse jaoks jaotusest väljas: mõõdetult
            # jääb <think> tühjaks ja mudel põletab kõik 4096 tokenit.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{LLAMACPP_ENDPOINTS[model_type]}/v1/chat/completions", data=keha,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=LLAMACPP_TIMEOUT) as r:
                vastus = json.loads(r.read())
            return txt_path, strip_output(vastus["choices"][0]["message"]["content"])
        except Exception as e:
            logger.exception(f"llama-server päring ebaõnnestus {img_path}: {e}")
            write_err_marker(txt_path, e, KAT_MUDEL)
            return None

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
        tulemused = list(pool.map(uks, batch_items))

    for t in tulemused:
        if t is None:
            continue
        txt_out_path, clean_text = t
        leid = is_looped(clean_text)
        if leid:
            periood, kordused = leid
            write_err_marker(txt_out_path, KordusLoop(
                "periood {} sõna, {} kordust — tuvastatud valminud väljundist".format(
                    periood, kordused)), KAT_MUDEL)
            continue
        if not clean_text.strip():
            logger.warning(f"Tühi väljund: {os.path.basename(txt_out_path)} — "
                           f"mudel ei genereerinud teksti")
        try:
            with open(txt_out_path, "w", encoding="utf-8") as f:
                f.write(clean_text)
        except Exception as e:
            logger.error(f"Ei suutnud kirjutada {txt_out_path}: {e}")
            write_err_marker(txt_out_path, e, KAT_KIRJUTUS)
            continue
        logger.info(f"Transkribeeritud: {os.path.basename(txt_out_path)}")


def process_batch(batch_items):
    """
    Töötleb ühe batchi pilte.
    batch_items: list tuple'itest (pildi_täistee, txt_väljundi_täistee)

    Pildi suuruse piiramine käib tokenizer.image_processor kaudu (5M px),
    käsitsi resize'i pole vaja – image_processor skaleerib automaatselt.
    """
    if not batch_items:
        return

    images_pil = []
    valid_items = []

    for img_path, txt_path in batch_items:
        try:
            img = PILImage.open(img_path).convert("RGB")
            images_pil.append(img)
            valid_items.append((img_path, txt_path))
        except Exception as e:
            logger.error(f"Viga pildi avamisel {img_path}: {e}")
            write_err_marker(txt_path, e, KAT_PILT)

    if not images_pil:
        return

    try:
        chat_template = get_chat_template()
        inputs = tokenizer(
            images_pil,
            [chat_template] * len(images_pil),
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        stopper = LoopStopper(prompt_len=inputs["input_ids"].shape[1])
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                use_cache=True,
                stopping_criteria=[stopper],
            )

        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        # Terve batch kukkus (nt CUDA OOM). Varem propageerus erand main_loop'ist
        # mooduli tasemele, kus on sys.exit(1) — see tappis TERVE teenuse ja
        # katkestas kõigi kasutajate järjekorra. Nüüd saab iga selle batchi leht
        # .err märgendi ja tsükkel jätkab järgmisega.
        logger.exception(f"Batchi töötlemine ebaõnnestus: {e}")
        for _, txt_out_path in valid_items:
            write_err_marker(txt_out_path, e, KAT_MUDEL)
        for img in images_pil:
            img.close()
        torch.cuda.empty_cache()
        return

    for i, raw_text in enumerate(decoded_texts):
        _, txt_out_path = valid_items[i]
        if i in stopper.looped:
            # Loopinud väljund EI OLE transkriptsioon — .err jätab otsuse
            # tellijale (kustuta leht või proovi uuesti), .txt peidaks prahi ära.
            periood, kordused, tokeneid = stopper.looped[i]
            write_err_marker(txt_out_path, KordusLoop(
                "periood {} sõna, {} kordust — genereerimine peatatud {} tokeni järel".format(
                    periood, kordused, tokeneid)), KAT_MUDEL)
            continue
        clean_text = strip_output(raw_text)
        if not clean_text.strip():
            logger.warning(f"Tühi väljund: {os.path.basename(txt_out_path)} — mudel ei genereerinud teksti")
        try:
            with open(txt_out_path, "w", encoding="utf-8") as f:
                f.write(clean_text)
        except Exception as e:
            # Kataloog võib olla katkestamise järel kadunud (VUTT ADR 0024).
            # Üksik kirjutusviga ei tohi tsüklit ega teenust katkestada.
            logger.error(f"Ei suutnud kirjutada {txt_out_path}: {e}")
            write_err_marker(txt_out_path, e, KAT_KIRJUTUS)
            continue
        logger.info(f"Transkribeeritud: {os.path.basename(txt_out_path)}")

    for img in images_pil:
        img.close()
    del inputs, outputs, decoded_texts, images_pil
    torch.cuda.empty_cache()

# --- 4. PEAMINE TÖÖTSÜKKEL ---

HEARTBEAT_INTERVAL = 60

def main_loop():
    last_heartbeat = time.time()

    # Loo alamkaustad kui puuduvad
    for mt in MODEL_CONFIGS:
        Path(BASE_OCR_KAUST, mt).mkdir(parents=True, exist_ok=True)

    while not shutdown_requested:
        # 1. Paki lahti PDF-id mõlemas alamkaustas
        for mt in MODEL_CONFIGS:
            scan_root = Path(BASE_OCR_KAUST, mt)
            for pdf in list(scan_root.rglob("*.pdf")):
                if shutdown_requested:
                    break
                expand_pdf(pdf)

        # 2. Kogu töötlemata pildid tüübi järgi
        tasks_by_type: dict = {mt: [] for mt in MODEL_CONFIGS}

        for mt in MODEL_CONFIGS:
            scan_root = Path(BASE_OCR_KAUST, mt)
            candidates = natsorted(
                [f for f in scan_root.rglob("*")
                 if f.suffix.lower() in EXTENSIONS and f.is_file()],
                key=lambda x: str(x)
            )
            # .err märgend on LÕPLIK: ilma selle tingimuseta võtaks teenus
            # vigase lehe igal tsüklil uuesti ette, põletaks GPU-d ja kirjutaks
            # märgendi lõputult üle. Kordus = tellija kustutab .err faili.
            tasks_by_type[mt] = [
                (str(img), str(img.with_suffix(".txt")))
                for img in candidates
                if not img.with_suffix(".txt").exists()
                and not img.with_suffix(".err").exists()
            ]

        total = sum(len(v) for v in tasks_by_type.values())

        # 3. Töötle tüüp-tüübi kaupa (minimeerib mudeli vahetusi)
        if total > 0:
            logger.info(f"Leidsin {total} pilti ({', '.join(f'{mt}:{len(tasks_by_type[mt])}' for mt in MODEL_CONFIGS)}).")
            for mt, tasks in tasks_by_type.items():
                if not tasks or shutdown_requested:
                    continue
                ensure_model(mt)
                for i in range(0, len(tasks), BATCH_SIZE):
                    if shutdown_requested:
                        break
                    batch = tasks[i:i + BATCH_SIZE]
                    logger.info(f"[{mt}] Töötlen {i+1}–{min(i+BATCH_SIZE, len(tasks))} / {len(tasks)}")
                    if ENGINE_CONFIGS.get(mt) == "llamacpp":
                        process_batch_http(batch, mt)
                    else:
                        process_batch(batch)
                    gc.collect()
            logger.info("Kõik hetke tööd tehtud. Ootan uusi...")
            last_heartbeat = time.time()
        else:
            if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
                logger.info("Heartbeat: teenus töötab, ootan uusi faile...")
                last_heartbeat = time.time()

        time.sleep(5)

    logger.info("Teenus peatatud.")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("Skript peatatud (Ctrl+C).")
    except Exception as e:
        logger.exception(f"Kriitiline viga: {e}")
        sys.exit(1)

"""
Batch: Gemini few-shot OCR freilingshauseni lehekuljele 100-200,
kolme naitega (63, 70, 229), et koguda mustandtekst treeningandmestiku
jaoks (kasutaja proofib ule enne VUTTi panekut).
"""
import google.generativeai as genai
import PIL.Image
import os
import time
import threading
import queue
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY puudub .env failis")

genai.configure(api_key=api_key)
MODEL_NAME = 'gemini-3.1-flash-lite'  # LLM! TÄHTIS!!! Ära seda mudelit ise muuda kunagi!!!

DATA_DIR = "/home/mf/Dokumendid/LLM/qwen3.5/data/freilingshausen-2col/raw"

PROMPT = """Transkribeeri kogu tekst sellel vanas fraktuurkirjas leheküljel. Reeglid:
1. Säilita pikk s (ſ) ja ß täpselt nagu originaalis, ära asenda tavalise s-iga.
2. Fraktuuri poolitusmärk on ⸗ (topeltkriips), mitte tavaline sidekriips -.
3. Ära tõlgi ega moderniseeri õigekirja, transkribeeri täpselt nagu kirjutatud.
4. Lehekülg on TIHTI kahes tulbas. Kui nii, transkribeeri vasak tulp algusest
   lõpuni, seejärel jätka KOHE parema tulbaga, ühtse jooksva tekstina.
   Ära lisa tulpade vahele mingit märgendit ega päisemärki, ära jäta
   parempoolset tulpa vahele.
5. Kui lehe tipus on kummaski veerus katkendrida (eelmise lehe jätk), pane
   need järjest kirja (vasak veerg, siis parem veerg) enne pealkirja/nooti.
6. Kui lehel on nooditekst (noodijoonestik), asenda see ühe reaga: <noodid>
7. Laulu/psalmi numbrid ja rubriigipealkirjad (nt "10.", "23.") märgi
   <b>...</b> tagidega, nagu näidetes. Viitereal (nt "1. Th. 4.", "Aus 3.10.")
   ja meloodiareal (Mel. ...) ei kasuta <b>-tagi.
8. Väljasta ainult puhas transkriptsioon, ilma markdown koodiplokkideta,
   ilma kommentaarideta.
"""

examples = [
    {
        "image": os.path.join(DATA_DIR, "bsb10592597_00063.jpg"),
        "text": os.path.join(DATA_DIR, "bsb10592597_00063.txt"),
    },
    {
        "image": os.path.join(DATA_DIR, "bsb10592597_00070.jpg"),
        "text": os.path.join(DATA_DIR, "bsb10592597_00070.txt"),
    },
    {
        "image": os.path.join(DATA_DIR, "bsb10592597_00229.jpg"),
        "text": os.path.join(DATA_DIR, "bsb10592597_00229.gemini-test.txt"),
    },
]

PAGE_START = 100
PAGE_END = 200  # kaasav

OUT_SUFFIX = ".gemini.txt"


def clean(text):
    text = text.strip()
    for fence in ("```markdown", "```"):
        if text.startswith(fence):
            text = text[len(fence):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def build_prompt_parts(target_image_path):
    parts = [PROMPT]
    for ex in examples:
        parts.append(PIL.Image.open(ex["image"]))
        with open(ex["text"], "r", encoding="utf-8") as f:
            parts.append(f"Vastus:\n{f.read()}\n")
    parts.append(PIL.Image.open(target_image_path))
    return parts


def process_one(name, max_retries=2, retry_delay=5):
    img_path = os.path.join(DATA_DIR, name + ".jpg")
    out_path = os.path.join(DATA_DIR, name + OUT_SUFFIX)
    if os.path.exists(out_path):
        return True, "juba olemas"
    if not os.path.exists(img_path):
        return False, "pilt puudub"

    model = genai.GenerativeModel(MODEL_NAME)
    parts = build_prompt_parts(img_path)

    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(parts)
            if response and response.text:
                text = clean(response.text)
                if text:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    return True, f"{len(text)} chars"
            time.sleep(retry_delay)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return False, str(e)
    return False, "tuhi vastus"


def worker(q, results, lock):
    while True:
        try:
            name = q.get_nowait()
        except queue.Empty:
            return
        ok, info = process_one(name)
        with lock:
            results.append((name, ok, info))
        marker = "OK" if ok else "FAIL"
        print(f"[{marker}] {name}: {info}")
        q.task_done()


if __name__ == "__main__":
    names = [f"bsb10592597_{p:05d}" for p in range(PAGE_START, PAGE_END + 1)]
    q = queue.Queue()
    for n in names:
        q.put(n)

    results = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(q, results, lock)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok_count = sum(1 for _, ok, _ in results if ok)
    print(f"\nKokku: {len(results)}, onnestus: {ok_count}, ebaonnestus: {len(results) - ok_count}")
    for n, ok, info in results:
        if not ok:
            print(f"EBAONNESTUS: {n}: {info}")

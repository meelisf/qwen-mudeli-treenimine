"""
Test: kas Gemini (few-shot, 2 näidet) saab hakkama freilingshauseni
kahetulbaliste lehekülgedega, kus praegune Qwen3.5 print-mudel
kas jookseb <m>-silmusesse voi jatab veeru 2 valikult vahele.

Naited: bsb10592597_00063 (2 tulpa + noot + laulunumber <b>),
        bsb10592597_00070 (2 tulpa + laulunumber <b>)
Testitavad: 00071 (vahepealkiri poleb), 00087 (vaikiv veeru-2 kadu),
            00229 (noot + 2 tulpa, praegu <m>-silmus)
"""
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY puudub .env failis")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3.1-flash-lite')  # LLM! TÄHTIS!!! Ära seda mudelit ise muuda kunagi!!!

DATA_DIR = "/home/mf/Dokumendid/LLM/qwen3.5/data/freilingshausen-2col/raw"

PROMPT = """Transkribeeri kogu tekst sellel vanas fraktuurkirjas leheküljel. Reeglid:
1. Säilita pikk s (ſ) ja ß täpselt nagu originaalis, ära asenda tavalise s-iga.
2. Fraktuuri poolitusmärk on ⸗ (topeltkriips), mitte tavaline sidekriips -.
3. Ära tõlgi ega moderniseeri õigekirja, transkribeeri täpselt nagu kirjutatud.
4. Lehekülg on TIHTI kahes tulbas. Kui nii, transkribeeri vasak tulp algusest
   lõpuni, seejärel jätka KOHE parema tulbaga, ühtse jooksva tekstina.
   Ära lisa tulpade vahele mingit märgendit ega päisemärki, ära jäta
   parempoolset tulpa vahele.
5. Kui lehel on nooditekst (noodijoonestik), asenda see ühe reaga: <noodid>
6. Laulu/psalmi numbrid ja rubriigipealkirjad (nt "10.", "23.") märgi
   <b>...</b> tagidega, nagu näidetes.
7. Väljasta ainult puhas transkriptsioon, ilma markdown koodiplokkideta,
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

TARGETS = ["bsb10592597_00887", "bsb10592597_00100"]


def build_prompt_parts(target_image_path):
    parts = [PROMPT]
    for ex in examples:
        parts.append(PIL.Image.open(ex["image"]))
        with open(ex["text"], "r", encoding="utf-8") as f:
            parts.append(f"Vastus:\n{f.read()}\n")
    parts.append(PIL.Image.open(target_image_path))
    return parts


def clean(text):
    text = text.strip()
    for fence in ("```markdown", "```"):
        if text.startswith(fence):
            text = text[len(fence):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


if __name__ == "__main__":
    for name in TARGETS:
        img_path = os.path.join(DATA_DIR, name + ".jpg")
        out_path = os.path.join(DATA_DIR, name + ".gemini-test.txt")
        print(f"=== {name} ===")
        parts = build_prompt_parts(img_path)
        response = model.generate_content(parts)
        text = clean(response.text) if response and response.text else ""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"-> {out_path} ({len(text)} chars)")

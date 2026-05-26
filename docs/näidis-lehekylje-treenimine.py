import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from unsloth import FastVisionModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TextStreamer
from PIL import Image as PILImage

# --- 1. SAMM: Andmestiku asukoha määramine (jääb samaks) ---
DATA_ROOT_DIR = "andmed/lehekyljed" 
CSV_PATH = os.path.join(DATA_ROOT_DIR, "metadata.csv")
IMAGES_DIR = os.path.join(DATA_ROOT_DIR, "images") 

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Andmestiku faili ei leitud asukohast: {CSV_PATH}")
if not os.path.exists(IMAGES_DIR):
    raise FileNotFoundError(f"Piltide kausta ei leitud asukohast: {IMAGES_DIR}")
print(f"Andmestik laetakse asukohast: {DATA_ROOT_DIR}")

# --- 2. SAMM: Mudeli laadimine ja KÄRPIMISE KEELAMINE ---
if not torch.cuda.is_available():
    raise RuntimeError("CUDA ei ole saadaval.")

model_name = "unsloth/Qwen3-VL-8B-Instruct"

# Laeme mudeli ja tokenizeri täiesti tavaliselt
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# === OLULINE PARANDUS #1: Keelame kärpimise otse laaditud tokenizeri objektil ===
# See lahendab "Mismatch in `image` token count" vea.
tokenizer.truncation = False

model = FastVisionModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
    finetune_vision_layers=True, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
)
print("Mudel on edukalt laaditud ja ette valmistatud LoRA treeninguks.")
model.print_trainable_parameters()

# --- 3. SAMM: Andmestiku laadimine ja ettevalmistamine (jääb samaks) ---

dataset = load_dataset("csv", data_files=CSV_PATH, split="train")

def resolve_path(example):
    example['failinimi'] = os.path.join(IMAGES_DIR, os.path.basename(example['failinimi']))
    return example
dataset = dataset.map(resolve_path)
print("Piltide asukohad on lisatud andmestikku.")

instruction = """You are an expert OCR assistant for historical documents.

Instructions:
1. Transcribe the entire page from the provided image.
2. Preserve original line breaks and hyphenation.
3. Do not translate; keep the original language (Latin, Greek, etc.).
4. Carefully preserve historical characters and ligatures, including:
   - long s (ſ), and ligatures like ſſ, ſi, ſt,
   - æ, œ, and similar.
5. Include visible page numbers if present.

Return only the exact transcription as plain text."""

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["failinimi"]}
          ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["transkriptsioon"]} ]
        },
    ]
    return { "messages" : conversation }

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
print("\nAndmestik on teisendatud 'messages' formaati.")

# --- 4. SAMM: Treeneri (Trainer) seadistamine ---

FastVisionModel.for_training(model)

    
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=converted_dataset,
    
    # === LÕPLIK PARANDUS: Lisame max_length ka siia ===
    data_collator=UnslothVisionDataCollator(model, tokenizer, resize="max", max_seq_length=32768),
    
    args=SFTConfig(
        # Hoiame max_length ka siin, et tagada ühilduvus
        max_length=32768,
        
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=2, 
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_pages",
        report_to="none",
        
        remove_unused_columns=False, 
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_num_workers=0, 
    ),
)

  
print("\nTreener on korrektselt seadistatud. Alustan treeningut...")
trainer.train()

print("Treening on lõppenud!")

# --- 5. SAMM: Mudeli salvestamine ---
final_model_path = "qwen-ocr-finetuned-greek"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Treenitud LoRA adapter on salvestatud kausta: {final_model_path}")

# --- 6. SAMM: Järeldamine (Inference) ---
print("\nAlustan järeldamist, et testida treenitud mudelit...")

FastVisionModel.for_inference(model)

try:
    test_image_path = os.path.join(IMAGES_DIR, "0004_Gezelius-Comenius-Ianua-0004.jpg")
    image_to_test = PILImage.open(test_image_path)
    print(f"Testpilt laaditud: {test_image_path}")
except FileNotFoundError:
    print(f"Testpilti ei leitud asukohast: {test_image_path}")
    first_row = load_dataset("csv", data_files=CSV_PATH, split="train[0]")
    test_image_path = os.path.join(IMAGES_DIR, os.path.basename(first_row['failinimi']))
    image_to_test = PILImage.open(test_image_path)
    print(f"Kasutan esimest pilti andmestikust: {test_image_path}")

print(f"Testpilt laaditud. Suurus: {image_to_test.size}.")

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "image"},
    ]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image_to_test,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=True)

decoded_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

print("\n--- TULEMUSED ---")
print(f"Mudeli genereeritud vastus:\n{decoded_text}")
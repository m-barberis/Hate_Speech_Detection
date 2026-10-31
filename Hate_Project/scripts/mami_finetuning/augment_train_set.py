import os
import json
import random
import pandas as pd
from pathlib import Path
from PIL import Image

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = Path("/scratch/datasets/MAMI/train")
CLEAN_IMG_DIR = BASE_DIR 
AUG_IMG_DIR = BASE_DIR / "images_augmented"
CSV_FILE = BASE_DIR / "training.csv"
JSONL_OUT = BASE_DIR / "mami_train_augmented.jsonl"
EMOJI_DIR = Path("/scratch/utils/emoticons/emoticons_png/512")

# ==============================================================================
# EMOJI ASSETS (ZERO-SHOT ROBUSTNESS SPLIT)
# ==============================================================================
# Held out completely from your 20-emoji test set to prove generalized robustness
NEW_TRAIN_EMOJIS = {
    "1f47d": "alien",
    "1f47b": "ghost",
    "1f4a3": "bomb",
    "1f911": "money_mouth",
    "1f92e": "face_vomiting",
    "1f34e": "red_apple",
    "1f680": "rocket",
    "1f346": "eggplant",
    "1f6d1": "stop_sign",
    "1f4a7": "droplet"
}

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================
def load_emoji_assets() -> list:
    assets = []
    print("Loading training emoji assets...")
    for hex_code, name in NEW_TRAIN_EMOJIS.items():
        path1 = EMOJI_DIR / f"emoji_u{hex_code}.png"
        path2 = EMOJI_DIR / f"emoji_u{hex_code.upper()}.png"
        
        if path1.exists():
            assets.append(Image.open(path1).convert("RGBA"))
            print(f"  [+] Loaded {name} ({path1.name})")
        elif path2.exists():
            assets.append(Image.open(path2).convert("RGBA"))
            print(f"  [+] Loaded {name} ({path2.name})")
        else:
            print(f"  [-] Warning: Asset for {name} not found. Skipping.")
    return assets

def apply_4x4_grid(base_img: Image.Image, emoji_img: Image.Image, alpha: float = 0.8) -> Image.Image:
    w, h = base_img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    min_side = min(w, h)
    e_size = int(min_side * 0.25)
    emoji = emoji_img.resize((e_size, e_size), resample=Image.Resampling.LANCZOS)
    
    r, g, b, a = emoji.split()
    a = a.point(lambda p: int(p * alpha))
    emoji = Image.merge("RGBA", (r, g, b, a))
    
    for i in range(4):
        for j in range(4):
            x = int((i + 0.5) * (w / 4) - e_size / 2)
            y = int((j + 0.5) * (h / 4) - e_size / 2)
            overlay.paste(emoji, (x, y), emoji)
            
    return Image.alpha_composite(base_img.convert("RGBA"), overlay).convert("RGB")

def construct_prompts(text_transcript: str, drop_probability: float = 0.5, rng=None):
    """
    Returns exact system and user prompts matching the inference script.
    Implements 50% text dropout.
    """
    system_prompt = "You are an automated classification API. You must output EXACTLY ONE WORD. Do not output any full sentences, explanations, or prefixes."
    
    base_task = (
        "Task: Identify if this meme contains misogyny.\n"
        "Allowed Labels: 'misogynous' or 'clean'.\n"
        "Answer:"
    )
    
    rng = random if rng is None else rng
    if rng.random() > drop_probability and text_transcript.strip():
        user_prompt = f"Meme Text: '{text_transcript}'\n\n{base_task}"
    else:
        user_prompt = base_task
        
    return system_prompt, user_prompt

# ==============================================================================
# MAIN GENERATION PIPELINE
# ==============================================================================
def main():
    AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    print(f"Parsing labels and transcripts from {CSV_FILE.name}...")
    
    dataset = []
    
    # Using Pandas for robust parsing to avoid the csv.DictReader bug
    df = pd.read_csv(CSV_FILE, sep='\t', encoding='utf-8')
    df.columns = df.columns.str.strip() # Clean headers
    
    for _, row in df.iterrows():
        filename = str(row.get('file_name', '')).strip()
        
        if not filename or filename == 'nan':
            continue
            
        # Parse label safely
        is_misogynous = int(row.get('misogynous', 0)) == 1
        
        # Parse text safely (handle pandas nan)
        text = str(row.get('Text Transcription', '')).strip()
        if text == 'nan':
            text = ""
            
        dataset.append({
            "filename": filename,
            "target": "misogynous" if is_misogynous else "clean",
            "text": text
        })

    print(f"Loaded {len(dataset)} training examples.")
    
    emoji_assets = load_emoji_assets()
    if not emoji_assets:
        raise FileNotFoundError("No emoji assets found.")

    jsonl_records = []
    processed_count = 0

    print("Initializing augmentation pipeline...")
    for data in dataset:
        clean_img_path = CLEAN_IMG_DIR / data["filename"]
        
        # MAMI dataset mixes .jpg and .png, check for both if necessary
        if not clean_img_path.exists():
            alt_path = clean_img_path.with_suffix('.png')
            if alt_path.exists():
                clean_img_path = alt_path
            else:
                continue
            
        try:
            # Generate Prompts (Independently rolled for clean vs augmented)
            sys_clean, user_clean = construct_prompts(data["text"])
            sys_aug, user_aug = construct_prompts(data["text"])

            # --- A. Handle Clean Image Record ---
            jsonl_records.append({
                "image_path": str(clean_img_path),
                "system_prompt": sys_clean,
                "user_prompt": user_clean,
                "target": data["target"],
                "is_augmented": False
            })

            # --- B. Handle Augmented Image Record ---
            aug_filename = f"aug_{clean_img_path.name}"
            aug_img_path = AUG_IMG_DIR / aug_filename
            
            base_img = Image.open(clean_img_path)
            selected_emoji = random.choice(emoji_assets)
            
            # Apply semantic trap at Alpha = 0.8
            augmented_img = apply_4x4_grid(base_img, selected_emoji, alpha=0.8)
            augmented_img.save(aug_img_path, format="JPEG")
            
            jsonl_records.append({
                "image_path": str(aug_img_path),
                "system_prompt": sys_aug,
                "user_prompt": user_aug,
                "target": data["target"],
                "is_augmented": True
            })
            
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"  -> Processed {processed_count}/{len(dataset)} base images...")
                
        except Exception as e:
            print(f"  [-] Error processing {data['filename']}: {e}")

    print(f"Writing {len(jsonl_records)} total records to {JSONL_OUT}...")
    with open(JSONL_OUT, 'w', encoding='utf-8') as f:
        for record in jsonl_records:
            f.write(json.dumps(record) + '\n')

    print("\n✅ Step A Complete. Augmentation dataset successfully generated.")

if __name__ == "__main__":
    main()

import os
import requests
from tqdm import tqdm

MODEL_URL = "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf?download=true"
MODEL_FILENAME = "LFM2.5-350M-Q4_K_M.gguf"

def ensure_model_exists():
    """
    Checks if the GGUF model exists. If not, prompts the user to download it.
    Returns True if the model is available (either existed or downloaded successfully),
    False if the user opted out or download failed.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_dir = os.path.join(base_dir, 'models', 'llm')
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    
    if os.path.exists(model_path):
        return True
        
    print("\n[Warning] The 200MB PathoIntern LLM model is missing.")
    user_input = input("Would you like to download it now from Hugging Face? (y/n): ")
    
    if user_input.strip().lower() != 'y':
        return False
        
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading {MODEL_FILENAME}...")
    try:
        response = requests.get(MODEL_URL, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(model_path, 'wb') as f, tqdm(
            desc=MODEL_FILENAME,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
                
        print(f"\n[Success] Model successfully downloaded to {model_path}")
        return True
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return False

if __name__ == "__main__":
    ensure_model_exists()

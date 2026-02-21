import subprocess
import time

def stop_ollama_model(model_name: str, timeout: int = 15):
    """Force stop an Ollama model and wait for it to clear VRAM."""
    try:
        # 1. Issue stop command
        subprocess.run(["ollama", "stop", model_name], check=False, capture_output=True)
        print(f"🛑 Stop signal sent to: {model_name}")
        
        # 2. Poll 'ollama ps' until model is gone or timeout reached
        start_time = time.time()
        while time.time() - start_time < timeout:
            ps_output = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
            if model_name not in ps_output:
                print(f"✅ Model {model_name} cleared from VRAM.")
                return True
            time.sleep(1)
            
        print(f"⚠️ Timeout: Model {model_name} still stopping after {timeout}s.")
        return False
    except Exception as e:
        print(f"⚠️ VRAM util error: {e}")
        return False

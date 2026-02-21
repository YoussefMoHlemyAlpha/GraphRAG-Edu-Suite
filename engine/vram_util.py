import subprocess

def stop_ollama_model(model_name: str):
    """Force stop an Ollama model to free up VRAM."""
    try:
        # We use 'ollama stop' command
        subprocess.run(["ollama", "stop", model_name], check=False, capture_output=True)
        print(f"🛑 Stopped model: {model_name}")
    except Exception as e:
        print(f"⚠️ Could not stop model {model_name}: {e}")

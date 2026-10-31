"""[D64] Minimal check: can this workspace actually get an L4 GPU?"""
import modal
app = modal.App("gpu-probe")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, gpu="L4", timeout=300)
def g():
    import subprocess
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout


@app.function(image=image, timeout=300)
def c():
    return "cpu ok"


@app.local_entrypoint()
def main():
    print("CPU:", c.remote())
    print("GPU:", g.remote())

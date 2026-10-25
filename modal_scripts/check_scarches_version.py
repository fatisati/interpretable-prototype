"""[D64] What scarches does Modal actually install, and does its scPoli NB path
match the local checkout at F:\codes\scarches (0.6.1)?"""
import modal

app = modal.App("scarches-version-check")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "scipy", "pandas", "anndata", "scanpy", "torch",
                      "scarches", "tqdm"))


@app.function(image=image, timeout=900)
def check():
    import inspect
    import anndata
    if not hasattr(anndata, "read"):
        anndata.read = anndata.read_h5ad      # shim: scarches 0.6.x imports it
    import scarches
    out = {"version": getattr(scarches, "__version__", "?")}
    import scarches.models.scpoli.scpoli as sp
    out["classes_in_scpoli_py"] = [n for n in dir(sp) if n[:1].isupper()]
    cls = None
    for n in ("scpoli", "scPoli", "SCPOLI"):
        if hasattr(sp, n):
            cls = getattr(sp, n); out["class_used"] = n; break
    out["forward_src"] = inspect.getsource(cls.forward) if cls else "NO CLASS FOUND"
    try:
        import scarches.trainers.scpoli.trainer as tr
        out["trainer_src_keys"] = [l for l in inspect.getsource(tr).splitlines()
                                   if "sizefactor" in l or "combined_batch" in l]
    except Exception as e:
        out["trainer_src_keys"] = [f"(trainer import failed: {e})"]
    return out


@app.local_entrypoint()
def main():
    r = check.remote()
    print("MODAL scarches version:", r["version"])
    print("\n=== scPoli.forward NB branch (MODAL) ===")
    s = r["forward_src"]
    i = s.find('elif self.recon_loss == "nb"')
    print(s[:s.find("batch_embeddings")+200][:400] if i < 0 else s[i:i+700])
    print("\n=== trainer lines mentioning sizefactor / combined_batch ===")
    for l in r["trainer_src_keys"][:20]:
        print("   ", l.strip())

"""[D64] Verify the migrated h5ad files on the volume are intact (size + h5py open)."""
import modal
app = modal.App("scproto-volcheck")
image = modal.Image.debian_slim(python_version="3.11").pip_install("h5py", "numpy")
volume = modal.Volume.from_name("scproto-data")

FILES = ["/vol/data/node_heldout/pbmc-immune_arbf_ctb20_seed0_train.h5ad",
         "/vol/data/scpoli/Immune_ALL_human_hvg.h5ad"]


@app.function(image=image, timeout=900, volumes={"/vol": volume})
def check():
    import os, hashlib, h5py
    out = []
    for p in FILES:
        r = {"path": p, "exists": os.path.exists(p)}
        if r["exists"]:
            r["size"] = os.path.getsize(p)
            h = hashlib.md5()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(8 << 20), b""):
                    h.update(blk)
            r["md5"] = h.hexdigest()
            try:
                with h5py.File(p, "r") as f:
                    r["keys"] = list(f.keys())[:6]
                    r["h5py_ok"] = True
            except Exception as e:
                r["h5py_ok"] = False
                r["err"] = f"{type(e).__name__}: {e}"
        out.append(r)
    return out


@app.local_entrypoint()
def main():
    for r in check.remote():
        print(f"{r['path']}")
        print(f"   exists={r['exists']} size={r.get('size'):,} h5py_ok={r.get('h5py_ok')}")
        print(f"   md5={r.get('md5')}")
        if r.get("err"):
            print(f"   ERROR {r['err']}")

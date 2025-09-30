import numba
from scib_metrics.benchmark import Benchmarker

# numba.config.CACHE_DIR = ''  # Proper way to disable caching
numba.config.CACHE = False

import sys
import argparse
import os
os.environ["TMPDIR"] = os.path.expanduser("~/tmp")

# os.environ["JAX_ENABLE_X64"] = "0"
# os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"

# import jax
# jax.config.update("jax_enable_x64", False)

def get_trainer(model, parser):
    if model == "scproto":
        from interpretable_ssl.trainers.scproto import SCProtoTrainer
        return SCProtoTrainer(parser = parser)
    elif model == "scpoli":
        from interpretable_ssl.trainers.scpoli_original import OriginalTrainer
        return OriginalTrainer(parser = parser)
    else:
        raise ValueError(f"Unknown model name: {model}")

def main():
    if len(sys.argv) < 2:
        raise ValueError("Model name must be provided as the first argument.")
    
    model = sys.argv[1]
    print(model)
    sys.argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=f"{model} Trainer Parameters")
    
    trainer = get_trainer(model, parser)
    trainer.setup()
    print(trainer.dump_path, " has been set up")
    trainer.run()

if __name__ == "__main__":
    print('-----main started----')
    main()

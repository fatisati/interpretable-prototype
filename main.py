# import numba
# numba.config.CACHE_DIR = ''  # Proper way to disable caching
# numba.config.CACHE = False

import sys
import argparse
import os

os.environ["TMPDIR"] = os.path.expanduser("~/tmp")
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def get_trainer(model, parser, mode):
    if model == "swav":
        from interpretable_ssl.trainers.scproto import SCProtoTrainer

        return SCProtoTrainer(parser=parser, mode=mode)
    elif model == "scpoli":
        from interpretable_ssl.trainers.scpoli_original import OriginalTrainer

        return OriginalTrainer(parser=parser, mode=mode)
    else:
        raise ValueError(f"Unknown model name: {model}")


def main():
    if len(sys.argv) < 2:
        raise ValueError("Model name must be provided as the first argument.")

    model = sys.argv[1]
    ds_id = sys.argv[2]
    mode = sys.argv[3]
    print(mode)
    if model == "seacell":
        from seacell_train import train_seacell
        train_seacell(ds_id, mode)
        return

    sys.argv = sys.argv[3:]
    parser = argparse.ArgumentParser(description=f"{model} Trainer Parameters")

    trainer = get_trainer(model, parser, mode)
    trainer.setup()
    print(trainer.dump_path, " has been set up")
    trainer.run()


if __name__ == "__main__":
    print("-----main started----")
    main()

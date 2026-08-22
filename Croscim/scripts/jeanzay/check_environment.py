#!/usr/bin/env python3
"""Fail fast when the Jean Zay environment misses a training dependency."""

import importlib
import os
import sys


REQUIRED_MODULES = (
    "torch",
    "pytorch_lightning",
    "hydra",
    "omegaconf",
    "numpy",
    "pandas",
    "xarray",
    "zarr",
    "dask",
    "netCDF4",
    "scipy",
    "matplotlib",
    "kornia",
    "skimage",
    "PIL",
    "yaml",
    "tqdm",
    "tensorboard",
    "contrib.SST.data_multires",
    "contrib.SST.models",
)


def main():
    requested_arch = os.environ.get("CROSCIM_ARCH_MODULE")
    loaded_modules = os.environ.get("LOADEDMODULES", "").split(":")
    if requested_arch and requested_arch not in loaded_modules:
        print(
            f"Required Jean Zay architecture module is not loaded: {requested_arch}",
            file=sys.stderr,
        )
        return 1

    missing = []
    versions = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        versions.append(f"{module_name}={getattr(module, '__version__', 'unknown')}")

    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print(f"Architecture module: {requested_arch or 'default'}")
    if missing:
        print("Missing or broken required modules:", file=sys.stderr)
        for failure in missing:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("Croscim environment OK")
    print("\n".join(versions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

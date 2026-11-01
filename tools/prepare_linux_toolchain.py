"""Prepare conventional linker names inside a private pip CUDA toolchain."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    lib = root / "lib"
    runtime = lib / "libcudart.so.13"
    if not (root / "bin/nvcc").is_file() or not runtime.is_file():
        raise ValueError("expected private pip CUDA toolkit layout")
    links = [(root / "lib64", Path("lib")), (lib / "libcudart.so", Path("libcudart.so.13"))]
    created = []
    for path, target in links:
        if path.is_symlink():
            if path.readlink() != target:
                raise ValueError("existing linker alias has an unexpected target")
        elif path.exists():
            raise ValueError("existing path would be overwritten")
        else:
            path.symlink_to(target, target_is_directory=path.name == "lib64")
            created.append(path.relative_to(root).as_posix())
    print(json.dumps({"created_aliases": created,
                      "driver_link_directory_present": Path("/usr/lib/wsl/lib/libcuda.so").is_file()}))
if __name__ == "__main__":
    main()

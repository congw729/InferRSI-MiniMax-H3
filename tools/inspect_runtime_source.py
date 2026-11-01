"""Read selected source declarations without importing runtime packages."""
import argparse, ast, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--names", default="")
    args = p.parse_args()
    out = {}
    names = set(args.names.split(","))
    for path in args.paths:
        source = path.read_text()
        tree = ast.parse(source)
        selected = [ast.get_source_segment(source, n) for n in ast.walk(tree)
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        out[path.name] = selected if args.names else source
    print(json.dumps(out))
if __name__ == "__main__":
    main()

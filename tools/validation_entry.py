"""Keep path-bearing Python exceptions out of executor stdout/stderr."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    prototype = Path(sys.argv[sys.argv.index("--prototype") + 1])
    project = Path(__file__).resolve().parents[1]
    def redact(text):
        for source, label in ((str(prototype), "<PROTOTYPE_ROOT>"),
                              (str(project), "<H3_PROJECT_ROOT>"), (str(project.parent), "<WORKSPACE_ROOT>"),
                              (str(Path(sys.executable).parents[2]), "<ENV_ROOT>"),
                              (str(Path.home()), "<USER_HOME>")):
            text = text.replace(source, label).replace(source.replace("\\", "/"), label)
        return text
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            runpy.run_path(str(project / "tools/validate_tensors.py"), run_name="__main__")
        except SystemExit as exc:
            code = exc.code or 0
        except Exception as exc:
            code = 1
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"campaign_verdict": "INCONCLUSIVE",
                              "execution_error": {"type": type(exc).__name__,
                                                  "message": redact(str(exc))}}, indent=2) + "\n",
                              encoding="utf-8")
    print(redact(stdout.getvalue()), end="")
    print(redact(stderr.getvalue()), end="", file=sys.stderr)
    return code

if __name__ == "__main__":
    raise SystemExit(main())

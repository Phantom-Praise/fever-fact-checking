"""Small entry point for inspecting the project and its frozen working prompt.

No inference, training or model download is triggered by the default command.
The results viewer serves saved artifacts. Prompt selection is a research
configuration, not an automatic replacement of every historical verifier.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "docs/working_parts_prompt.json"
REPORTS = {
    "parts": "build_claim_parts_report",
    "parts-v2": "build_claim_parts_v2_report",
    "rag": "build_ephemeral_rag_report",
    "references": "build_parts_reference_review",
}


def load_working_prompt(selection_path=SELECTION):
    """Read and verify the selected prompt without calling a model."""
    selected = json.loads(Path(selection_path).read_text())
    prompt_file = ROOT / selected["prompt_file"]
    content = prompt_file.read_bytes()
    if hashlib.sha256(content).hexdigest() != selected["prompt_sha256"]:
        raise ValueError("Working prompt differs from the frozen selection")
    implementation = ROOT / selected["implementation"]
    if hashlib.sha256(implementation.read_bytes()).hexdigest() != selected["implementation_sha256"]:
        raise ValueError("Working prompt implementation differs from the frozen selection")
    return selected, content.decode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("prompt", help="Print the selected 7/10 development prompt; no inference")
    viewer = commands.add_parser("serve", help="Serve existing results and traces; no inference")
    viewer.add_argument("--port", type=int, default=8765)
    report = commands.add_parser("report", help="Rebuild a derived HTML report from saved results")
    report.add_argument("name", choices=sorted(REPORTS))
    args = parser.parse_args(argv)
    if args.command == "prompt":
        selected, text = load_working_prompt()
        print("Selected condition:", selected["condition"])
        print("Saved prompt:", selected["prompt_file"])
        print("Development result: 7/10 derived labels; 5/10 model overall labels.")
        print("Evidence/model/settings must remain paired; this is not held-out accuracy.\n")
        print(text, end="" if text.endswith("\n") else "\n")
    elif args.command == "serve":
        module = importlib.import_module("evidence_explorer")
        previous = sys.argv
        try:
            sys.argv = [module.__file__, "--port", str(args.port)]
            module.main()
        finally:
            sys.argv = previous
    elif args.command == "report":
        importlib.import_module("reporting." + REPORTS[args.name]).main()
    else:
        parser.print_help()
        print("\nCurrent source map: src/README.md")
        print("Report tools: src/reporting/; historical prototypes: src/legacy/.")
        print("Frozen experiment files remain in place to preserve source-hash checks.")


if __name__ == "__main__":
    main()

"""Build a CPU-only Colab notebook with a distinct GGUF experiment identity."""
import ast
import base64
import hashlib
import json
import runpy
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "colab/build_independent_notebook.py"))
output = ROOT / "output/FEVER_Project"
notebook = json.loads((output / "FEVER_Independent_Systems_19998.ipynb").read_text())
adapter = (ROOT / "colab/independent_cpu.py").read_bytes()
for cell in notebook["cells"]:
    text = "".join(cell["source"])
    if cell["cell_type"] == "markdown":
        text = text.replace("Select T4 GPU", "Select Runtime > Change runtime type > Hardware accelerator > None (CPU)")
        text = text.replace("We load one classifier at a time in 4-bit NF4.",
            "LLMs run through llama.cpp with four-bit GGUF weights and zero GPU layers.")
        if text.startswith("# FEVER"):
            text += """

## CPU experiment
This CPU notebook uses a separate output folder. The earlier GPU/NF4 predictions
remain saved in their original folder. GGUF quantization and generation differ,
so CPU predictions are reported as their own full 19,998-claim condition.
No model is retrained. Search prompts, selection rules and final label rules
are retained. Quantization filenames and revisions are recorded.

Qwen requires approximately 4.36 GiB of CPU weights. Other model sizes are
printed before downloading. Files are cached in Drive and copied into the
runtime for inference. CPU execution may be substantially slower than the
previous T4 run. The live monitor measures actual per-claim throughput and ETA.
"""
    elif text.startswith("import sys, subprocess"):
        text = textwrap.dedent("""
        import os, sys
        from pathlib import Path
        import torch
        print("Python:", sys.version.split()[0])
        print("CPU cores:", os.cpu_count())
        if torch.cuda.is_available():
            raise RuntimeError("Select Runtime > Change runtime type > Hardware accelerator > None to release the GPU, then Run all.")
        print("CPU mode selected. No GPU compute units are needed by this notebook.")
        """).strip()
    elif text.startswith("import subprocess, sys, os"):
        text = textwrap.dedent("""
        import subprocess, sys, os
        os.environ["USE_TF"] = "0"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HOME"] = "/content/fever_hf_cache"
        os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=OFF -DGGML_METAL=OFF"
        os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = "2"
        packages = ["transformers==4.57.6", "scikit-learn==1.6.1",
                    "requests>=2.32,<3", "huggingface-hub<1",
                    "llama-cpp-python==0.3.36", "librosa>=0.11,<0.12"]
        print("Installing CPU libraries. llama.cpp may compile locally; installation output follows.", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        print("CPU libraries ready.")
        """).strip()
    elif "independent = importlib.reload(independent)" in text:
        text += "\n" + textwrap.dedent(
            "cpu_source = zlib.decompress(base64.b64decode(" +
            repr(base64.b64encode(zlib.compress(adapter)).decode()) + "))\n" +
            "assert hashlib.sha256(cpu_source).hexdigest() == " + repr(hashlib.sha256(adapter).hexdigest()) + "\n" +
            "(PROJECT / 'src/independent_cpu.py').write_bytes(cpu_source)\n" +
            "import independent_cpu as cpu\ncpu = importlib.reload(cpu)\n" +
            "print('CPU adapter verified. GGUF inference uses n_gpu_layers=0.')")
    elif text.startswith("EXPECTED_ROBERTA ="):
        text = text.replace("independent_systems_dev19998_v1", "independent_systems_cpu_dev19998_v1")
        text = text.replace(
            'fever.remote_metadata(independent.MODELS[family]) if family != "roberta" else None',
            'cpu.metadata(family) if family != "roberta" else None')
        text = text.replace(
            'family, metadata, CHECKPOINT, minutes=remaining)',
            'family, metadata, CHECKPOINT, minutes=remaining,\n'
            '        generator_class=cpu.CPUGenerator, roberta_class=cpu.CPURobertaSelector,\n'
            '        execution_environment=cpu.environment())')
    if cell["cell_type"] == "code":
        ast.parse(text)
    cell["source"] = text.splitlines(keepends=True)

notebook["metadata"].pop("accelerator", None)
notebook["metadata"]["colab"].pop("gpuType", None)
notebook["metadata"]["colab"]["name"] = "FEVER_Independent_Systems_CPU_19998.ipynb"
target = output / "FEVER_Independent_Systems_CPU_19998.ipynb"
target.write_text(json.dumps(notebook, indent=1))
print("Created CPU notebook:", target)

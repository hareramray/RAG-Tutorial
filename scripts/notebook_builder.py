"""Small authoring utility; notebooks remain editable, ordinary .ipynb files."""
import json
import textwrap
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "id": uuid.uuid4().hex[:8],
            "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "metadata": {}, "id": uuid.uuid4().hex[:8],
            "execution_count": None, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


BOOTSTRAP = '''
from pathlib import Path
import sys

# Works when Jupyter starts in the repository or in notebooks/.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "rag_workshop").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook from the LangchainPractical project folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_workshop import (
    Document, load_documents, chunk_documents, TfidfIndex, DenseIndex,
    bm25_search, reciprocal_rank_fusion, context_text, ask_claude,
    rag_answer, show_hits, settings, offline_mode,
)
import pandas as pd
from IPython.display import display, Markdown

print("Mode:", "OFFLINE: Claude calls skipped" if offline_mode() else "LIVE: Claude calls use API credits")
print("Claude model:", settings().model)
'''


def write_notebook(filename, cells):
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python (RAG Workshop)", "language": "python", "name": "rag-workshop"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "cells": cells,
    }
    target = ROOT / "notebooks" / filename
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target.name)

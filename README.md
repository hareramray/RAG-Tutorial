# RAG: text, multimodal, graph, and structured data with Claude

A practical course of **18 Jupyter notebooks** for teaching by running code, from beginner foundations to advanced retrieval and expert practice. Each notebook includes learning goals, numbered steps, visible intermediate results, demo variations, and exercises with sample solutions. Every notebook starts from a fresh kernel; there is no hidden dependency on a previous notebook's variables.

Claude reads your existing **`CLAUDE_KEY` from `.env`**. The project does not modify that file. Generation uses Anthropic, with a LangChain integration in lesson 05. Text embeddings, reranking, CLIP image embeddings, CLAP audio embeddings, and Whisper speech recognition run locally on CPU; they need no additional paid API key. Anthropic does not provide an embedding model in its API. See the [Anthropic embedding documentation](https://platform.claude.com/docs/en/build-with-claude/embeddings).

## Start here

The `.venv` environment and **Python (RAG Workshop)** kernel have been created in this workspace. From PowerShell in this folder:

```powershell
.\.venv\Scripts\jupyter-lab.exe notebooks
```

Open **[00_setup_and_claude.ipynb](notebooks/00_setup_and_claude.ipynb)**, choose **Python (RAG Workshop)**, and run cells from top to bottom with **Shift+Enter**. In VS Code, select `.venv\Scripts\python.exe` through the notebook's kernel picker. For a new demo, use **Restart Kernel and Run All**.

To recreate the environment on Windows with Python 3.13:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This command applies the execution-policy setting to that process only. Setup uses the exact dependency versions in `requirements.lock.txt`. If installing on another supported Python/platform combination, create a virtual environment and install `requirements.txt` instead, then run `python -m ipykernel install --sys-prefix --name rag-workshop --display-name "Python (RAG Workshop)"`. The provided lock was resolved on Windows with Python 3.13.

## Learning path

| Notebook | Level / time | What you demonstrate |
|---|---|---|
| [00 · Setup and Claude](notebooks/00_setup_and_claude.ipynb) | Beginner · 20 min | Environment, safe key loading, Messages API, tokens and cache |
| [01 · RAG from scratch](notebooks/01_rag_from_scratch.ipynb) | Beginner · 30 min | Question → lexical retrieval → evidence → cited answer; missing evidence |
| [02 · Ingestion and chunking](notebooks/02_ingestion_and_chunking.ipynb) | Beginner · 35 min | TXT/Markdown/PDF loading, metadata, overlap and chunk-size tradeoffs |
| [03 · Embeddings and semantic search](notebooks/03_embeddings_and_semantic_search.ipynb) | Intermediate · 35 min | Real local embeddings, cosine similarity, lexical comparison and plots |
| [04 · Vector store and metadata](notebooks/04_vector_store_and_metadata.ipynb) | Intermediate · 35 min | Persistent Chroma, supplied vectors, filtering, upserts and deletion |
| [05 · LangChain RAG](notebooks/05_langchain_rag.ipynb) | Intermediate · 30 min | LCEL composition, prompt inspection and `ChatAnthropic` |
| [06 · Hybrid search and reranking](notebooks/06_hybrid_search_and_reranking.ipynb) | Advanced · 40 min | BM25, dense retrieval, reciprocal rank fusion and local cross-encoder |
| [07 · Advanced retrieval](notebooks/07_advanced_retrieval.ipynb) | Advanced · 40 min | Query rewriting, multi-query fusion, parent documents and HyDE |
| [08 · Conversational RAG](notebooks/08_conversational_rag.ipynb) | Advanced · 35 min | Follow-up questions, standalone rewrites and bounded conversation memory |
| [09 · Evaluation and experiments](notebooks/09_evaluation_and_experiments.ipynb) | Advanced · 45 min | Precision/recall/MRR, held-out questions, abstention and optional LLM judge |
| [10 · Agentic RAG](notebooks/10_agentic_rag.ipynb) | Expert practice · 45 min | Actual Claude tool calls, local search dispatcher and bounded agent loop |
| [11 · Production capstone](notebooks/11_production_capstone.ipynb) | Expert practice · 50 min | Authorization before retrieval, cache invalidation, citations, injection exercise and your documents |
| [12 · Multimodal RAG](notebooks/12_multimodal_rag.ipynb) | Intermediate → advanced · 45–60 min | Text/image/recording catalog, actual CLIP pixel retrieval, rank fusion and Claude vision with original assets |
| [13 · Image/document RAG](notebooks/13_image_document_rag.ipynb) | Intermediate → advanced · 45–60 min | Image-only PDF failure, page rendering, vision ingestion, page retrieval and selected image/PDF answers |
| [14 · Audio RAG](notebooks/14_audio_rag.ipynb) | Intermediate → advanced · 45–60 min | Timestamped speech retrieval, optional real ASR, playback, signal analysis and optional CLAP sound retrieval |
| [15 · Video RAG](notebooks/15_video_rag.ipynb) | Advanced · 50–65 min | Real MP4 decoding, audio extraction, frames, temporal alignment, optional ASR/CLIP and cited visual answers |
| [16 · GraphRAG](notebooks/16_graph_rag.ipynb) | Advanced · 45–60 min | Source-backed triples, directed graph traversal, multi-hop evidence, negation and community overviews |
| [17 · Structured-data RAG](notebooks/17_structured_data_rag.ipynb) | Advanced · 45–60 min | Claude query plans, read-only SQLite joins/aggregations and answers combining computed rows with policy text |

Times are teaching estimates excluding first-time downloads and exercises. The full course is approximately 12–14 hours; use several sessions or two workshop days, with a separate session for lessons 12–17. These are compact instructional implementations, not a deployed production service or a guarantee of expert mastery.

### What the additional RAG families mean here

The multimodal lesson distinguishes **searchable text descriptions** from **original visual evidence**. CLIP searches actual image pixels; Claude receives selected original images and text. Audio retrieval has separate transcript and raw-waveform paths. Video retrieval combines speech with decoded frames and timestamps; the answer call sends images and transcript text, not an MP4 or audio stream. These distinctions are visible in each notebook's switches and outputs.

GraphRAG is a small, source-backed knowledge-graph implementation with a community overview extension, not Microsoft's complete GraphRAG pipeline. Structured-data RAG uses a validated query-plan interface to reviewed SQL templates; it does not execute arbitrary model-written SQL. The course covers these major families without claiming every RAG variant, native video embeddings, motion recognition, or speaker diarization.

## Configuration and cost

Your `.env` already contains `CLAUDE_KEY`. Optional settings are documented in [.env.example](.env.example); **do not replace your existing key**.

```dotenv
CLAUDE_MODEL=claude-haiku-4-5-20251001
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_OFFLINE=0
RAG_CACHE=1
```

Your current key has passed live Claude checks through both the Anthropic SDK and LangChain; no workspace ID is needed for this key. Restart any running notebook kernel after replacing a key so it loads the new configuration.

If another key returns an error about workspace scoping, add `ANTHROPIC_WORKSPACE_ID=wrkspc_...` using the ID in **Claude Console → Settings → Workspaces**, then restart the kernel. The SDK, LangChain, and agent examples forward this optional header. Alternatively, use a workspace-scoped key in `CLAUDE_KEY`. See [Anthropic workspace scoping](https://platform.claude.com/docs/en/manage-claude/workspaces).

- **Live mode is the default.** Running Claude cells consumes your Anthropic API credits. The notebooks use short responses and bounded loops. Optional bulk judging and additional demos have explicit switches. Check actual rates in [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
- `CLAUDE_MODEL` lets you choose a model available to your account. Notebook 00 includes an optional model-listing step. See [Anthropic models](https://platform.claude.com/docs/en/about-claude/models/overview).
- The first run of each local model downloads public weights from Hugging Face; subsequent runs use its local cache. Downloads can be substantial, especially CLAP. Models run on CPU, and first runs can be slow. Local embedding/transcription inputs stay local; a Claude answer sends its selected text and image/PDF evidence to Anthropic.
- `ask_claude` stores completed responses in `.cache/claude/`, keyed by workspace, model, prompt, system instructions and token limit. A cache hit makes no API call. Reported input/output tokens on a cache hit describe the original response, not new usage. LangChain and the manual agent loop have their own direct API calls and do not use this helper cache. Notebooks 08 and 11 disable it in their own kernels to demonstrate in-memory state and accurate per-call accounting.
- Media responses use `.cache/claude_media/`, keyed by the model, workspace, question, instructions, token limit and encoded media content. The response cache stores returned text and usage, not the input image/PDF bytes. Recognized transcripts have a separate local cache under `.cache/transcripts/`.
- `.env`, model/demo artifacts, custom documents and response caches are excluded from Git by `.gitignore`. Notebook outputs can still contain your questions and retrieved text; clear outputs before sharing a notebook that used private data.

The new notebooks keep costly or large demonstrations explicit:

| Notebook | Default live behavior | Optional changes |
|---|---|---|
| 12 | Real local CLIP retrieval and one Claude vision answer | Set `RUN_CLIP=False` to use the labelled caption-only search baseline |
| 13 | One Claude vision ingestion call and one selected-page answer | `USE_AUTHORED_DESCRIPTIONS=True` skips model ingestion; `ANSWER_INPUT="pdf"` switches the answer input to a retrieved-page PDF |
| 14 | Authored transcript retrieval and one Claude answer | Enable `RUN_ASR`, `RUN_DENSE`, or `RUN_CLAP` for the separate local-model demonstrations; all default to `False` |
| 15 | Authored transcript retrieval, decoded frame evidence and one Claude vision answer | `RUN_ASR` and `RUN_CLIP` default to `False`; enable them for automatic speech and visual retrieval |
| 16 | Curated graph traversal and one Claude answer | Extraction and community-summary switches default to `False`; each enabled generation adds a call |
| 17 | One Claude query-plan call and one grounded answer | Inspect and modify the supported query templates locally |

That is at most eight default Claude requests across lessons 12–17 on a successful first Run All, before retries or optional reruns. Cache hits skip calls. Local model switches do not add Anthropic charges; they use your computer and network for downloads.

For **free rehearsal without API calls or model downloads**:

```powershell
$env:RAG_OFFLINE = '1'
.\.venv\Scripts\jupyter-lab.exe notebooks
```

Offline mode uses a clearly labelled **TF-IDF lexical substitute** in the original dense index and skips Claude and cross-encoder generation. **CLIP, CLAP and ASR are skipped without fabricated embeddings or transcripts.** Media lessons label their supplied descriptions, narration scripts and manifests as authored fixtures. PDF rendering, media decoding/playback, lexical retrieval, graph traversal and SQLite calculations still run locally. Prepared query rewrites and SQL plans are labelled examples. Offline mode rehearses code flow; it does not validate learned models or answer quality. Change to `$env:RAG_OFFLINE = '0'` and restart Jupyter for a live session. Existing process variables take precedence over `.env`.

## Included data and your own files

[data/knowledge_base.json](data/knowledge_base.json) contains 12 invented Northstar Analytics documents: employee benefits, remote work, expenses, security, product plans, refunds, API limits, incidents, retention, onboarding, release notes, and support targets. All organizations and policies in these examples are fictional.

[data/eval_questions.json](data/eval_questions.json) contains 15 labelled questions with development/test splits, reference answers, source IDs, and unknown-answer cases. This tiny benchmark teaches evaluation; it is not evidence of production accuracy. Source relevance judgements are a small curated set and can be extended for alternative supporting passages.

[data/media/](data/media/) adds deterministic charts and diagrams, a synthetic scanned policy, an image-only three-page PDF, a **38.6-second narrated WAV and MP4 with an audio track**, and three synthetic sound examples. The narration uses Windows text-to-speech. Supplied transcript timestamps are measured boundaries of the authored utterances, not ASR results. Scene metadata records authored visual intervals and separate narration times; it is not automatic scene detection. All media and facts are fictional; see [the media provenance notes](data/media/README.md).

[data/graph/gold_graph.json](data/graph/gold_graph.json) contains manually curated entities and relationship triples linked to exact source quotes. Notebook 16 keeps optional model-extracted candidates separate. Notebook 17 creates a small SQLite fixture under `artifacts/structured/`, with plans, accounts, seats and an order ledger.

To bring your own text documents, create `data/custom/` and add UTF-8 `.txt`, `.md`, or text-based `.pdf` files. Notebooks 02 and 11 show the text loader. **Notebook 13 handles image-only pages using rendering and Claude vision**; a separate OCR executable is not required for that lesson. Preserve originals and review extracted text before indexing. For your own audio/video, regenerate transcripts and time windows instead of reusing the sample's authored metadata. Set appropriate source-access labels before adapting the authorization demo.

## Demonstrating effectively

The [PowerPoint lesson index](presentations/README.md) provides a separate deck for every notebook. Each deck contains nine slides with concept explanations, worked examples, a live-demo walkthrough, an exercise, and presenter notes. Open the matching notebook alongside its slides when teaching.

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for a short presentation route, questions to ask the audience, expected outcomes, and troubleshooting. Show retrieved passages before displaying Claude's answer. Change one variable at a time, such as chunk size or `k`, and explain the measured result rather than assuming a more complex retriever always wins.

## Verification and maintenance

Verified in this workspace: **18/18 notebooks executed in fresh offline kernels and 57/57 tests passed**. Live checks cover Claude SDK/LangChain, agent tool calls, all six new notebooks, and validated SQL plans with computed answers. Audio/video notebooks also passed with every optional local model enabled: Whisper ASR, dense text retrieval, CLAP waveform retrieval, and CLIP image-pixel retrieval. Timestamp validation retains original ASR estimates and visibly flags corrections. These are bounded integration checks, not a comprehensive answer-quality evaluation.

```powershell
# Local behavior tests; no API calls or model downloads
.\.venv\Scripts\python.exe -m pytest -q

# Fresh-kernel execution of every notebook in offline mode
.\.venv\Scripts\python.exe scripts/verify_notebooks.py

# One short billed Claude request
.\.venv\Scripts\python.exe scripts/smoke_check.py --live

# Real local embedding and cross-encoder checks (downloads on first run)
.\.venv\Scripts\python.exe scripts/smoke_check.py --models

# Actual media models; local CPU, downloads on first run, no Claude request
.\.venv\Scripts\python.exe scripts/smoke_media.py --clip --asr --clap

# One short Claude vision request with the sample chart (cache may avoid a new call)
.\.venv\Scripts\python.exe scripts/smoke_media.py --vision
```

Executed verification copies and the report are written to `artifacts/offline_notebooks/`; source notebooks retain empty outputs so a presentation starts cleanly. The offline runner also accepts notebook prefixes, for example `scripts/verify_notebooks.py 04 10`. Optional `--langchain` on the smoke checker makes a separate short Claude call through LangChain.

Notebook authoring scripts are `scripts/build_beginner.py` (00–03), `scripts/build_intermediate.py` (04–07), `scripts/build_advanced.py` (08–11), `scripts/build_visual_rag.py` (12–13), `scripts/build_temporal_rag.py` (14–15), and `scripts/build_graph_sql_rag.py` (16–17). Running a generator overwrites edits made directly to its notebooks. `scripts/build_data.py` recreates the text corpus; `scripts/build_media.py` recreates the bundled media and requires Windows TTS and fonts. Media is already included, so ordinary notebook users do not need to run these build scripts. Shared utilities are in [core.py](rag_workshop/core.py), [media.py](rag_workshop/media.py), and [structured.py](rag_workshop/structured.py).

## Primary references

- [Anthropic Python SDK](https://platform.claude.com/docs/en/api/python) and [tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Sentence Transformers usage](https://sbert.net/docs/sentence_transformer/usage/usage.html) and [CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
- [LangChain ChatAnthropic](https://docs.langchain.com/oss/python/integrations/chat/anthropic)
- [Chroma: adding data](https://docs.trychroma.com/docs/collections/add-data)
- [Claude vision](https://platform.claude.com/docs/en/build-with-claude/vision) and [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support)
- [CLIP image/text features](https://huggingface.co/docs/transformers/model_doc/clip), [CLAP audio/text features](https://huggingface.co/docs/transformers/model_doc/clap), and [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

Implementation documentation was checked on October 11, 2026. Package versions are captured in the lock file; provider availability and prices can change.

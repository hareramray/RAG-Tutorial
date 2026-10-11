"""Small explicit integration checks: --live is one short billed Claude call.

--models downloads/loads both CPU models; --langchain makes one additional call.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--live", action="store_true")
parser.add_argument("--models", action="store_true")
parser.add_argument("--langchain", action="store_true")
args = parser.parse_args()
os.environ["RAG_OFFLINE"] = "0" if args.live or args.models or args.langchain else "1"

from rag_workshop import settings, load_documents, chunk_documents, DenseIndex, ask_claude, anthropic_headers

print("CLAUDE_KEY:", "configured" if settings().api_key else "missing")
print("Model:", settings().model)
print("Explicit workspace:", "configured" if settings().workspace_id else "not configured (needed for unscoped keys)")
if args.live:
    response = ask_claude("Reply with exactly: RAG workshop ready", max_tokens=32)
    print("Claude:", response["text"])
    print("Usage:", response["input_tokens"], "input tokens,", response["output_tokens"], "output tokens; cached:", response["cached"])
if args.models:
    index = DenseIndex(chunk_documents(load_documents()))
    print("Embeddings:", index.backend, "shape:", index.vectors.shape)
    hits = index.search("How much paid holiday do employees get?", k=2)
    print("Semantic retrieval sources:", [h["metadata"]["source_id"] for h in hits])
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    scores = reranker.predict([("paid holiday allowance", h["text"]) for h in hits])
    print("Reranker returned", len(scores), "scores.")
if args.langchain:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    chain = (ChatPromptTemplate.from_template("Reply with exactly: {message}")
             | ChatAnthropic(model=settings().model, api_key=settings().api_key,
                             default_headers=anthropic_headers(), max_tokens=32)
             | StrOutputParser())
    print("LangChain:", chain.invoke({"message": "LangChain ready"}))
if not any(vars(args).values()):
    index = DenseIndex(chunk_documents(load_documents()))
    print(index.backend)
    print("Corpus documents:", len(load_documents()), "; indexed chunks:", len(index.docs))
    print("Local setup ready. Use --live for one short Claude API check.")

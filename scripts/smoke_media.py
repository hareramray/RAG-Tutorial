"""Explicit integration checks for the new RAG lessons; no hidden downloads.

--clip, --asr and --clap download/run local models; --vision is one billed Claude request.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
for flag in ["clip", "asr", "clap", "vision"]:
    parser.add_argument("--" + flag, action="store_true")
args = parser.parse_args()
os.environ["RAG_OFFLINE"] = "0"

from rag_workshop.media import (MEDIA_ROOT, load_media_catalog, clip_rank_images,
                               clap_rank_audio, transcribe_audio, ask_claude_media)

if args.clip or args.clap:
    import torch
    torch.set_num_threads(4)
if args.clip:
    images = [r for r in load_media_catalog() if r["modality"] == "image"]
    hits = clip_rank_images("A bar chart comparing data retention for two subscription plans", images, k=3)
    assert len(hits) == 3 and all(-1.01 <= h["score"] <= 1.01 for h in hits)
    print("CLIP actual pixel ranks:", [(h["id"], round(h["score"], 3)) for h in hits], flush=True)
if args.asr:
    segments = transcribe_audio(MEDIA_ROOT / "audio/northstar_briefing.wav")
    assert segments and all(s["start"] < s["end"] for s in segments)
    print("Actual ASR segments:", len(segments), flush=True)
    print("ASR text:", " ".join(s["text"] for s in segments), flush=True)
if args.clap:
    records = [{"id": name, "path": f"audio/{name}.wav"}
               for name in ["alarm_beeps", "ascending_tone", "noise_bursts"]]
    hits = clap_rank_audio("The sound of repeated electronic beeps", records, k=3)
    assert len(hits) == 3 and all(-1.01 <= h["score"] <= 1.01 for h in hits)
    print("CLAP actual waveform ranks:", [(h["id"], round(h["score"], 3)) for h in hits], flush=True)
if args.vision:
    result = ask_claude_media("Read Image 1. What is the retention of each plan, and what color is the taller bar? Cite [chart].",
                            image_paths=[MEDIA_ROOT / "images/plan_retention_chart.png"], max_tokens=180)
    assert not result["offline"] and "90" in result["text"] and "365" in result["text"]
    print("Claude vision:", result["text"], flush=True)
    print("Usage:", result["input_tokens"], "input,", result["output_tokens"], "output; cached:", result["cached"], flush=True)
if not any(vars(args).values()):
    print("Choose --clip, --asr, --clap, or --vision. Only --vision calls Anthropic.")

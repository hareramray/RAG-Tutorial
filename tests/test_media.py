"""Actual local fixture/I/O checks, no API calls or neural model downloads."""
import json
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader

from rag_workshop import media


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("RAG_OFFLINE", "1")


def test_catalog_paths_and_narration_timestamps():
    records = media.load_media_catalog()
    assert {r["modality"] for r in records} == {"text", "image", "audio", "video"}
    assert all((media.MEDIA_ROOT / r["path"]).is_file() for r in records)
    samples, rate = media.read_audio("audio/northstar_briefing.wav")
    segments = media.load_segments()
    assert np.isfinite(samples).all() and samples.ndim == 1
    assert all(0 <= s["start"] < s["end"] <= len(samples) / rate + 0.002 for s in segments)
    assert all(a["end"] <= b["start"] for a, b in zip(segments, segments[1:]))
    assert all("authored" in s["provenance"] for s in segments)


def test_half_open_overlap_and_exact_excerpt_length():
    segments = [{"start": 0, "end": 2}, {"start": 2, "end": 4}, {"start": 4, "end": 6}]
    assert media.align_segments(segments, 2, 4) == [segments[1]]
    samples, rate = media.audio_excerpt("audio/northstar_briefing.wav", 1.25, 2.75)
    assert len(samples) == round(rate * 1.5)
    with pytest.raises(ValueError):
        media.audio_excerpt("audio/northstar_briefing.wav", -1, 2)


def test_resampling_preserves_duration_and_finite_samples():
    source, source_rate = media.read_audio("audio/alarm_beeps.wav")
    resampled, rate = media.read_audio("audio/alarm_beeps.wav", target_sr=16000)
    assert rate == 16000 and np.isfinite(resampled).all()
    assert len(source) / source_rate == pytest.approx(len(resampled) / rate)


def test_timestamp_rounding_and_validation():
    assert media.format_timestamp(61.234) == "01:01.234"
    assert media.format_timestamp(59.9999) == "01:00.000"
    with pytest.raises(ValueError):
        media.format_timestamp(float("nan"))


def test_scanned_pdf_has_pixels_but_no_text_layer(tmp_path):
    pdf = media.MEDIA_ROOT / "documents/northstar_visual_guide.pdf"
    assert len(PdfReader(pdf).pages) == 3
    assert all(not (page.extract_text() or "").strip() for page in PdfReader(pdf).pages)
    paths = media.render_pdf_pages(pdf, tmp_path)
    assert len(paths) == 3 and all(p.stat().st_size > 1000 for p in paths)
    block = media.image_block(paths[0])
    assert block["type"] == "image" and block["source"]["media_type"] == "image/png"


def test_real_video_decodes_timestamped_frames_and_audio(tmp_path):
    video = media.MEDIA_ROOT / "video/northstar_demo.mp4"
    frames = media.extract_video_frames(video, tmp_path / "frames", every_seconds=5, max_frames=3)
    assert [r["timestamp"] for r in frames] == [0, 5, 10]
    assert all(Path(r["path"]).exists() for r in frames)
    audio = media.extract_video_audio(video, tmp_path / "track.wav")
    samples, rate = media.read_audio(audio)
    original, original_rate = media.read_audio("audio/northstar_briefing.wav")
    assert abs(len(samples) / rate - len(original) / original_rate) < 0.2


def test_visual_scene_windows_cover_silences_and_match_encoded_frames():
    scenes = json.loads((media.MEDIA_ROOT / "video/scenes.json").read_text())
    assert scenes[0]["start"] == 0
    assert all(a["end"] == b["start"] for a, b in zip(scenes, scenes[1:]))
    assert all(s["start"] * 10 == pytest.approx(round(s["start"] * 10)) for s in scenes)
    assert all(s["end"] >= s["narration_end"] for s in scenes)
    assert all(0 <= s["start"] - s["narration_start"] < 0.10001 for s in scenes)


def test_offline_models_do_not_download_or_invent_results(monkeypatch):
    monkeypatch.setattr(media, "_clip", lambda: pytest.fail("No CLIP load in offline mode"))
    monkeypatch.setattr(media, "_clap", lambda: pytest.fail("No CLAP load in offline mode"))
    with pytest.warns(UserWarning, match="OFFLINE"):
        assert media.clip_rank_images("chart", [{"id": "x", "path": "missing.png"}]) == []
    with pytest.warns(UserWarning, match="OFFLINE"):
        assert media.clap_rank_audio("beeps", [{"id": "x", "path": "missing.wav"}]) == []
    with pytest.warns(UserWarning, match="OFFLINE"):
        assert media.transcribe_audio("missing.wav") == []
    result = media.ask_claude_media("What is shown?", image_paths=["missing.png"])
    assert result["offline"] and "skipped" in result["text"] and result["output_tokens"] == 0


def test_asr_out_of_bounds_timing_is_explicit_and_original_is_preserved():
    raw = [{"id": "ok", "start": 1, "end": 2, "source": "old.wav", "text": "valid"},
           {"id": "overrun", "start": 36, "end": 51.92, "text": "check this"},
           {"id": "outside", "start": 41, "end": 48, "text": "discard"},
           {"id": "invalid", "start": 2, "end": float("nan"), "text": "discard"}]
    with pytest.warns(UserWarning, match="1 clipped, 2 discarded"):
        accepted = media._bounded_transcript(raw, 38.66, "new.wav")
    assert [s["id"] for s in accepted] == ["ok", "overrun"]
    assert all(s["source"] == "new.wav" for s in accepted)
    assert accepted[1]["end"] == 38.66 and accepted[1]["raw_end"] == 51.92
    assert accepted[1]["timing_adjusted"] is True
    assert accepted[0]["timing_adjusted"] is False
    with pytest.warns(UserWarning):
        repeat = media._bounded_transcript(accepted, 38.66, "another.wav")
    assert repeat[1]["raw_end"] == 51.92  # cache normalization must not erase original evidence

"""Create small, original fixtures; no external company media or API keys used.

Rebuilding spoken audio needs Windows SAPI (Microsoft David Desktop). All generated
assets are bundled, so notebook users on other platforms do not need to rebuild.
"""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "data/media"
for folder in ["images", "documents", "audio", "video", "notes"]:
    (MEDIA / folder).mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def slide(title, subtitle):
    image = Image.new("RGB", (1280, 720), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 12), fill="#0f766e")
    draw.text((64, 55), title, fill="#0f172a", font=font(42, True))
    draw.text((64, 122), subtitle, fill="#475569", font=font(23))
    draw.text((64, 672), "NORTHSTAR ANALYTICS  |  FICTIONAL WORKSHOP DATA", fill="#64748b", font=font(18))
    return image, draw


# The colors are deliberately NOT in the text caption: a later question needs pixels.
figure, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
figure.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")
bars = ax.bar(["Team", "Enterprise"], [90, 365], color=["#0d9488", "#7c3aed"], width=0.48)
ax.bar_label(bars, labels=["90 days", "365 days"], fontsize=24, padding=10)
ax.set_ylim(0, 440)
ax.set_ylabel("Event retention (days)", fontsize=20)
ax.set_title("Northstar Cloud | Event data retention", fontsize=27, pad=22)
ax.tick_params(labelsize=20)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.16)
ax.set_axisbelow(True)
figure.text(0.5, 0.02, "FICTIONAL WORKSHOP DATA | Upgrades do not restore already expired events", ha="center", fontsize=14)
figure.tight_layout(rect=(0.035, 0.055, 0.98, 0.97))
figure.savefig(MEDIA / "images/plan_retention_chart.png")
plt.close(figure)

image, draw = slide("Severity-one incident workflow", "Internal playbook: acknowledgement is not a resolution guarantee")
steps = [("DETECT", "Alert received"), ("ACKNOWLEDGE", "Within 15 minutes"),
         ("UPDATE", "Every 30 minutes"), ("REVIEW", "Within 3 working days")]
for i, (label, detail) in enumerate(steps):
    x = 48 + i * 312
    draw.rounded_rectangle((x, 255, x + 270, 455), radius=18, fill="#e0f2fe", outline="#0369a1", width=3)
    draw.text((x + 18, 305), label, font=font(25, True), fill="#0c4a6e")
    draw.text((x + 18, 368), detail, font=font(19), fill="#164e63")
    if i < 3:
        draw.line((x + 273, 355, x + 302, 355), fill="#334155", width=4)
        draw.polygon([(x + 302, 355), (x + 292, 348), (x + 292, 362)], fill="#334155")
image.save(MEDIA / "images/incident_flow.png")

image, draw = slide("REFUND POLICY - SCANNED COPY", "Synthetic raster document: there is no searchable PDF text layer")
lines = ["An annual subscription is eligible for review only if:",
         "1. The request is made within 14 days of purchase.",
         "2. The subscription has never been used.",
         "Both conditions are required.",
         "Send the invoice number and workspace ID to Billing.",
         "Cancelling renewal does not automatically trigger a refund."]
for i, line in enumerate(lines):
    draw.text((82, 220 + i * 58), line, font=font(28, i == 3), fill="#27272a")
image.save(MEDIA / "images/scanned_refund.png")

image, draw = slide("Team plan | Annual billing", "The monthly figure is an equivalent; invoices are annual")
for i, line in enumerate(["INR 2,400 / user / month", "Minimum 5 users", "Minimum annual subtotal = INR 144,000", "Taxes are additional"]):
    draw.text((86, 220 + i * 92), line, font=font(40 if i == 0 else 32, i == 0), fill="#0f172a")
image.save(MEDIA / "images/team_billing.png")

image, draw = slide("API limits and retries", "Per workspace - request counts, not model token limits")
for i, line in enumerate(["TEAM: 600 requests / minute", "ENTERPRISE: 3,000 requests / minute", "HTTP 429: respect Retry-After", "Use bounded exponential backoff with jitter"]):
    draw.text((82, 220 + i * 94), line, font=font(34, i < 2), fill="#0f172a")
image.save(MEDIA / "images/api_limits.png")

pdf_path = MEDIA / "documents/northstar_visual_guide.pdf"
pdf = canvas.Canvas(str(pdf_path), pagesize=(960, 540))
pdf.setTitle("Northstar visual guide - fictional RAG workshop")
for name in ["plan_retention_chart", "incident_flow", "scanned_refund"]:
    pdf.drawImage(ImageReader(str(MEDIA / f"images/{name}.png")), 0, 0, width=960, height=540)
    pdf.showPage()
pdf.save()

script = [
    {"id": "billing", "image_path": "images/team_billing.png", "text": "Welcome to the Northstar product briefing. The Team plan costs two thousand four hundred rupees per user per month, billed annually, with a minimum of five users."},
    {"id": "retention", "image_path": "images/plan_retention_chart.png", "text": "Team event data is retained for ninety days. Enterprise event data is retained for three hundred and sixty five days. Upgrading cannot restore data that has already expired."},
    {"id": "api", "image_path": "images/api_limits.png", "text": "The Team API limit is six hundred requests per minute. Enterprise allows three thousand requests per minute. After an HTTP four twenty nine response, respect the Retry After header."},
]
write_json(MEDIA / "audio/speech_script.json", script)
if sys.platform != "win32":
    raise RuntimeError("Rebuilding narration uses Windows SAPI. Use the bundled media on other platforms.")
subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/synthesize_media.ps1")], check=True)
segments, scenes, parts = [], [], []
offset = 0
rate = 16000
for i, row in enumerate(script):
    samples, actual_rate = sf.read(MEDIA / f"audio/speech_{row['id']}.wav", dtype="float32")
    assert actual_rate == rate
    end = offset + len(samples) / rate
    record = {"id": f"speech:{row['id']}", "start": round(offset, 5), "end": round(end, 5),
              "text": row["text"], "source": "northstar_briefing.wav",
              "provenance": "authored TTS script; exact segment boundaries"}
    segments.append(record)
    scenes.append({"id": f"scene:{row['id']}", "start": record["start"], "end": record["end"],
                   "image_path": row["image_path"], "text": row["text"]})
    parts.append(samples)
    if i < len(script) - 1:
        parts.append(np.zeros(round(rate * 0.45), dtype="float32"))
        end += 0.45
    offset = end
audio = np.concatenate(parts)
sf.write(MEDIA / "audio/northstar_briefing.wav", audio, rate, subtype="PCM_16")
write_json(MEDIA / "audio/segments.json", segments)
# Visual scene boundaries are quantized to encoded frame times. A slide remains
# visible during the silence before the next narration segment.
fps = 10
for i, scene in enumerate(scenes):
    scene["narration_start"] = scene["start"]
    scene["narration_end"] = scene["end"]
    scene["start"] = float(np.ceil(scene["start"] * fps) / fps)
for i, scene in enumerate(scenes):
    scene["end"] = scenes[i + 1]["start"] if i + 1 < len(scenes) else float(np.ceil(len(audio) / rate * fps) / fps)
write_json(MEDIA / "video/scenes.json", scenes)

# Three synthetic non-speech sounds for actual waveform-to-text CLAP retrieval.
sound_rate = 48000
t = np.arange(sound_rate * 4) / sound_rate
rng = np.random.default_rng(7)
envelope = np.minimum(1, t * 30) * np.minimum(1, (4 - t) * 30)
sounds = {
    "alarm_beeps": 0.3 * np.sin(2 * np.pi * 880 * t) * ((t % 0.7) < 0.35) * envelope,
    "ascending_tone": 0.3 * np.sin(2 * np.pi * (200 * t + 150 * t ** 2)) * envelope,
    "noise_bursts": 0.15 * rng.normal(size=len(t)) * ((t % 0.8) < 0.45) * envelope,
}
for name, samples in sounds.items():
    sf.write(MEDIA / f"audio/{name}.wav", samples, sound_rate, subtype="PCM_16")

# Encode actual image frames and multiplex the narration into a playable MP4.
video_path = MEDIA / "video/northstar_demo.mp4"
command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(fps), "-i", "pipe:0",
           "-i", str(MEDIA / "audio/northstar_briefing.wav"), "-c:v", "libx264", "-preset", "fast",
           "-crf", "25", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
           "-shortest", "-movflags", "+faststart", str(video_path)]
with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
    frame_bytes = [(MEDIA / s["image_path"]) for s in scenes]
    frames = []
    for path in frame_bytes:
        with Image.open(path) as im:
            frames.append(im.convert("RGB").tobytes())
    for frame_index in range(int(np.ceil(len(audio) / rate * fps))):
        timestamp = frame_index / fps
        scene = max(i for i, s in enumerate(scenes) if s["start"] <= timestamp)
        process.stdin.write(frames[scene])
    process.stdin.close()
    error = process.stderr.read().decode(errors="replace")
    if process.wait() != 0:
        raise RuntimeError(error)

policy = "Team costs INR 2400 per user per month, billed annually, with a minimum of 5 users. Taxes are additional. Enterprise pricing is custom. Upgrading cannot restore expired events."
(MEDIA / "notes/plan_policy.md").write_text(policy + "\n", encoding="utf-8")
catalog = [
    {"id": "plan_retention_chart", "modality": "image", "path": "images/plan_retention_chart.png", "text": "Northstar plan event retention chart: Team 90 days; Enterprise 365 days.", "metadata": {"source": "northstar_visual_guide.pdf", "page": 1}},
    {"id": "incident_flow", "modality": "image", "path": "images/incident_flow.png", "text": "Severity-one incident flow: detect, acknowledge within 15 minutes, update every 30 minutes, review within 3 working days.", "metadata": {"source": "northstar_visual_guide.pdf", "page": 2}},
    {"id": "scanned_refund", "modality": "image", "path": "images/scanned_refund.png", "text": "Synthetic scanned refund policy: request within 14 days and unused annual subscription; both conditions required.", "metadata": {"source": "northstar_visual_guide.pdf", "page": 3}},
    {"id": "plan_policy", "modality": "text", "path": "notes/plan_policy.md", "text": policy, "metadata": {"source": "plan_policy.md"}},
    {"id": "product_briefing_audio", "modality": "audio", "path": "audio/northstar_briefing.wav", "text": " ".join(s["text"] for s in script), "metadata": {"source": "northstar_briefing.wav", "provenance": "authored narration script, not ASR"}},
    {"id": "product_demo_video", "modality": "video", "path": "video/northstar_demo.mp4", "text": "Narrated Northstar slides about Team annual billing, plan retention, API rate limits and retries.", "metadata": {"source": "northstar_demo.mp4", "provenance": "authored scene description"}},
]
write_json(MEDIA / "catalog.json", catalog)
(MEDIA / "README.md").write_text("# Original workshop media\n\nAll company facts and media are fictional teaching fixtures, generated locally by scripts/build_media.py. PNG charts and diagrams use deterministic code. The PDF has three image-only pages. Speech uses Windows Microsoft David Desktop TTS; segments.json contains the authored script and measured clip boundaries, NOT ASR output. The MP4 contains changing slides and that same narration. Three sound WAVs are synthetic sine/noise signals, not field recordings. No human speakers, external recordings, or API secrets are included.\n", encoding="utf-8")
print(f"Built 3-page PDF, 5 images, {len(audio)/rate:.2f}s narrated audio/video, and 3 acoustic clips.")

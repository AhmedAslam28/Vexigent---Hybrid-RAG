"""
media_utils.py — Audio, image and video processing utilities.

Exports
-------
image_captioning_service  Singleton ImageCaptioningService
process_image()           Describe an image via the best available vision provider.
process_audio()           Transcribe audio → List[Document] with [MM:SS] timestamps.
process_video()           Extract audio track + sampled frame descriptions.
stitch_frame_narrative()  Synthesize per-frame GPT-4o captions into a scene narrative.
detect_content_type()     Classify a file path as "image" | "audio" | "video" | "text".
"""

import base64
import logging
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import cv2
import librosa
import numpy as np
import openai
import whisper
from pydub import AudioSegment
from PIL import Image

import anthropic
import google.generativeai as genai
import requests as http_requests
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_content_type(file_path: str) -> str:
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        ext = Path(file_path).suffix.lower()
        if mime_type and mime_type.startswith("image/"): return "image"
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg"}: return "image"
        if mime_type and mime_type.startswith("audio/"): return "audio"
        if ext in {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}: return "audio"
        if mime_type and mime_type.startswith("video/"): return "video"
        if ext in {".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"}: return "video"
        if ext in {".pdf", ".txt", ".docx", ".doc", ".rtf", ".md", ".csv", ".json"}: return "text"
        return "text"
    except Exception:
        return "text"


# ─────────────────────────────────────────────────────────────────────────────
# VISION CAPTIONING SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class ImageCaptioningService:
    VISION_PROMPT = """You are an expert image analyst. Analyze this image and return a structured description.

1. IMAGE TYPE: Classify into one of these categories:
   - Nature/Landscape | Street/Urban | Food/Drink | People/Portrait | Event/Crowd
   - Mobile Screenshot | Desktop Screenshot | App UI | Error Screen | Chat/Messaging
   - Job Posting | Business Flyer | Advertisement | Infographic | Educational Poster
   - Document/Form | Receipt/Invoice | ID/Card | Handwritten Note
   - Product/Object | Vehicle | Architecture/Building | Art/Illustration | Meme/Graphic

2. FULL TEXT EXTRACTION: Extract every single word visible in the image exactly as written.

3. SCENE DESCRIPTION: Write a detailed paragraph describing what is shown.

4. CONTEXT & PURPOSE: What is the intent of this image?

5. SEARCHABLE KEYWORDS: List 20+ keywords."""

    def __init__(self):
        self.initialized = True

    def initialize_blip(self):
        logger.info("ℹ️ Multi-provider Vision mode active — BLIP initialization skipped")
        return True

    # ── helpers ───────────────────────────────────────────────────────────────

    def _read_b64(self, image_path: str):
        """Return (base64_str, mime_type) for the image."""
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return b64, mime_type

    # ── per-provider caption methods ──────────────────────────────────────────

    def _caption_openai(self, image_path: str, api_key: str) -> str:
        b64, mime_type = self._read_b64(image_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [
                {"type": "text",      "text": self.VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ]}],
            "max_tokens": 800,
        }
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=payload, timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"OpenAI Vision error {resp.status_code}: {resp.text[:200]}")

    def _caption_anthropic(self, image_path: str, api_key: str) -> str:
        b64, mime_type = self._read_b64(image_path)
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": mime_type,
                                             "data": b64}},
                {"type": "text",  "text": self.VISION_PROMPT},
            ]}],
        )
        return resp.content[0].text

    def _caption_gemini(self, image_path: str, api_key: str) -> str:
        from PIL import Image as PILImage
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-1.5-flash")
        img    = PILImage.open(image_path)
        result = model.generate_content([self.VISION_PROMPT, img])
        return result.text

    # ── public entry point ────────────────────────────────────────────────────

    def generate_comprehensive_description(
        self,
        image_path: str,
        provider: str = None,
        api_key: str = None,
    ) -> str:
        """
        Describe the image using the best available provider.

        Priority:
          1. Explicit provider + api_key (user-supplied at upload time)
          2. Auto-detect: OpenAI env key → Anthropic env key → Gemini env key
          3. PIL fallback (dimensions only)
        """
        candidates: list = []

        p = (provider or "").lower()
        if p == "openai" and api_key:
            candidates.append(("openai", api_key))
        elif p in ("anthropic", "claude") and api_key:
            candidates.append(("anthropic", api_key))
        elif p in ("gemini", "google") and api_key:
            candidates.append(("gemini", api_key))

        if OPENAI_API_KEY:
            candidates.append(("openai",    OPENAI_API_KEY))
        if ANTHROPIC_API_KEY:
            candidates.append(("anthropic", ANTHROPIC_API_KEY))
        if GOOGLE_API_KEY:
            candidates.append(("gemini",    GOOGLE_API_KEY))

        seen: set = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        last_error = None
        for prov, key in unique_candidates:
            try:
                logger.info(f"🖼️ Vision: trying {prov} for {Path(image_path).name}")
                if prov == "openai":
                    result = self._caption_openai(image_path, key)
                elif prov == "anthropic":
                    result = self._caption_anthropic(image_path, key)
                elif prov == "gemini":
                    result = self._caption_gemini(image_path, key)
                else:
                    continue
                logger.info(f"✅ Vision: {prov} succeeded for {Path(image_path).name}")
                return result
            except Exception as e:
                logger.warning(f"⚠️ Vision {prov} failed for {Path(image_path).name}: {e}")
                last_error = e

        logger.error(f"❌ All vision providers failed for {image_path}: {last_error}")
        try:
            from PIL import Image as PILImage
            with PILImage.open(image_path) as img:
                w, h = img.size
                return (
                    f"Image file: {Path(image_path).name}, "
                    f"Size: {w}x{h}, Mode: {img.mode}, Format: {img.format or 'Unknown'}"
                )
        except Exception as e2:
            return f"Image file: {Path(image_path).name} (processing error: {str(e2)})"


image_captioning_service = ImageCaptioningService()


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_image(file_path: str, provider: str = None, api_key: str = None) -> str:
    """Describe an image via the best available vision provider."""
    try:
        logger.info(f"🖼️ Processing image: {Path(file_path).name} (provider={provider or 'auto'})")
        description = image_captioning_service.generate_comprehensive_description(
            file_path, provider=provider, api_key=api_key
        )
        logger.info(f"✅ Image processing completed for {Path(file_path).name}")
        return description
    except Exception as e:
        logger.error(f"❌ Image processing failed for {file_path}: {str(e)}")
        try:
            from PIL import Image as PILImage
            with PILImage.open(file_path) as img:
                w, h = img.size
                return (
                    f"Image file: {Path(file_path).name}, "
                    f"Size: {w}x{h}, Mode: {img.mode}, Format: {img.format or 'Unknown'}"
                )
        except Exception as e2:
            return f"Image file: {Path(file_path).name} (processing error: {str(e2)})"


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER MODEL — loaded once, reused for every audio file
# ─────────────────────────────────────────────────────────────────────────────

logger.info("Loading Whisper base model...")
try:
    whisper_model = whisper.load_model("base")
    logger.info("✅ Whisper base model loaded successfully")
except Exception as _we:
    whisper_model = None
    logger.error(f"❌ Failed to load Whisper model: {str(_we)}")


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_audio(file_path: str) -> List[Document]:
    """
    Transcribe audio → List[Document] with [MM:SS] timestamps.

    Returns a header document (full transcript + acoustics) followed by
    one Document per merged Whisper segment so timestamp queries work.
    """
    try:
        _model = whisper_model
        if _model is None:
            try:
                _model = whisper.load_model("base")
            except Exception as e:
                return [Document(
                    page_content=f"Audio file: {Path(file_path).name} (Whisper model unavailable: {str(e)})",
                    metadata={"source": file_path, "filename": Path(file_path).name,
                              "type": "audio_error", "content_type": "audio", "origin": "audio_upload"},
                )]

        try:
            audio_seg    = AudioSegment.from_file(file_path)
            duration_sec = len(audio_seg) / 1000.0
        except Exception:
            duration_sec = 0.0

        logger.info(f"🎵 Transcribing: {Path(file_path).name} ({duration_sec:.1f}s)")

        result = _model.transcribe(
            file_path, language="en", fp16=False, verbose=False, word_timestamps=True,
        )

        full_transcript = result.get("text", "").strip()
        segments        = result.get("segments", [])

        if not segments:
            logger.warning(f"⚠️ No speech detected in {Path(file_path).name}")
            return [Document(
                page_content=(
                    f"Audio file: {Path(file_path).name}\n"
                    f"Duration: {duration_sec:.1f} seconds\n"
                    f"Transcript: (no speech detected)"
                ),
                metadata={"source": file_path, "filename": Path(file_path).name,
                          "type": "audio_header", "content_type": "audio",
                          "origin": "audio_upload", "duration_sec": duration_sec},
            )]

        logger.info(f"✅ Whisper returned {len(segments)} segments for {Path(file_path).name}")

        def fmt_ts(sec: float) -> str:
            m, s = divmod(int(sec), 60)
            return f"{m:02d}:{s:02d}"

        # Pause detection
        pauses = []
        for i in range(1, len(segments)):
            seg_start = float(segments[i]["start"])
            seg_end   = float(segments[i - 1]["end"])
            gap = seg_start - seg_end
            if gap > 0.5:
                pauses.append({
                    "at_seconds":       round(seg_end, 2),
                    "at_timestamp":     fmt_ts(seg_end),
                    "duration_seconds": round(gap, 2),
                    "after_segment":    i - 1,
                    "after_text":       segments[i - 1]["text"].strip(),
                    "before_text":      segments[i]["text"].strip(),
                })

        # Acoustic features
        acoustic_info = ""
        try:
            y, sr_rate = librosa.load(file_path, sr=None)
            tempo, _   = librosa.beat.beat_track(y=y, sr=sr_rate)
            tempo_val  = float(np.asarray(tempo).flat[0])
            energy     = float(np.mean(librosa.feature.rms(y=y)))
            speech_rate = len(segments) / duration_sec if duration_sec > 0 else 0
            acoustic_info = (
                f"\nAcoustic Features:"
                f"\n- Estimated tempo: {tempo_val:.1f} BPM"
                f"\n- Average energy level: {energy:.4f}"
                f"\n- Speech segment rate: {speech_rate:.2f} segments/sec"
            )
        except Exception as ae:
            logger.warning(f"⚠️ Acoustic analysis failed: {ae}")
            acoustic_info = "\nAcoustic Features: (unavailable)"

        pause_lines = []
        for i, p in enumerate(pauses):
            pause_lines.append(
                f"  Pause {i+1}: {p['duration_seconds']}s at {p['at_timestamp']} ({p['at_seconds']}s)"
                f"\n    Before pause: \"{p['after_text']}\""
                f"\n    After pause:  \"{p['before_text']}\""
            )
        pause_summary = (
            f"\nPauses Detected ({len(pauses)} total):\n" + "\n".join(pause_lines)
            if pauses else "\nPauses Detected: none above 500ms threshold"
        )

        first_seg = segments[0]
        last_seg  = segments[-1]
        header_text = (
            f"Audio file: {Path(file_path).name}\n"
            f"Duration: {duration_sec:.1f} seconds\n"
            f"Total segments: {len(segments)}\n"
            f"First sentence [{fmt_ts(first_seg['start'])}]: {first_seg['text'].strip()}\n"
            f"Last sentence [{fmt_ts(last_seg['start'])}]: {last_seg['text'].strip()}\n"
            f"Full Transcript:\n{full_transcript}"
            f"{pause_summary}"
            f"{acoustic_info}"
        )

        docs: List[Document] = [Document(
            page_content=header_text,
            metadata={
                "source":        file_path,
                "filename":      Path(file_path).name,
                "type":          "audio_header",
                "content_type":  "audio",
                "origin":        "audio_upload",
                "duration_sec":  duration_sec,
                "segment_count": len(segments),
                "pause_count":   len(pauses),
            },
        )]

        # Merge short segments into ~10-word groups
        merged_groups: List[list] = []
        buf: list = []
        for seg in segments:
            buf.append(seg)
            word_count = sum(len(s["text"].split()) for s in buf)
            if word_count >= 10:
                merged_groups.append(buf)
                buf = []
        if buf:
            merged_groups.append(buf)

        for grp_idx, grp in enumerate(merged_groups):
            start_sec  = float(grp[0]["start"])
            end_sec    = float(grp[-1]["end"])
            text       = " ".join(s["text"].strip() for s in grp)
            chunk_text = f"[{fmt_ts(start_sec)}] {text}"

            pause_flags = [
                p for p in pauses
                if p["at_seconds"] >= start_sec - 0.5 and p["at_seconds"] <= end_sec + 0.5
            ]
            if pause_flags:
                p = pause_flags[0]
                chunk_text += f"\n  [Pause: {p['duration_seconds']}s at {p['at_timestamp']}]"

            docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "source":        file_path,
                    "filename":      Path(file_path).name,
                    "type":          "audio_segment",
                    "content_type":  "audio",
                    "origin":        "audio_upload",
                    "start_sec":     round(start_sec, 2),
                    "end_sec":       round(end_sec, 2),
                    "start_ts":      fmt_ts(start_sec),
                    "end_ts":        fmt_ts(end_sec),
                    "segment_index": grp_idx,
                    "has_pause":     len(pause_flags) > 0,
                },
            ))

        logger.info(
            f"✅ Audio → {len(docs)} documents (1 header + {len(docs)-1} timestamped segments) "
            f"from {Path(file_path).name}"
        )
        return docs

    except Exception as e:
        logger.error(f"❌ Error processing audio {file_path}: {str(e)}")
        return [Document(
            page_content=f"Audio file: {Path(file_path).name} (processing error: {str(e)})",
            metadata={"source": file_path, "filename": Path(file_path).name,
                      "type": "audio_error", "content_type": "audio", "origin": "audio_upload"},
        )]


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def stitch_frame_narrative(
    frame_docs: List[Document],
    duration_sec: float,
    source_path: str,
) -> Document:
    """Synthesize per-frame GPT-4o descriptions into a coherent scene narrative."""
    try:
        parts = []
        for doc in frame_docs:
            t       = doc.metadata.get("frame_time_sec", "?")
            parts.append(f"[Frame at {int(t)}s]\n{doc.page_content}")

        combined = "\n\n---\n\n".join(parts)
        prompt = (
            f"You are analyzing sampled frames from a {int(duration_sec)}-second video. "
            f"The frames were captured every 30 seconds. Below are GPT-4o descriptions of each frame.\n\n"
            f"Your task: Write a single, coherent SCENE NARRATIVE that describes the video as a whole.\n"
            f"Focus on:\n"
            f"1. OVERALL TOPIC: What is this video about?\n"
            f"2. SCENE PROGRESSION: How does the content/setting/activity change over time?\n"
            f"3. KEY MOMENTS: Notable actions, text visible on screen, people, objects, or events at specific timestamps.\n"
            f"4. CONTEXT & PURPOSE: What is the purpose of this video?\n"
            f"5. SEARCHABLE KEYWORDS: List 15+ keywords.\n\n"
            f"Frame-by-frame descriptions:\n{combined}\n\n"
            f"Respond with a structured scene narrative (synthesize, do not list frame summaries)."
        )

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            timeout=60,
        )
        narrative = response.choices[0].message.content.strip()
        logger.info(f"✅ Temporal scene narrative stitched from {len(frame_docs)} frames")

    except Exception as e:
        logger.warning(f"⚠️ Frame narrative stitching failed: {e} — falling back to concatenation")
        narrative = "\n\n".join([
            f"[{int(d.metadata.get('frame_time_sec', 0))}s] {d.page_content}"
            for d in frame_docs
        ])

    return Document(
        page_content=(
            f"[Video Scene Narrative | Duration: {int(duration_sec)}s | Frames: {len(frame_docs)}]\n\n"
            f"{narrative}"
        ),
        metadata={
            "source":       source_path,
            "type":         "video_scene",
            "content_type": "video",
            "duration_sec": duration_sec,
            "frame_count":  len(frame_docs),
            "filename":     Path(source_path).name,
            "origin":       "video_frame_narrative",
        },
    )


def process_video(file_path: str, provider: str = None, api_key: str = None) -> dict:
    """
    Process video file.
    Returns {"transcript_doc", "transcript_segments", "frame_docs"}.
    """
    result = {"transcript_doc": None, "frame_docs": [], "transcript_segments": []}

    # 1. Audio transcript
    try:
        audio_docs = process_audio(file_path)
        video_audio_docs = []
        for doc in audio_docs:
            doc.metadata["content_type"] = "video"
            doc.metadata["origin"]       = "video_audio_track"
            doc.metadata["type"]         = doc.metadata.get("type", "").replace("audio_", "video_")
            video_audio_docs.append(doc)
        result["transcript_segments"] = video_audio_docs
        result["transcript_doc"] = video_audio_docs[0] if video_audio_docs else None
        logger.info(
            f"✅ Audio track extracted from video: {Path(file_path).name} "
            f"({len(video_audio_docs)} timestamped docs)"
        )
    except Exception as e:
        logger.error(f"❌ Audio extraction failed for video {file_path}: {e}")

    # 2. Visual frame extraction + per-frame vision description
    raw_frame_docs = []
    duration_sec   = 0.0
    try:
        cap          = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 1
        duration_sec = total_frames / fps

        video_frame_prompt_prefix = (
            f"This image is a frame extracted from a video at {{timestamp}}s "
            f"(total video duration: {int(duration_sec)}s). "
            "Focus on describing what is HAPPENING in this scene — actions, people, "
            "visible text, objects — as part of a continuous video, not just a static photo.\n\n"
        )

        sample_interval_sec = 30
        sample_times = [
            i * sample_interval_sec
            for i in range(int(duration_sec / sample_interval_sec) + 1)
        ][:10]

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_filename = f"{Path(file_path).stem}_frame_{int(t)}s.jpg"
            frame_path     = os.path.join(UPLOAD_DIR, frame_filename)
            cv2.imwrite(frame_path, frame)

            try:
                original_prompt = image_captioning_service.VISION_PROMPT
                image_captioning_service.VISION_PROMPT = (
                    video_frame_prompt_prefix.format(timestamp=int(t)) + original_prompt
                )
                description = image_captioning_service.generate_comprehensive_description(
                    frame_path, provider=provider, api_key=api_key
                )
                image_captioning_service.VISION_PROMPT = original_prompt

                m, s      = divmod(int(t), 60)
                ts_label  = f"{m:02d}:{s:02d}"
                raw_frame_docs.append(Document(
                    page_content=f"[Video Frame at {ts_label} ({int(t)}s)] {description}",
                    metadata={
                        "source":          file_path,
                        "type":            "video_frame",
                        "content_type":    "video",
                        "frame_time_sec":  float(t),
                        "frame_timestamp": ts_label,
                        "filename":        Path(file_path).name,
                        "origin":          "video_frame",
                    },
                ))
                logger.info(f"✅ Frame at {ts_label} described via {provider or 'auto'}")
            except Exception as e:
                image_captioning_service.VISION_PROMPT = original_prompt
                logger.warning(f"⚠️ Frame description failed at {int(t)}s: {e}")

        cap.release()
    except Exception as e:
        logger.error(f"❌ Frame extraction failed for video {file_path}: {e}")

    # 3. Stitch frames into one temporal narrative
    if raw_frame_docs:
        result["frame_docs"] = [stitch_frame_narrative(raw_frame_docs, duration_sec, file_path)]
    else:
        result["frame_docs"] = []

    return result

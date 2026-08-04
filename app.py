import base64
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import uuid
from typing import List, Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Query, Request
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel

RENDER_API_KEY = os.environ.get("RENDER_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "reels")
FONT_ITALIC_PATH = "/app/fonts/Poppins-Italic.ttf"
FONT_BOLD_ITALIC_PATH = "/app/fonts/Poppins-BoldItalic.ttf"
WIDTH, HEIGHT = 1080, 1920
PHRASE_FONT_SIZE = 54
LINE_SPACING = 8
REF_FONT_SIZE = 32
REF_GAP = 34
SHADOW_OFFSET = (0, 6)
SHADOW_BLUR_RADIUS = 5
SHADOW_ALPHA = 150
CENTER_SCRIM_HEIGHT = 550
SCRIM_MAX_ALPHA = 130
SLIDESHOW_SEGMENT_DURATION = 0.2
MIN_DURATION = 4.0
MAX_DURATION = 60.0
THUMB_OFFSET_SECONDS = 3.0  # matches IG's thumb_offset / TikTok's video_cover_timestamp_ms

app = FastAPI()


class RenderRequest(BaseModel):
    style: str = "zoom"  # "zoom" (single image, Ken Burns), "slideshow" (fast cuts), or "video" (real video clips)
    image_url: Optional[str] = None
    image_urls: Optional[List[str]] = None
    image_b64: Optional[str] = None
    video_urls: Optional[List[str]] = None  # candidate real-video clips for style="video"
    srt_url: Optional[str] = None  # real timed captions for style="video" -- overrides phrase/highlight_word on-screen
    custom_thumbnail_url: Optional[str] = None  # skips auto frame-grab, uses this image as cover instead
    phrase: str
    highlight_word: Optional[str] = None
    verse_reference: Optional[str] = None
    music_url: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/render")
def render(req: RenderRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    if req.style == "slideshow" and not req.image_urls:
        raise HTTPException(status_code=422, detail="image_urls is required for slideshow style")
    if req.style == "video" and not req.video_urls:
        raise HTTPException(status_code=422, detail="video_urls is required for video style")
    if req.style not in ("slideshow", "video") and not req.image_url and not req.image_b64:
        raise HTTPException(status_code=422, detail="either image_url or image_b64 is required")

    workdir = tempfile.mkdtemp(prefix="render_")
    try:
        music_path = os.path.join(workdir, "music" + _guess_ext(req.music_url))
        overlay_path = os.path.join(workdir, "overlay.png")
        output_path = os.path.join(workdir, f"{uuid.uuid4().hex}.mp4")
        _download(req.music_url, music_path)

        duration = _probe_duration(music_path)
        duration = max(MIN_DURATION, min(duration, MAX_DURATION))

        overlay_img = _create_text_overlay(req.phrase, req.highlight_word, req.verse_reference)
        overlay_img.save(overlay_path)

        if req.style == "slideshow":
            photo_paths = []
            for i, url in enumerate(req.image_urls):
                p = os.path.join(workdir, f"photo_{i}.jpg")
                _download(url, p)
                photo_paths.append(p)
            _run_ffmpeg_slideshow(photo_paths, overlay_path, music_path, output_path, duration)
        elif req.style == "video":
            clip_paths = []
            for i, url in enumerate(req.video_urls):
                p = os.path.join(workdir, f"clip_{i}" + _guess_ext(url))
                _download(url, p)
                clip_paths.append(p)

            timed_overlays = None
            if req.srt_url:
                # Real on-screen captions synced to what's actually being said in the audio,
                # instead of the LLM-generated phrase (which has nothing to do with this
                # particular audio track) -- same font/highlight-word visual treatment as
                # before, just timed per caption line instead of one static phrase.
                srt_path = os.path.join(workdir, "captions.srt")
                _download(req.srt_url, srt_path)
                with open(srt_path, encoding="utf-8", errors="ignore") as f:
                    srt_segments = _parse_srt(f.read())
                built_overlays = []
                for i, seg in enumerate(srt_segments):
                    if seg["start"] >= duration:
                        break
                    seg_overlay_path = os.path.join(workdir, f"caption_{i}.png")
                    _create_text_overlay(seg["text"], _pick_highlight_word(seg["text"])).save(seg_overlay_path)
                    built_overlays.append((seg_overlay_path, seg["start"], min(seg["end"], duration)))
                if built_overlays:
                    timed_overlays = built_overlays
                # if the .srt failed to parse into anything usable, timed_overlays stays None
                # and _run_ffmpeg_video_source falls back to the single static phrase overlay.

            _run_ffmpeg_video_source(clip_paths, overlay_path, music_path, output_path, duration, timed_overlays)
        else:
            bg_path = os.path.join(workdir, "bg.jpg")
            if req.image_b64:
                with open(bg_path, "wb") as f:
                    f.write(base64.b64decode(req.image_b64))
            else:
                _download(req.image_url, bg_path)
            _run_ffmpeg(bg_path, overlay_path, music_path, output_path, duration)

        ig_thumb_offset_ms = int(THUMB_OFFSET_SECONDS * 1000)
        thumbnail_url = None
        if req.custom_thumbnail_url:
            # Custom cover image: Facebook/YouTube accept it directly (returned as-is below).
            # Instagram's API can only grab a frame FROM the video itself, so we append a
            # near-invisible 0.1s tail showing this same image and point IG's thumb_offset there.
            thumb_img_path = os.path.join(workdir, "custom_thumb" + _guess_ext(req.custom_thumbnail_url))
            _download(req.custom_thumbnail_url, thumb_img_path)
            tailed_path = os.path.join(workdir, f"{uuid.uuid4().hex}_tailed.mp4")
            ig_thumb_offset_ms = _append_thumbnail_tail(output_path, thumb_img_path, tailed_path)
            output_path = tailed_path
            # Source thumbnails from the pendrive can be several MB (well past YouTube's 2MB
            # thumbnail limit) -- re-encode a compressed JPEG copy for platform uploads instead
            # of passing the raw file through.
            compressed_path = os.path.join(workdir, "custom_thumb_compressed.jpg")
            _compress_thumbnail(thumb_img_path, compressed_path)
            thumbnail_url = _upload_to_supabase(
                compressed_path, folder="thumbs/", ext=".jpg", content_type="image/jpeg"
            )
        else:
            thumb_path = os.path.join(workdir, "thumb.jpg")
            if _extract_thumbnail(output_path, duration, thumb_path):
                thumbnail_url = _upload_to_supabase(
                    thumb_path, folder="thumbs/", ext=".jpg", content_type="image/jpeg"
                )

        video_url = _upload_to_supabase(output_path)
        shutil.rmtree(workdir, ignore_errors=True)
        return {"video_url": video_url, "thumbnail_url": thumbnail_url, "ig_thumb_offset_ms": ig_thumb_offset_ms}
    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


class ExtractAudioRequest(BaseModel):
    video_url: str


@app.post("/extract-audio")
def extract_audio(req: ExtractAudioRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    workdir = tempfile.mkdtemp(prefix="extract_")
    try:
        video_path = os.path.join(workdir, "input" + _guess_ext(req.video_url))
        audio_path = os.path.join(workdir, f"{uuid.uuid4().hex}.mp3")
        _download(req.video_url, video_path)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-q:a", "2",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg (extract-audio) failed: {result.stderr[-2000:]}")

        audio_url = _upload_to_supabase(audio_path, folder="musicas/", ext=".mp3", content_type="audio/mpeg")
        shutil.rmtree(workdir, ignore_errors=True)
        return {"audio_url": audio_url}
    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-raw")
async def upload_raw(
    request: Request,
    folder: str = Query(...),
    ext: str = Query(...),
    content_type: str = Query(default="application/octet-stream"),
    x_api_key: str = Header(default=""),
):
    # Lets n8n (or any caller) push a raw file straight into Supabase Storage without
    # needing Supabase's own service-role secret -- this service already holds it correctly
    # (SUPABASE_SERVICE_KEY env var) and its _upload_to_supabase() helper sends both the
    # apikey and Authorization headers Supabase's object-create endpoint actually requires.
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=422, detail="empty body")

    workdir = tempfile.mkdtemp(prefix="uploadraw_")
    try:
        tmp_path = os.path.join(workdir, f"file{ext}")
        with open(tmp_path, "wb") as f:
            f.write(body)
        url = _upload_to_supabase(tmp_path, folder=folder, ext=ext, content_type=content_type)
        return {"url": url}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _upload_to_supabase(
    file_path: str, folder: str = "", ext: str = ".mp4", content_type: str = "video/mp4"
) -> str:
    filename = f"{folder}{uuid.uuid4().hex}{ext}"
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
                "Content-Type": content_type,
            },
            data=f,
            timeout=60,
        )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"supabase upload failed: {r.text}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"


def _guess_ext(url: str) -> str:
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1]
    return ext if ext else ".mp3"


def _download(url: str, dest: str) -> None:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)


def _probe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {result.stderr[-500:]}")
    return float(result.stdout.strip())


def _extract_thumbnail(video_path: str, duration: float, out_path: str) -> bool:
    # Grabs a frame past the fade-in-from-black intro so platforms that default
    # to frame 0 (Facebook Reels) don't show a solid black cover.
    offset = min(THUMB_OFFSET_SECONDS, max(0.0, duration - 0.1))
    cmd = ["ffmpeg", "-y", "-ss", str(offset), "-i", video_path, "-frames:v", "1", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(out_path)


THUMB_MAX_BYTES = 2 * 1024 * 1024  # YouTube's documented thumbnail upload limit
THUMB_MAX_DIMENSION = 1280  # plenty for any platform's cover image, keeps file size well under the limit


def _compress_thumbnail(src_path: str, out_path: str) -> None:
    img = Image.open(src_path)
    if img.mode in ("RGBA", "P", "LA"):
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    if max(img.size) > THUMB_MAX_DIMENSION:
        ratio = THUMB_MAX_DIMENSION / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.LANCZOS)

    quality = 90
    while quality >= 40:
        img.save(out_path, "JPEG", quality=quality)
        if os.path.getsize(out_path) <= THUMB_MAX_BYTES:
            return
        quality -= 10
    # last resort, whatever quality=40 produced is what we ship


_SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")
_PT_STOPWORDS = {
    "de", "da", "do", "das", "dos", "que", "um", "uma", "uns", "umas", "e", "a", "o", "as", "os",
    "em", "no", "na", "nos", "nas", "para", "por", "com", "se", "sua", "seu", "suas", "seus",
    "ao", "aos", "a", "as", "nao", "mais", "mas", "como", "ja", "tem", "vai", "vou", "voce",
    "eu", "ele", "ela", "nos", "isso", "essa", "esse", "muito", "so", "ta", "pra", "pro", "que",
    "quem", "quando", "onde", "porque", "entao", "la", "aqui", "sao", "foi", "ser", "estar",
}


def _parse_srt(text: str) -> list[dict]:
    # Standard .srt format: blocks of "index\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext...", separated
    # by a blank line. Tolerates the timestamp using either a comma or a dot before milliseconds,
    # and strips any inline HTML-ish styling tags (<i>, <b>, etc) some downloaders leave in.
    blocks = re.split(r"\n\s*\n", text.strip())
    segments = []
    for block in blocks:
        lines = [l for l in block.strip().split("\n") if l.strip()]
        time_line_idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if time_line_idx is None:
            continue
        m = _SRT_TIME_RE.search(lines[time_line_idx])
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        caption_text = " ".join(lines[time_line_idx + 1:]).strip()
        caption_text = _TAG_RE.sub("", caption_text)
        if caption_text and end > start:
            segments.append({"start": start, "end": end, "text": caption_text})
    segments.sort(key=lambda s: s["start"])
    return segments


def _pick_highlight_word(text: str) -> Optional[str]:
    # No LLM-picked highlight word exists for real transcript captions -- approximate the same
    # "one emphasized word per line" visual style with a simple, free heuristic instead: the
    # longest word that isn't a common stopword. Not always the objectively best choice, but
    # keeps the same look without an extra paid call for what's just a formatting/emphasis detail.
    words = [w.strip(".,!?;:\"'()") for w in text.split()]
    candidates = [w for w in words if w and len(w) > 2 and _fold_accents(w).lower() not in _PT_STOPWORDS]
    if not candidates:
        return None
    return max(candidates, key=len)


def _fold_accents(s: str) -> str:
    # The LLM occasionally generates the phrase and the highlighted word with slightly
    # different accenting (e.g. "Idolos" vs "ídolos") -- strip diacritics before comparing
    # so the highlight still matches instead of silently finding no word to bold.
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _find_highlight_index(words: list[str], highlight_word: Optional[str]) -> Optional[int]:
    if not highlight_word:
        return None
    target = _fold_accents(highlight_word.strip(".,!?;:\"'()").lower())
    for i, word in enumerate(words):
        if _fold_accents(word.strip(".,!?;:\"'()").lower()) == target:
            return i
    return None


def _wrap_words_mixed(
    draw: ImageDraw.ImageDraw, words: list[str], highlight_idx: Optional[int],
    font_regular: ImageFont.FreeTypeFont, font_bold: ImageFont.FreeTypeFont, max_width: int
) -> list[list[tuple]]:
    space_width = draw.textlength(" ", font=font_regular)
    lines: list[list[tuple]] = []
    current: list[tuple] = []
    current_width = 0.0
    for i, word in enumerate(words):
        font = font_bold if i == highlight_idx else font_regular
        bbox = draw.textbbox((0, 0), word, font=font)
        word_width = bbox[2] - bbox[0]
        extra = (space_width if current else 0) + word_width
        if current and current_width + extra > max_width:
            lines.append(current)
            current = []
            current_width = 0.0
            extra = word_width
        current.append((word, font, word_width))
        current_width += extra
    if current:
        lines.append(current)
    return lines


def _draw_center_scrim(img: Image.Image, y_start: int, y_end: int, max_alpha: int) -> None:
    draw = ImageDraw.Draw(img)
    height = y_end - y_start
    center = height / 2
    for i in range(height):
        t = 1 - abs(i - center) / center
        alpha = int(max_alpha * max(t, 0))
        draw.line([(0, y_start + i), (WIDTH, y_start + i)], fill=(0, 0, 0, alpha))


def _create_text_overlay(
    phrase: str, highlight_word: Optional[str] = None, verse_reference: Optional[str] = None
) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    max_text_width = int(WIDTH * 0.85)
    font_italic = ImageFont.truetype(FONT_ITALIC_PATH, PHRASE_FONT_SIZE)
    font_bold_italic = ImageFont.truetype(FONT_BOLD_ITALIC_PATH, PHRASE_FONT_SIZE)
    font_ref = ImageFont.truetype(FONT_ITALIC_PATH, REF_FONT_SIZE)

    words = phrase.split()
    highlight_idx = _find_highlight_index(words, highlight_word)
    lines = _wrap_words_mixed(probe, words, highlight_idx, font_italic, font_bold_italic, max_text_width)
    space_width = probe.textlength(" ", font=font_italic)

    ref_bbox = probe.textbbox((0, 0), "Ág", font=font_italic)
    line_height = ref_bbox[3] - ref_bbox[1]
    phrase_height = len(lines) * line_height + (len(lines) - 1) * LINE_SPACING

    ref_text = f"({verse_reference})" if verse_reference else None
    ref_line_height = 0
    if ref_text:
        rb = probe.textbbox((0, 0), ref_text, font=font_ref)
        ref_line_height = rb[3] - rb[1]

    total_height = phrase_height + (REF_GAP + ref_line_height if ref_text else 0)
    scrim_height = max(CENTER_SCRIM_HEIGHT, total_height + 160)

    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    _draw_center_scrim(img, HEIGHT // 2 - scrim_height // 2, HEIGHT // 2 + scrim_height // 2, SCRIM_MAX_ALPHA)

    y = HEIGHT // 2 - total_height // 2

    # Soft drop shadow layer, blurred and composited behind the crisp text for a sense of depth.
    shadow_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    word_positions = []
    for line in lines:
        line_width = sum(w[2] for w in line) + space_width * (len(line) - 1)
        x = (WIDTH - line_width) // 2
        for word, font, word_width in line:
            word_positions.append((x, y, word, font))
            shadow_draw.text(
                (x + SHADOW_OFFSET[0], y + SHADOW_OFFSET[1]), word, font=font, fill=(0, 0, 0, SHADOW_ALPHA)
            )
            x += word_width + space_width
        y += line_height + LINE_SPACING

    ref_position = None
    if ref_text:
        y += REF_GAP - LINE_SPACING
        ref_width = probe.textlength(ref_text, font=font_ref)
        rx = (WIDTH - ref_width) // 2
        ref_position = (rx, y)
        shadow_draw.text(
            (rx + SHADOW_OFFSET[0], y + SHADOW_OFFSET[1]), ref_text, font=font_ref, fill=(0, 0, 0, SHADOW_ALPHA)
        )

    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
    img = Image.alpha_composite(img, shadow_layer)

    draw = ImageDraw.Draw(img)
    for x, y, word, font in word_positions:
        draw.text((x, y), word, font=font, fill="white", stroke_width=3, stroke_fill="black")
    if ref_position:
        draw.text(ref_position, ref_text, font=font_ref, fill="white", stroke_width=2, stroke_fill="black")

    return img


def _run_ffmpeg(bg_path: str, overlay_path: str, music_path: str, output_path: str, duration: float) -> None:
    fade_dur = 1.2
    text_start = fade_dur + 0.1
    audio_fade_out_start = max(duration - 0.5, 0)

    zoom_w, zoom_h = WIDTH * 2, HEIGHT * 2
    total_frames = int(round(duration * 30))

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", bg_path,
        "-loop", "1", "-t", str(duration), "-i", overlay_path,
        "-i", music_path,
        "-filter_complex",
        f"[0:v]scale={zoom_w}:{zoom_h}:force_original_aspect_ratio=increase,"
        f"crop={zoom_w}:{zoom_h},"
        f"zoompan=z='min(zoom+0.0022,1.45)':d={total_frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps=30,"
        f"eq=contrast=1.18:brightness=-0.05:saturation=0.82,"
        f"colorbalance=rs=0.05:gs=0:bs=-0.1,"
        f"vignette=PI/3.5,"
        f"fade=t=in:st=0:d={fade_dur}[bg];"
        f"[bg][1:v]overlay=0:0:enable='gte(t,{text_start})'[outv]",
        "-map", "[outv]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "128k",
        "-af", f"afade=t=in:st=0:d=0.5,afade=t=out:st={audio_fade_out_start}:d=0.5",
        "-t", str(duration),
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {result.stderr[-2000:]}")


def _run_ffmpeg_slideshow(
    photo_paths: list[str], overlay_path: str, music_path: str, output_path: str, duration: float
) -> None:
    fade_dur = 1.2
    text_start = fade_dur + 0.1
    audio_fade_out_start = max(duration - 0.5, 0)
    workdir = os.path.dirname(output_path)

    # Render each unique photo into its own tiny, self-contained clip first. Reusing a single
    # split "infinite loop" stream across dozens of independent trims in one filter graph is a
    # known ffmpeg trouble spot (frames from later trims can come back with the wrong
    # dimensions) -- pre-materializing finite clips and concatenating them sidesteps that.
    segment_paths = []
    for i, photo_path in enumerate(photo_paths):
        seg_path = os.path.join(workdir, f"slide_seg_{i}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-r", "30", "-i", photo_path,
            "-t", str(SLIDESHOW_SEGMENT_DURATION),
            "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
            seg_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg (slideshow segment {i}) failed: {result.stderr[-1500:]}")
        segment_paths.append(seg_path)

    num_segments = max(1, round(duration / SLIDESHOW_SEGMENT_DURATION))
    concat_list_path = os.path.join(workdir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for i in range(num_segments):
            seg = segment_paths[i % len(segment_paths)]
            f.write(f"file '{seg}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-loop", "1", "-t", str(duration), "-i", overlay_path,
        "-i", music_path,
        "-filter_complex",
        f"[0:v]eq=contrast=1.18:brightness=-0.05:saturation=0.82,"
        f"colorbalance=rs=0.05:gs=0:bs=-0.1,"
        f"vignette=PI/3.5,"
        f"fade=t=in:st=0:d={fade_dur}[bg];"
        f"[bg][1:v]overlay=0:0:enable='gte(t,{text_start})'[outv]",
        "-map", "[outv]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "128k",
        "-af", f"afade=t=in:st=0:d=0.5,afade=t=out:st={audio_fade_out_start}:d=0.5",
        "-t", str(duration),
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg (slideshow concat) failed: {result.stderr[-2000:]}")


def _run_ffmpeg_video_source(
    video_paths: list[str], overlay_path: str, music_path: str, output_path: str, duration: float,
    timed_overlays: Optional[list[tuple[str, float, float]]] = None,
) -> None:
    fade_dur = 1.2
    text_start = fade_dur + 0.1
    audio_fade_out_start = max(duration - 0.5, 0)

    # (overlay_image_path, start_seconds, end_seconds) for each caption to show, in order.
    # Defaults to the single static overlay shown for the whole clip (the original behavior).
    if timed_overlays is None:
        timed_overlays = [(overlay_path, text_start, duration)]

    # Figure out which candidate clips to use (looping the pool if one pass isn't enough) and how
    # much of each, to cover the target duration exactly. Trimmed/scaled/concatenated in a single
    # filter_complex pass, rather than pre-encoding each clip to its own file first and then
    # re-encoding *again* on concat -- that extra encode pass was visibly softening real video
    # footage (unlike a still photo, real video has much more fine detail/motion to lose from a
    # second lossy pass).
    segments = []  # (path, clip_duration)
    accumulated = 0.0
    idx = 0
    guard = 0
    while accumulated < duration and guard < len(video_paths) * 5 + 5:
        guard += 1
        src = video_paths[idx % len(video_paths)]
        idx += 1
        src_duration = _probe_duration(src)
        remaining = duration - accumulated
        clip_duration = min(src_duration, remaining)
        if clip_duration <= 0.05:
            continue
        segments.append((src, clip_duration))
        accumulated += clip_duration

    if not segments:
        raise HTTPException(status_code=500, detail="no usable video segments to cover the audio duration")

    inputs = []
    trim_filters = []
    concat_labels = []
    for i, (path, clip_duration) in enumerate(segments):
        inputs += ["-i", path]
        trim_filters.append(
            f"[{i}:v]trim=duration={clip_duration},setpts=PTS-STARTPTS,"
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},setsar=1,fps=30[c{i}]"
        )
        concat_labels.append(f"[c{i}]")
    concat_filter = "".join(concat_labels) + f"concat=n={len(segments)}:v=1:a=0[rawv]"

    overlay_input_base = len(segments)
    for op, _, _ in timed_overlays:
        inputs += ["-loop", "1", "-t", str(duration), "-i", op]
    music_input_idx = overlay_input_base + len(timed_overlays)
    inputs += ["-i", music_path]

    grade_filter = (
        f"[rawv]eq=contrast=1.18:brightness=-0.05:saturation=0.82,"
        f"colorbalance=rs=0.05:gs=0:bs=-0.1,"
        f"vignette=PI/3.5,"
        f"fade=t=in:st=0:d={fade_dur}[bg0]"
    )

    overlay_chain = []
    prev_label = "bg0"
    n_overlays = len(timed_overlays)
    for oi, (_, start, end) in enumerate(timed_overlays):
        in_idx = overlay_input_base + oi
        out_label = "outv" if oi == n_overlays - 1 else f"bg{oi + 1}"
        overlay_chain.append(
            f"[{prev_label}][{in_idx}:v]overlay=0:0:enable='between(t,{start},{end})'[{out_label}]"
        )
        prev_label = out_label

    filter_complex = ";".join(trim_filters + [concat_filter, grade_filter] + overlay_chain)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", f"{music_input_idx}:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-af", f"afade=t=in:st=0:d=0.5,afade=t=out:st={audio_fade_out_start}:d=0.5",
        "-t", str(duration),
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg (video concat) failed: {result.stderr[-2000:]}")


def _append_thumbnail_tail(video_path: str, thumb_path: str, output_path: str) -> int:
    # Appends a near-invisible 0.1s tail showing the custom thumbnail image, on the video
    # stream only (the audio track stays at its original length -- players just go silent
    # for that last 0.1s, imperceptible). Lets Instagram's thumb_offset -- which can only grab
    # a frame FROM the video itself, never an arbitrary separate image -- point at a moment
    # that actually IS the custom thumbnail. Returns the ms offset to use for thumb_offset.
    workdir = os.path.dirname(output_path)
    tail_duration = 0.1

    main_duration = _probe_duration(video_path)

    tail_path = os.path.join(workdir, "thumb_tail.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-r", "30", "-i", thumb_path,
        "-t", str(tail_duration),
        "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        tail_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg (thumb tail) failed: {result.stderr[-1500:]}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", tail_path,
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
        "-map", "[outv]", "-map", "0:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg (thumb tail concat) failed: {result.stderr[-2000:]}")

    return int((main_duration + tail_duration / 2) * 1000)

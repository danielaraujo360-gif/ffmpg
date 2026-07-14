import base64
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import List, Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel

RENDER_API_KEY = os.environ.get("RENDER_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "reels")
FONT_PATH = "/app/fonts/Poppins-Bold.ttf"
FONT_ITALIC_PATH = "/app/fonts/Poppins-Italic.ttf"
FONT_BOLD_ITALIC_PATH = "/app/fonts/Poppins-BoldItalic.ttf"
WIDTH, HEIGHT = 1080, 1920
PHRASE_FONT_SIZE = 54
WATERMARK_TEXT = "@divindadesabedoria_"
WATERMARK_FONT_SIZE = 30
WATERMARK_BOTTOM_MARGIN = 260
LINE_SPACING = 8
SHADOW_OFFSET = (0, 6)
SHADOW_BLUR_RADIUS = 5
SHADOW_ALPHA = 150
BOTTOM_SCRIM_HEIGHT = 420
CENTER_SCRIM_HEIGHT = 550
SCRIM_MAX_ALPHA = 130
SLIDESHOW_SEGMENT_DURATION = 0.2

app = FastAPI()


class RenderRequest(BaseModel):
    style: str = "zoom"  # "zoom" (single image, Ken Burns) or "slideshow" (fast cuts across image_urls)
    image_url: Optional[str] = None
    image_urls: Optional[List[str]] = None
    image_b64: Optional[str] = None
    phrase: str
    highlight_word: Optional[str] = None
    music_url: str
    duration: float = 7.5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/render")
def render(req: RenderRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    if req.style == "slideshow" and not req.image_urls:
        raise HTTPException(status_code=422, detail="image_urls is required for slideshow style")
    if req.style != "slideshow" and not req.image_url and not req.image_b64:
        raise HTTPException(status_code=422, detail="either image_url or image_b64 is required")

    workdir = tempfile.mkdtemp(prefix="render_")
    try:
        music_path = os.path.join(workdir, "music" + _guess_ext(req.music_url))
        overlay_path = os.path.join(workdir, "overlay.png")
        output_path = os.path.join(workdir, f"{uuid.uuid4().hex}.mp4")
        _download(req.music_url, music_path)

        overlay_img = _create_text_overlay(req.phrase, req.highlight_word)
        overlay_img.save(overlay_path)

        if req.style == "slideshow":
            photo_paths = []
            for i, url in enumerate(req.image_urls):
                p = os.path.join(workdir, f"photo_{i}.jpg")
                _download(url, p)
                photo_paths.append(p)
            _run_ffmpeg_slideshow(photo_paths, overlay_path, music_path, output_path, req.duration)
        else:
            bg_path = os.path.join(workdir, "bg.jpg")
            if req.image_b64:
                with open(bg_path, "wb") as f:
                    f.write(base64.b64decode(req.image_b64))
            else:
                _download(req.image_url, bg_path)
            _run_ffmpeg(bg_path, overlay_path, music_path, output_path, req.duration)

        video_url = _upload_to_supabase(output_path)
        shutil.rmtree(workdir, ignore_errors=True)
        return {"video_url": video_url}
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


def _find_highlight_index(words: list[str], highlight_word: Optional[str]) -> Optional[int]:
    if not highlight_word:
        return None
    target = highlight_word.strip(".,!?;:\"'()").lower()
    for i, word in enumerate(words):
        if word.strip(".,!?;:\"'()").lower() == target:
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


def _draw_gradient_scrim(img: Image.Image, y_start: int, y_end: int, max_alpha: int, fade_toward: str) -> None:
    draw = ImageDraw.Draw(img)
    height = y_end - y_start
    for i in range(height):
        t = (1 - i / height) if fade_toward == "top" else (i / height)
        alpha = int(max_alpha * t)
        draw.line([(0, y_start + i), (WIDTH, y_start + i)], fill=(0, 0, 0, alpha))


def _draw_center_scrim(img: Image.Image, y_start: int, y_end: int, max_alpha: int) -> None:
    draw = ImageDraw.Draw(img)
    height = y_end - y_start
    center = height / 2
    for i in range(height):
        t = 1 - abs(i - center) / center
        alpha = int(max_alpha * max(t, 0))
        draw.line([(0, y_start + i), (WIDTH, y_start + i)], fill=(0, 0, 0, alpha))


def _create_text_overlay(phrase: str, highlight_word: Optional[str] = None) -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    _draw_center_scrim(img, HEIGHT // 2 - CENTER_SCRIM_HEIGHT // 2, HEIGHT // 2 + CENTER_SCRIM_HEIGHT // 2, SCRIM_MAX_ALPHA)
    _draw_gradient_scrim(img, HEIGHT - BOTTOM_SCRIM_HEIGHT, HEIGHT, SCRIM_MAX_ALPHA, fade_toward="bottom")

    draw = ImageDraw.Draw(img)
    max_text_width = int(WIDTH * 0.85)
    font_italic = ImageFont.truetype(FONT_ITALIC_PATH, PHRASE_FONT_SIZE)
    font_bold_italic = ImageFont.truetype(FONT_BOLD_ITALIC_PATH, PHRASE_FONT_SIZE)
    watermark_font = ImageFont.truetype(FONT_PATH, WATERMARK_FONT_SIZE)

    words = phrase.split()
    highlight_idx = _find_highlight_index(words, highlight_word)
    lines = _wrap_words_mixed(draw, words, highlight_idx, font_italic, font_bold_italic, max_text_width)
    space_width = draw.textlength(" ", font=font_italic)

    ref_bbox = draw.textbbox((0, 0), "Ág", font=font_italic)
    line_height = ref_bbox[3] - ref_bbox[1]
    total_height = len(lines) * line_height + (len(lines) - 1) * LINE_SPACING
    y = HEIGHT // 2 - total_height // 2

    wm_bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=watermark_font)
    wm_x = (WIDTH - (wm_bbox[2] - wm_bbox[0])) // 2
    wm_y = HEIGHT - WATERMARK_BOTTOM_MARGIN

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
    shadow_draw.text(
        (wm_x + SHADOW_OFFSET[0], wm_y + SHADOW_OFFSET[1]), WATERMARK_TEXT, font=watermark_font,
        fill=(0, 0, 0, SHADOW_ALPHA),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
    img = Image.alpha_composite(img, shadow_layer)

    draw = ImageDraw.Draw(img)
    for x, y, word, font in word_positions:
        draw.text((x, y), word, font=font, fill="white", stroke_width=3, stroke_fill="black")
    draw.text(
        (wm_x, wm_y), WATERMARK_TEXT, font=watermark_font,
        fill=(255, 255, 255, 200), stroke_width=1, stroke_fill=(0, 0, 0, 200),
    )

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

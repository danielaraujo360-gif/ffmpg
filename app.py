import base64
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel

RENDER_API_KEY = os.environ.get("RENDER_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "reels")
FONT_PATH = "/app/fonts/Poppins-Bold.ttf"
WIDTH, HEIGHT = 1080, 1920
PHRASE_FONT_SIZE = 64
PHRASE_TOP_Y = 260
WATERMARK_TEXT = "@divindadesabedoria_"
WATERMARK_FONT_SIZE = 30
WATERMARK_BOTTOM_MARGIN = 260
LINE_SPACING = 8
SHADOW_OFFSET = (0, 6)
SHADOW_BLUR_RADIUS = 5
SHADOW_ALPHA = 150
TOP_SCRIM_HEIGHT = 560
BOTTOM_SCRIM_HEIGHT = 420
SCRIM_MAX_ALPHA = 130

app = FastAPI()


class RenderRequest(BaseModel):
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    phrase: str
    music_url: str
    duration: float = 7.5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/render")
def render(req: RenderRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    if not req.image_url and not req.image_b64:
        raise HTTPException(status_code=422, detail="either image_url or image_b64 is required")

    workdir = tempfile.mkdtemp(prefix="render_")
    try:
        bg_path = os.path.join(workdir, "bg.jpg")
        music_path = os.path.join(workdir, "music" + _guess_ext(req.music_url))
        overlay_path = os.path.join(workdir, "overlay.png")
        output_path = os.path.join(workdir, f"{uuid.uuid4().hex}.mp4")

        if req.image_b64:
            with open(bg_path, "wb") as f:
                f.write(base64.b64decode(req.image_b64))
        else:
            _download(req.image_url, bg_path)
        _download(req.music_url, music_path)

        overlay_img = _create_text_overlay(req.phrase)
        overlay_img.save(overlay_path)

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


def _upload_to_supabase(file_path: str) -> str:
    filename = f"{uuid.uuid4().hex}.mp4"
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
                "Content-Type": "video/mp4",
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


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = test
        else:
            lines.append(current)
            current = word
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


def _create_text_overlay(phrase: str) -> Image.Image:
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))

    _draw_gradient_scrim(img, 0, TOP_SCRIM_HEIGHT, SCRIM_MAX_ALPHA, fade_toward="top")
    _draw_gradient_scrim(img, HEIGHT - BOTTOM_SCRIM_HEIGHT, HEIGHT, SCRIM_MAX_ALPHA, fade_toward="bottom")

    draw = ImageDraw.Draw(img)
    max_text_width = int(WIDTH * 0.85)
    font = ImageFont.truetype(FONT_PATH, PHRASE_FONT_SIZE)
    lines = _wrap_text(draw, phrase, font, max_text_width)
    watermark_font = ImageFont.truetype(FONT_PATH, WATERMARK_FONT_SIZE)
    wm_bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=watermark_font)
    wm_x = (WIDTH - (wm_bbox[2] - wm_bbox[0])) // 2
    wm_y = HEIGHT - WATERMARK_BOTTOM_MARGIN

    # Soft drop shadow layer, blurred and composited behind the crisp text for a sense of depth.
    shadow_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    y = PHRASE_TOP_Y
    line_positions = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3]
        x = (WIDTH - (bbox[2] - bbox[0])) // 2
        line_positions.append((x, y, line))
        shadow_draw.text(
            (x + SHADOW_OFFSET[0], y + SHADOW_OFFSET[1]), line, font=font, fill=(0, 0, 0, SHADOW_ALPHA)
        )
        y += line_height + LINE_SPACING
    shadow_draw.text(
        (wm_x + SHADOW_OFFSET[0], wm_y + SHADOW_OFFSET[1]), WATERMARK_TEXT, font=watermark_font,
        fill=(0, 0, 0, SHADOW_ALPHA),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
    img = Image.alpha_composite(img, shadow_layer)

    draw = ImageDraw.Draw(img)
    for x, y, line in line_positions:
        draw.text((x, y), line, font=font, fill="white", stroke_width=3, stroke_fill="black")
    draw.text(
        (wm_x, wm_y), WATERMARK_TEXT, font=watermark_font,
        fill=(255, 255, 255, 200), stroke_width=1, stroke_fill=(0, 0, 0, 200),
    )

    return img


def _run_ffmpeg(bg_path: str, overlay_path: str, music_path: str, output_path: str, duration: float) -> None:
    fade_dur = 1.2
    text_start = fade_dur + 0.1
    audio_fade_out_start = max(duration - 0.5, 0)

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", str(duration), "-i", bg_path,
        "-loop", "1", "-t", str(duration), "-i", overlay_path,
        "-i", music_path,
        "-filter_complex",
        f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
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

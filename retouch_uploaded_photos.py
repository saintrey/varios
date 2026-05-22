from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


INPUT_DIR = Path("photo_work/originals")
OUTPUT_DIR = Path("fotos_retocadas")

ANTI_PIXELATED_FILES = {
    "WhatsApp Image 2026-05-22 at 12.17.41.jpeg",
    "WhatsApp Image 2026-05-22 at 12.17.46.jpeg",
    "WhatsApp Image 2026-05-22 at 12.18.08.jpeg",
}


def _background_seed_colors(arr: np.ndarray) -> np.ndarray:
    """Return conservative background colors sampled from the corners."""
    h, w, _ = arr.shape
    patch_h = max(4, h // 10)
    patch_w = max(4, w // 10)
    patches = [
        arr[:patch_h, :patch_w, :].reshape(-1, 3),
        arr[:patch_h, -patch_w:, :].reshape(-1, 3),
        arr[-patch_h:, :patch_w, :].reshape(-1, 3),
        arr[-patch_h:, -patch_w:, :].reshape(-1, 3),
    ]
    return np.array([np.median(patch, axis=0) for patch in patches], dtype=np.int16)


def _connected_to_edge(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for x in range(w):
        if mask[0, x]:
            out[0, x] = True
            q.append((0, x))
        if mask[h - 1, x]:
            out[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if mask[y, 0] and not out[y, 0]:
            out[y, 0] = True
            q.append((y, 0))
        if mask[y, w - 1] and not out[y, w - 1]:
            out[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def _remove_small_components(bg_mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove isolated remnants from patterned backgrounds while keeping the main art."""
    fg = ~bg_mask
    h, w = fg.shape
    seen = np.zeros_like(fg, dtype=bool)
    cleaned = bg_mask.copy()

    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or not fg[sy, sx]:
                continue

            pts: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(sy, sx)])
            seen[sy, sx] = True
            while q:
                y, x = q.popleft()
                pts.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))

            if len(pts) < min_area:
                ys, xs = zip(*pts)
                cleaned[np.array(ys), np.array(xs)] = True

    return cleaned


def white_background(
    image: Image.Image,
    *,
    remove_all_corner_like: bool = False,
    clean_exterior_gray_residue: bool = False,
) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    chroma = maxc - minc
    lightness = arr.mean(axis=2)

    seed_colors = _background_seed_colors(arr)
    diffs = arr[:, :, None, :] - seed_colors[None, None, :, :]
    distances = np.sqrt((diffs.astype(np.float32) ** 2).sum(axis=3))

    # Solid/photo-exported backgrounds are usually visible in the corners.
    corner_like = distances.min(axis=2) < 58
    # Transparency checkerboards in the uploaded JPGs are light, low-chroma grays.
    checker_like = (chroma < 14) & (lightness > 188) & (lightness < 248)
    # Existing white backgrounds should become clean #fff without changing outlines.
    white_like = (chroma < 18) & (lightness > 242)

    candidate = corner_like | checker_like | white_like
    bg_mask = _connected_to_edge(candidate)
    seed_chroma = seed_colors.max(axis=1) - seed_colors.min(axis=1)
    seed_distance_to_white = np.sqrt(((255 - seed_colors).astype(np.float32) ** 2).sum(axis=1))
    colored_or_patterned_bg = np.median(seed_distance_to_white) > 35 and np.median(seed_chroma) > 30
    if colored_or_patterned_bg:
        bg_mask = _remove_small_components(bg_mask, max(80, arr.shape[0] * arr.shape[1] // 100))
    if clean_exterior_gray_residue:
        gray_residue = (chroma < 34) & (lightness > 60) & (lightness < 252)
        exterior_residue = _connected_to_edge(bg_mask | gray_residue) & gray_residue
        bg_mask |= exterior_residue
    if remove_all_corner_like:
        bg_mask |= corner_like

    cleaned = arr.astype(np.uint8).copy()
    cleaned[bg_mask] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(cleaned, "RGB")


def enhance(image: Image.Image) -> Image.Image:
    # Upscale first so sharpening works on the final edge resolution.
    max_dim = max(image.size)
    scale = 2.0 if max_dim >= 900 else min(4.0, max(2.0, 1200 / max_dim))
    new_size = (round(image.width * scale), round(image.height * scale))
    upscaled = image.resize(new_size, Image.Resampling.LANCZOS)

    upscaled = ImageOps.autocontrast(upscaled, cutoff=0.35)
    upscaled = ImageEnhance.Color(upscaled).enhance(1.08)
    upscaled = ImageEnhance.Contrast(upscaled).enhance(1.05)
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.4, percent=135, threshold=3))
    return ImageEnhance.Sharpness(upscaled).enhance(1.08)


def enhance_low_resolution(image: Image.Image) -> Image.Image:
    max_dim = max(image.size)
    scale = max(4.0, 1800 / max_dim)
    new_size = (round(image.width * scale), round(image.height * scale))

    # Low-resolution sources look blocky when aggressively sharpened, so this
    # path favors smoother interpolation and only a mild final edge recovery.
    upscaled = image.resize(new_size, Image.Resampling.BICUBIC)
    upscaled = upscaled.filter(ImageFilter.GaussianBlur(radius=0.12))
    upscaled = ImageOps.autocontrast(upscaled, cutoff=0.2)
    upscaled = ImageEnhance.Color(upscaled).enhance(1.06)
    upscaled = ImageEnhance.Contrast(upscaled).enhance(1.03)
    return upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=70, threshold=4))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.jpeg")):
        if path.name.startswith("._"):
            continue
        image = Image.open(path).convert("RGB")
        cleaned = white_background(
            image,
            # This source has scattered pastel border marks that are the same
            # color family as the removed corner background.
            remove_all_corner_like=path.name.endswith("12.07.11 (3).jpeg"),
            clean_exterior_gray_residue=path.name
            in {
                "WhatsApp Image 2026-05-22 at 12.07.10 (1).jpeg",
                "WhatsApp Image 2026-05-22 at 12.18.02.jpeg",
                "WhatsApp Image 2026-05-22 at 12.18.08.jpeg",
            },
        )
        final = enhance_low_resolution(cleaned) if path.name in ANTI_PIXELATED_FILES else enhance(cleaned)
        final.save(OUTPUT_DIR / path.name, quality=96, subsampling=0, optimize=True)
        print(f"{path.name}: {image.size} -> {final.size}")


if __name__ == "__main__":
    main()

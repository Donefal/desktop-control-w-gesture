"""
extract_frames.py — Extract every frame of a video as an image file.

Dependencies:
    pip install opencv-python

Usage:
    python extract_frames.py <video_path> [options]

Examples:
    python extract_frames.py video.mp4
    python extract_frames.py video.mp4 --output frames/ --format png
    python extract_frames.py video.mp4 --every 5 --quality 90
"""

import cv2
import os
import argparse
import sys
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    image_format: str = "jpg",
    every_n_frames: int = 1,
    quality: int = 95,
    prefix: str = "frame",
    verbose: bool = True,
) -> int:
    """
    Extract frames from a video file and save them as images.

    Args:
        video_path:     Path to the input video file.
        output_dir:     Directory where extracted frames will be saved.
        image_format:   Output format — 'jpg', 'png', or 'webp'.
        every_n_frames: Save every Nth frame (1 = all frames, 2 = every other, etc.).
        quality:        JPEG/WebP quality (1–100). Ignored for PNG.
        prefix:         Filename prefix for saved frames.
        verbose:        Print progress to stdout.

    Returns:
        Number of frames saved.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if verbose:
        duration = total_frames / fps if fps > 0 else 0
        print(f"Video:      {video_path.name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS:        {fps:.2f}")
        print(f"Frames:     {total_frames}")
        print(f"Duration:   {duration:.2f}s")
        print(f"Saving to:  {output_dir}/")
        print(f"Format:     {image_format.upper()}, every {every_n_frames} frame(s)")
        print("-" * 40)

    # Build encode params
    fmt = image_format.lower()
    if fmt == "jpg":
        ext = ".jpg"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == "png":
        ext = ".png"
        # PNG compression 0–9; map quality 0–100 → 9–0
        compression = max(0, min(9, 9 - round(quality / 100 * 9)))
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    elif fmt == "webp":
        ext = ".webp"
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        raise ValueError(f"Unsupported format '{image_format}'. Use jpg, png, or webp.")

    pad = len(str(total_frames))  # zero-padding width
    saved = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % every_n_frames == 0:
            filename = output_dir / f"{prefix}_{str(frame_index).zfill(pad)}{ext}"
            cv2.imwrite(str(filename), frame, encode_params)
            saved += 1

            if verbose and (saved % 50 == 0 or frame_index == 0):
                pct = (frame_index / total_frames * 100) if total_frames > 0 else 0
                print(f"  [{pct:5.1f}%] Saved frame {frame_index:>{pad}} → {filename.name}")

        frame_index += 1

    cap.release()

    if verbose:
        print("-" * 40)
        print(f"Done. {saved} frame(s) saved to '{output_dir}/'.")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file as images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--output", "-o", default="frames",
        help="Output directory (default: frames/)"
    )
    parser.add_argument(
        "--format", "-f", default="jpg",
        choices=["jpg", "png", "webp"],
        help="Image format (default: jpg)"
    )
    parser.add_argument(
        "--every", "-e", type=int, default=1, metavar="N",
        help="Save every Nth frame — e.g. 5 saves one frame per 5 (default: 1)"
    )
    parser.add_argument(
        "--quality", "-q", type=int, default=95, metavar="Q",
        help="JPEG/WebP quality 1–100 (default: 95; ignored for PNG)"
    )
    parser.add_argument(
        "--prefix", "-p", default="frame",
        help="Filename prefix (default: frame)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    try:
        extract_frames(
            video_path=args.video,
            output_dir=args.output,
            image_format=args.format,
            every_n_frames=args.every,
            quality=args.quality,
            prefix=args.prefix,
            verbose=not args.quiet,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
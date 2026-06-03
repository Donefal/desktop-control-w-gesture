"""
extract_frames.py — Extract every frame of a video as an image file.
Supports single video or batch processing of a folder.

Dependencies:
    pip install opencv-python

Usage:
    python extract_frames.py <video_path> [options]
    python extract_frames.py <folder_path> --batch [options]

Examples:
    python extract_frames.py video.mp4
    python extract_frames.py videos/ --batch
    python extract_frames.py videos/ --batch --output frames/ --every 5
"""

import cv2
import os
import argparse
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

def extract_frames(
    video_path: str,
    output_dir: str = "frames",
    image_format: str = "jpg",
    every_n_frames: int = 1,
    quality: int = 95,
    prefix: str = "frame",
    verbose: bool = True,
) -> int:
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

    fmt = image_format.lower()
    if fmt == "jpg":
        ext = ".jpg"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == "png":
        ext = ".png"
        compression = max(0, min(9, 9 - round(quality / 100 * 9)))
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    elif fmt == "webp":
        ext = ".webp"
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        raise ValueError(f"Unsupported format '{image_format}'. Use jpg, png, or webp.")

    pad = len(str(total_frames))
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


def extract_batch(
    folder_path: str,
    output_root: str = "frames",
    image_format: str = "jpg",
    every_n_frames: int = 1,
    quality: int = 95,
    verbose: bool = True,
):
    folder_path = Path(folder_path)
    output_root = Path(output_root)

    videos = [f for f in folder_path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]

    if not videos:
        print(f"No video files found in '{folder_path}'")
        return

    print(f"Found {len(videos)} video(s) in '{folder_path}'")
    print("=" * 40)

    total_saved = 0
    for i, video in enumerate(videos, 1):
        # Save to output_root/<video_stem>/ e.g. frames/my_video/
        output_dir = output_root / video.stem
        print(f"\n[{i}/{len(videos)}] Processing: {video.name}")
        saved = extract_frames(
            video_path=str(video),
            output_dir=str(output_dir),
            image_format=image_format,
            every_n_frames=every_n_frames,
            quality=quality,
            prefix="frame",
            verbose=verbose,
        )
        total_saved += saved

    print("=" * 40)
    print(f"Batch complete. {total_saved} total frame(s) saved to '{output_root}/'.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file or a folder of videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", help="Path to a video file or a folder of videos")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Process all videos in a folder")
    parser.add_argument("--output", "-o", default="frames",
                        help="Output root directory (default: frames/)")
    parser.add_argument("--format", "-f", default="jpg",
                        choices=["jpg", "png", "webp"],
                        help="Image format (default: jpg)")
    parser.add_argument("--every", "-e", type=int, default=1, metavar="N",
                        help="Save every Nth frame (default: 1)")
    parser.add_argument("--quality", "-q", type=int, default=95, metavar="Q",
                        help="JPEG/WebP quality 1-100 (default: 95)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    try:
        if args.batch:
            extract_batch(
                folder_path=args.path,
                output_root=args.output,
                image_format=args.format,
                every_n_frames=args.every,
                quality=args.quality,
                verbose=not args.quiet,
            )
        else:
            extract_frames(
                video_path=args.path,
                output_dir=args.output,
                image_format=args.format,
                every_n_frames=args.every,
                quality=args.quality,
                verbose=not args.quiet,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
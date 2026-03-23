#!/usr/bin/env python3
"""Batch process dashboard webcam videos through openpilot's driving model.

Usage:
  python -m openpilot.tools.batch_process.run \
    --input-dir /mnt/nas/dmstagingarea01/DAC160/Recordings/ \
    --camera-config tools/batch_process/cameras/dac160.json \
    --output-dir /tmp/openpilot_batch_output \
    --formats json,csv,video
"""

import argparse
import os
import sys
import time
from pathlib import Path
from glob import glob

# Parse --device early so DEV is set before model imports
def _early_parse_device():
  for i, arg in enumerate(sys.argv):
    if arg == '--device' and i + 1 < len(sys.argv):
      return sys.argv[i + 1].upper()
    if arg.startswith('--device='):
      return arg.split('=', 1)[1].upper()
  return None

_device = _early_parse_device()
if _device:
  os.environ['DEV'] = _device

# Use ZMQ messaging backend (required for offline processing)
os.environ.setdefault('ZMQ', '1')

from openpilot.tools.batch_process.camera_config import ExternalCameraConfig
from openpilot.tools.batch_process.gva_reader import load_speed_source
from openpilot.tools.batch_process.model_runner import ModelRunner
from openpilot.tools.batch_process.outputs import OutputCollector
from openpilot.tools.batch_process.annotator import write_annotated_video
from openpilot.tools.batch_process.preprocessor import FramePreprocessor


def discover_videos(input_dir: Path, pattern: str) -> list[Path]:
  """Find all video files matching the glob pattern under input_dir."""
  matches = sorted(input_dir.glob(pattern))
  return matches


def process_single_video(video_path: Path, cam_config: ExternalCameraConfig,
                         output_dir: Path, formats: set[str],
                         mount_rpy: list[float], is_rhd: bool,
                         default_speed_kmh: float,
                         max_frames: int = 0) -> None:
  """Process a single video through the model and save outputs."""
  recording_dir = video_path.parent
  recording_name = recording_dir.name

  out_dir = output_dir / recording_name
  out_dir.mkdir(parents=True, exist_ok=True)

  print(f"\n{'='*70}")
  print(f"Processing: {recording_name}")
  print(f"  Video: {video_path}")
  print(f"  Output: {out_dir}")

  # Load speed source
  speed_source = load_speed_source(recording_dir, default_speed_kmh)
  print(f"  Speed source: {type(speed_source).__name__}")

  # Initialize model runner
  runner = ModelRunner(cam_config, mount_rpy=mount_rpy, is_rhd=is_rhd)

  # Collect outputs
  collector = OutputCollector()
  start_time = time.time()
  frame_count = 0
  model_frame_count = 0

  try:
    for result in runner.process_video(video_path, speed_source):
      frame_count += 1
      if result["model_output"] is not None:
        collector.add_frame(
          result["frame_id"],
          result["elapsed_time"],
          result["v_ego"],
          result["model_output"],
        )
        model_frame_count += 1
      if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"  Frame {frame_count} ({fps:.1f} fps, {model_frame_count} model evals)", end='\r', flush=True)
      if max_frames > 0 and frame_count >= max_frames:
        break
  except Exception as e:
    print(f"\n  ERROR processing {video_path}: {e}")
    import traceback
    traceback.print_exc()
    return
  finally:
    runner.close()

  elapsed = time.time() - start_time
  print(f"\n  Processed {frame_count} frames ({model_frame_count} model evals) in {elapsed:.1f}s "
        f"({frame_count/elapsed:.1f} fps)")

  # Save outputs
  if "json" in formats:
    json_path = out_dir / "model_outputs.json"
    collector.save_json(json_path)
    print(f"  Saved JSON: {json_path}")

  if "csv" in formats:
    csv_path = out_dir / "model_outputs.csv"
    collector.save_csv(csv_path)
    print(f"  Saved CSV: {csv_path}")

  if "video" in formats and collector.frames:
    video_out = out_dir / "annotated.mp4"
    # Re-create preprocessor for annotation pass
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    preprocessor = FramePreprocessor(cam_config, src_w, src_h)
    print(f"  Writing annotated video: {video_out}")
    write_annotated_video(
      video_path, collector.frames, video_out,
      preprocessor.K_target, mount_rpy, preprocessor,
      max_frames=max_frames if max_frames > 0 else None
    )
    print(f"  Saved annotated video: {video_out}")


def main():
  parser = argparse.ArgumentParser(
    description="Batch process dashboard videos through openpilot's driving model"
  )
  parser.add_argument("--input-dir", type=str, required=True,
                      help="Path to recordings directory")
  parser.add_argument("--camera-config", type=str, required=True,
                      help="Path to camera config JSON file")
  parser.add_argument("--output-dir", type=str, required=True,
                      help="Path to output directory")
  parser.add_argument("--formats", type=str, default="json,csv",
                      help="Comma-separated output formats: json,csv,video (default: json,csv)")
  parser.add_argument("--glob-pattern", type=str, default="*/Cam001.avi",
                      help="Glob pattern for video files (default: */Cam001.avi)")
  parser.add_argument("--mount-rpy", type=str, default="0,0,0",
                      help="Camera mount roll,pitch,yaw in degrees (default: 0,0,0)")
  parser.add_argument("--traffic-convention", type=str, default="lhd",
                      choices=["lhd", "rhd"],
                      help="Traffic convention: lhd or rhd (default: lhd)")
  parser.add_argument("--default-speed", type=float, default=0.0,
                      help="Default speed in km/h when no GVA CSV available (default: 0)")
  parser.add_argument("--max-videos", type=int, default=0,
                      help="Max number of videos to process (0 = all)")
  parser.add_argument("--max-frames", type=int, default=0,
                      help="Max frames to process per video (0 = all)")
  parser.add_argument("--device", type=str, default=None,
                      choices=["cpu", "cuda", "CPU", "CUDA", "AMD", "amd", "NV", "nv", "CL", "cl", "METAL", "metal"],
                      help="Tinygrad device for model inference (default: CPU). "
                           "Requires matching compiled pickles (*_tinygrad_{device}.pkl)")

  args = parser.parse_args()

  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  cam_config = ExternalCameraConfig.from_json(args.camera_config)
  formats = set(args.formats.split(","))
  mount_rpy = [float(x) for x in args.mount_rpy.split(",")]
  is_rhd = args.traffic_convention == "rhd"

  device = os.environ.get('DEV', 'CPU')
  print(f"Camera: {cam_config.name} ({cam_config.image_width}x{cam_config.image_height})")
  print(f"Device: {device}")
  print(f"Mount RPY: {mount_rpy}")
  print(f"Output formats: {formats}")

  # Discover videos
  videos = discover_videos(input_dir, args.glob_pattern)
  if args.max_videos > 0:
    videos = videos[:args.max_videos]

  if not videos:
    print(f"No videos found matching '{args.glob_pattern}' in {input_dir}")
    sys.exit(1)

  print(f"Found {len(videos)} videos to process")

  # Process each video
  total_start = time.time()
  for i, video_path in enumerate(videos):
    print(f"\n[{i+1}/{len(videos)}]", end="")
    process_single_video(
      video_path, cam_config, output_dir, formats,
      mount_rpy, is_rhd, args.default_speed,
      max_frames=args.max_frames
    )

  total_elapsed = time.time() - total_start
  print(f"\n{'='*70}")
  print(f"Batch complete: {len(videos)} videos processed in {total_elapsed:.0f}s")


if __name__ == "__main__":
  main()

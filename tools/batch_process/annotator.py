"""Video annotator: overlays model outputs (lane lines, path, leads) on video frames."""

import cv2
import numpy as np
from pathlib import Path

from openpilot.common.transformations.camera import get_view_frame_from_calib_frame, view_frame_from_device_frame
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan


def calib_point_to_pixel(x, y, z, K, calib_transform):
  """Project a 3D point in calibration frame to 2D pixel coordinates.

  Args:
    x, y, z: 3D point in calibration frame (x=forward, y=left, z=up)
    K: 3x3 camera intrinsic matrix (for the 1928x1208 output)
    calib_transform: 3x4 view_from_calib matrix

  Returns:
    (px, py) pixel coordinates, or None if behind camera
  """
  pt_calib = np.array([x, y, z, 1.0])
  pt_view = calib_transform @ pt_calib
  if pt_view[2] <= 0:
    return None
  pt_img = K @ pt_view[:3]
  px = pt_img[0] / pt_img[2]
  py = pt_img[1] / pt_img[2]
  return (int(round(px)), int(round(py)))


class VideoAnnotator:
  """Overlays model outputs on preprocessed video frames (1928x1208)."""

  def __init__(self, K_target: np.ndarray, mount_rpy: list[float] | None = None,
               output_width: int = 1928, output_height: int = 1208):
    self.K = K_target
    rpy = mount_rpy or [0.0, 0.0, 0.0]
    self.calib_transform = get_view_frame_from_calib_frame(rpy[0], rpy[1], rpy[2], 1.22)
    self.output_size = (output_width, output_height)

  def _project_path(self, positions, color, frame, thickness=3):
    """Draw a path (list of [x,y,z] positions) on the frame."""
    pts = []
    for pos in positions:
      pix = calib_point_to_pixel(pos[0], pos[1], pos[2], self.K, self.calib_transform)
      if pix is not None:
        w, h = self.output_size
        if 0 <= pix[0] < w and 0 <= pix[1] < h:
          pts.append(pix)
    if len(pts) >= 2:
      pts_arr = np.array(pts, dtype=np.int32)
      cv2.polylines(frame, [pts_arr], isClosed=False, color=color, thickness=thickness)

  def _project_lane_line(self, x_idxs, y_vals, z_vals, color, frame, thickness=2):
    """Draw a lane line given x (forward), y (lateral), z (height) arrays."""
    pts = []
    for x, y, z in zip(x_idxs, y_vals, z_vals):
      if x < 1.0:
        continue
      pix = calib_point_to_pixel(x, y, z, self.K, self.calib_transform)
      if pix is not None:
        w, h = self.output_size
        if 0 <= pix[0] < w and 0 <= pix[1] < h:
          pts.append(pix)
    if len(pts) >= 2:
      pts_arr = np.array(pts, dtype=np.int32)
      cv2.polylines(frame, [pts_arr], isClosed=False, color=color, thickness=thickness)

  def annotate(self, frame_bgr: np.ndarray, frame_data: dict) -> np.ndarray:
    """Draw model outputs on a single frame. Returns annotated BGR frame.

    Args:
      frame_bgr: BGR image at output resolution (1928x1208)
      frame_data: dict from OutputCollector.extract_frame_data()
    """
    out = frame_bgr.copy()

    # Lane lines (green for inner, yellow for outer)
    ll_colors = [
      (0, 200, 200),   # outer left - yellow
      (0, 255, 0),     # inner left - green
      (0, 255, 0),     # inner right - green
      (0, 200, 200),   # outer right - yellow
    ]
    for i, ll in enumerate(frame_data.get("lane_lines", [])):
      probs = frame_data.get("lane_line_probs", [0]*4)
      alpha = probs[i] if i < len(probs) else 0.5
      if alpha < 0.1:
        continue
      self._project_lane_line(ll["x"], ll["y"], ll["z"], ll_colors[i], out)

    # Road edges (red)
    for re in frame_data.get("road_edges", []):
      self._project_lane_line(re["x"], re["y"], re["z"], (0, 0, 200), out)

    # Driving path (blue)
    plan = frame_data.get("plan", {})
    positions = plan.get("position", [])
    if positions:
      self._project_path(positions, (255, 150, 0), out, thickness=4)

    # Lead vehicle indicator
    leads = frame_data.get("leads", [])
    if leads and leads[0].get("prob", 0) > 0.3:
      xyva = leads[0].get("xyva", [[]])
      if xyva and len(xyva[0]) >= 2:
        x, y = xyva[0][0], xyva[0][1]
        pix = calib_point_to_pixel(x, y, 0.0, self.K, self.calib_transform)
        if pix is not None:
          # Draw a small rectangle indicating lead
          sz = max(10, int(600 / max(x, 5)))
          cv2.rectangle(out, (pix[0]-sz, pix[1]-sz), (pix[0]+sz, pix[1]+sz), (0, 255, 255), 2)
          cv2.putText(out, f"{x:.0f}m", (pix[0]-sz, pix[1]-sz-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Speed overlay
    v_ego = frame_data.get("v_ego", 0)
    cv2.putText(out, f"{v_ego*3.6:.0f} km/h", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return out


def write_annotated_video(video_in: str | Path, frames_data: list[dict],
                          output_path: str | Path, K_target: np.ndarray,
                          mount_rpy: list[float] | None = None,
                          preprocessor=None,
                          max_frames: int | None = None):
  """Write an annotated video from the input video + collected frame data.

  Args:
    video_in: Path to original input video
    frames_data: List of frame dicts from OutputCollector
    output_path: Where to write the annotated video
    K_target: 3x3 target camera matrix
    mount_rpy: Camera mount roll/pitch/yaw
    preprocessor: FramePreprocessor instance for remapping frames
    max_frames: If set, only write this many frames (useful for partial processing)
  """
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  annotator = VideoAnnotator(K_target, mount_rpy)

  cap = cv2.VideoCapture(str(video_in))
  fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  writer = cv2.VideoWriter(str(output_path), fourcc, fps, (1928, 1208))

  # Build a lookup from frame_id to frame_data
  data_by_id = {f["frame_id"]: f for f in frames_data}

  frame_id = 0
  last_data = None
  while True:
    if max_frames is not None and frame_id >= max_frames:
      break
    ret, frame_bgr = cap.read()
    if not ret:
      break

    # Remap to target resolution
    if preprocessor is not None:
      frame_remapped = preprocessor.process_bgr(frame_bgr)
    else:
      frame_remapped = cv2.resize(frame_bgr, (1928, 1208))

    # Use this frame's data, or last known data for prepare_only frames
    data = data_by_id.get(frame_id, last_data)
    if data is not None:
      last_data = data
      frame_annotated = annotator.annotate(frame_remapped, data)
    else:
      frame_annotated = frame_remapped

    writer.write(frame_annotated)
    frame_id += 1

  cap.release()
  writer.release()

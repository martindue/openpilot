import av
import cv2
import numpy as np

from openpilot.tools.batch_process.camera_config import ExternalCameraConfig

# openpilot native road camera resolution
TARGET_W, TARGET_H = 1928, 1208


class FramePreprocessor:
  """Undistorts and remaps frames from an external camera to openpilot's native resolution.

  The remap tables are computed once at init time. Per-frame cost is a single cv2.remap().
  The output has a centered principal point, matching openpilot's CameraConfig assumption.
  """

  def __init__(self, cam_config: ExternalCameraConfig, source_width: int, source_height: int,
               target_width: int = TARGET_W, target_height: int = TARGET_H):
    # Scale intrinsics from reference resolution to actual source video resolution
    scaled = cam_config.scale_to(source_width, source_height)
    K_source = scaled.camera_matrix
    dist = scaled.dist_coeffs

    # Target focal length: preserve horizontal FOV
    self.focal_length = scaled.focal_length_x * (target_width / source_width)

    # Target camera matrix: centered principal point (openpilot convention)
    self.K_target = np.array([
      [self.focal_length, 0.0, target_width / 2.0],
      [0.0, self.focal_length, target_height / 2.0],
      [0.0, 0.0, 1.0],
    ])

    self.target_size = (target_width, target_height)

    # Precompute remap tables (undistort + resize + re-center PP in one step)
    self.map1, self.map2 = cv2.initUndistortRectifyMap(
      K_source, dist, None, self.K_target, self.target_size, cv2.CV_16SC2
    )

  def process_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
    """Remap a BGR frame to the target resolution. Returns BGR."""
    return cv2.remap(frame_bgr, self.map1, self.map2, cv2.INTER_LINEAR)

  def bgr_to_nv12(self, frame_bgr: np.ndarray) -> bytes:
    """Convert a BGR frame (at target resolution) to NV12 bytes."""
    frame = av.VideoFrame.from_ndarray(frame_bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray().data.tobytes()

  def process(self, frame_bgr: np.ndarray) -> bytes:
    """Full pipeline: remap + BGR→NV12. Returns NV12 bytes."""
    remapped = self.process_bgr(frame_bgr)
    return self.bgr_to_nv12(remapped)

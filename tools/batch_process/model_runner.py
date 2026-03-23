"""Offline model runner: feeds frames through openpilot's ModelState without running the full daemon."""

import os
import time
import numpy as np
import cv2

from pathlib import Path

# Save user's device choice before imports (modeld.py unconditionally sets DEV=CPU on non-TICI)
_user_dev = os.environ.get('DEV')
os.environ.setdefault('DEV', 'CPU')
os.environ.setdefault('ZMQ', '1')

from msgq.visionipc import VisionIpcClient, VisionStreamType

from openpilot.common.transformations.camera import CameraConfig, DeviceCameraConfig, DEVICE_CAMERAS, _NoneCameraConfig
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.modeld import ModelState
from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext
from openpilot.selfdrive.modeld.constants import ModelConstants

# Restore user's device choice after modeld import overwrote it
if _user_dev is not None:
  os.environ['DEV'] = _user_dev

from openpilot.selfdrive.modeld import modeld as _modeld_module
from openpilot.tools.batch_process.camera_config import ExternalCameraConfig
from openpilot.tools.batch_process.preprocessor import FramePreprocessor
from openpilot.tools.batch_process.feeder import OpenpilotFeeder
from openpilot.tools.batch_process.gva_reader import SpeedSource, ConstantSpeed


def _select_device_pickles() -> str:
  """Patch modeld's PKL paths to use device-specific pickles if available.

  Looks for driving_{vision,policy}_tinygrad_{device}.pkl (e.g. _cuda.pkl, _cpu.pkl).
  Falls back to the default _tinygrad.pkl if no device-specific file exists.
  Returns the device name used.
  """
  dev = os.environ.get('DEV', 'CPU').lower()
  models_dir = Path(__file__).resolve().parents[2] / 'selfdrive' / 'modeld' / 'models'

  vision_dev = models_dir / f'driving_vision_tinygrad_{dev}.pkl'
  policy_dev = models_dir / f'driving_policy_tinygrad_{dev}.pkl'

  if vision_dev.exists() and policy_dev.exists():
    _modeld_module.VISION_PKL_PATH = vision_dev
    _modeld_module.POLICY_PKL_PATH = policy_dev
    return dev.upper()
  else:
    # Fall back to defaults
    return os.environ.get('DEV', 'CPU')


class ModelRunner:
  """Processes video frames through openpilot's driving model offline.

  Usage:
    runner = ModelRunner(cam_config, is_rhd=False)
    for result in runner.process_video("/path/to/video.avi", speed_source):
      # result is a dict with frame_id, elapsed_time, model_output
      pass
    runner.close()
  """

  def __init__(self, cam_config: ExternalCameraConfig,
               mount_rpy: list[float] | None = None,
               is_rhd: bool = False):
    self.cam_config = cam_config
    self.mount_rpy = mount_rpy or [0.0, 0.0, 0.0]
    self.is_rhd = is_rhd
    self._preprocessor = None  # lazily initialized per video resolution

  def _register_camera(self, focal_length: float) -> None:
    """Monkey-patch DEVICE_CAMERAS so modeld uses our custom intrinsics."""
    custom_fcam = CameraConfig(1928, 1208, focal_length)
    custom_config = DeviceCameraConfig(
      fcam=custom_fcam,
      dcam=_NoneCameraConfig(),
      ecam=custom_fcam,  # use same for ecam so wide-camera path also works
    )
    DEVICE_CAMERAS[("pc", "unknown")] = custom_config
    DEVICE_CAMERAS[("pc", "")] = custom_config

  def _init_preprocessor(self, source_width: int, source_height: int) -> None:
    """Initialize (or re-initialize) the frame preprocessor for a given source resolution."""
    self._preprocessor = FramePreprocessor(self.cam_config, source_width, source_height)
    self._register_camera(self._preprocessor.focal_length)

    # Compute the warp matrix (camera intrinsics → model input space)
    intrinsics = self._preprocessor.K_target
    device_from_calib = np.array(self.mount_rpy, dtype=np.float32)
    self._warp_main = get_warp_matrix(device_from_calib, intrinsics, bigmodel_frame=False).astype(np.float32)
    self._warp_extra = get_warp_matrix(device_from_calib, intrinsics, bigmodel_frame=True).astype(np.float32)

  def process_video(self, video_path: str | Path,
                    speed_source: SpeedSource | None = None,
                    progress_cb=None):
    """Generator that yields per-frame model outputs for a video file.

    Args:
      video_path: Path to input video (e.g. Cam001.avi)
      speed_source: SpeedSource providing v_ego per frame time. Defaults to 0 m/s.
      progress_cb: Optional callback(frame_id, total_frames) for progress reporting.

    Yields:
      dict with keys: frame_id, elapsed_time, v_ego, model_output (or None for prepare-only frames)
    """
    if speed_source is None:
      speed_source = ConstantSpeed(0.0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise FileNotFoundError(f"Cannot open video: {video_path}")

    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Init preprocessor for this video's resolution
    self._init_preprocessor(source_w, source_h)

    # Select device-specific pickles and init model
    device_used = _select_device_pickles()
    cl_context = CLContext()
    model = ModelState(cl_context)
    print(f"  Model device: {device_used} (DEV={os.environ.get('DEV', 'CPU')})")

    # Init feeder (VisionIPC server)
    feeder = OpenpilotFeeder()

    # Publish static messages and calibration
    feeder.publish_static_messages(is_rhd=self.is_rhd)
    feeder.publish_calibration(self.mount_rpy)

    # Connect VisionIPC client (how model receives frames)
    vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True, cl_context)
    while not vipc_client.connect(False):
      time.sleep(0.01)

    # Traffic convention
    traffic_convention = np.zeros(2, dtype=np.float32)
    traffic_convention[int(self.is_rhd)] = 1

    frame_id = 0
    try:
      while True:
        ret, frame_bgr = cap.read()
        if not ret:
          break

        elapsed = frame_id / fps
        v_ego = speed_source.get_speed(elapsed)

        # Preprocess: undistort + remap + BGR→NV12
        yuv_bytes = self._preprocessor.process(frame_bgr)

        # Send frame via VisionIPC
        feeder.send_frame(yuv_bytes, frame_id, v_ego=v_ego)

        # Re-publish calibration periodically so modeld's SubMaster sees it as "updated"
        if frame_id % 100 == 0:
          feeder.publish_calibration(self.mount_rpy)
          feeder.publish_static_messages(is_rhd=self.is_rhd)

        # Receive frame via VisionIPC client
        buf = vipc_client.recv()
        if buf is None:
          frame_id += 1
          continue

        # Build model inputs
        bufs = {name: buf for name in model.vision_input_names}
        transforms = {
          name: self._warp_extra if 'big' in name else self._warp_main
          for name in model.vision_input_names
        }
        inputs = {
          'desire_pulse': np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32),
          'traffic_convention': traffic_convention,
        }

        # Run model — always do full inference since we feed every frame sequentially
        prepare_only = False
        model_output = model.run(bufs, transforms, inputs, prepare_only)

        yield {
          "frame_id": frame_id,
          "elapsed_time": elapsed,
          "v_ego": v_ego,
          "model_output": model_output,  # None for prepare_only frames
        }

        if progress_cb:
          progress_cb(frame_id, total_frames)

        frame_id += 1
    finally:
      cap.release()
      feeder.close()

  def close(self):
    pass

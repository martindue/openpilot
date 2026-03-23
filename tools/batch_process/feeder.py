import numpy as np

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging, log

from openpilot.tools.batch_process.preprocessor import TARGET_W, TARGET_H


class OpenpilotFeeder:
  """Publishes video frames and supporting cereal messages for offline model inference.

  Creates a VisionIPC server (mimicking camerad) and publishes the minimum set of
  cereal messages that modeld's SubMaster expects.
  """

  def __init__(self, num_buffers: int = 20):
    self.pm = messaging.PubMaster([
      "roadCameraState",
      "wideRoadCameraState",
      "deviceState",
      "carState",
      "liveCalibration",
      "driverMonitoringState",
      "carControl",
      "liveDelay",
    ])

    self.vipc_server = VisionIpcServer("camerad")
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, num_buffers, TARGET_W, TARGET_H)
    self.vipc_server.start_listener()

  def send_frame(self, yuv_bytes: bytes, frame_id: int, v_ego: float = 0.0) -> None:
    """Send a single NV12 frame via VisionIPC and publish matching camera state."""
    eof_ns = int(frame_id * 0.05 * 1e9)  # 20 Hz frame timing
    self.vipc_server.send(VisionStreamType.VISION_STREAM_ROAD, yuv_bytes, frame_id, eof_ns, eof_ns)

    # roadCameraState
    dat = messaging.new_message("roadCameraState", valid=True)
    dat.roadCameraState.frameId = frame_id
    dat.roadCameraState.sensor = "unknown"
    dat.roadCameraState.timestampSof = eof_ns
    dat.roadCameraState.timestampEof = eof_ns
    self.pm.send("roadCameraState", dat)

    # carState with per-frame speed
    cs = messaging.new_message("carState", valid=True)
    cs.carState.vEgo = v_ego
    self.pm.send("carState", cs)

  def publish_calibration(self, rpy: list[float] | None = None) -> None:
    """Publish a liveCalibration message (call once before processing)."""
    if rpy is None:
      rpy = [0.0, 0.0, 0.0]
    dat = messaging.new_message("liveCalibration", valid=True)
    dat.liveCalibration.calStatus = log.LiveCalibrationData.Status.calibrated
    dat.liveCalibration.calPerc = 100
    dat.liveCalibration.validBlocks = 50
    dat.liveCalibration.rpyCalib = rpy
    dat.liveCalibration.rpyCalibSpread = [0.0, 0.0, 0.0]
    dat.liveCalibration.wideFromDeviceEuler = [0.0, 0.0, 0.0]
    dat.liveCalibration.height = [1.22]
    self.pm.send("liveCalibration", dat)

  def publish_static_messages(self, is_rhd: bool = False) -> None:
    """Publish static device/driver/control messages (call once before processing)."""
    # deviceState
    ds = messaging.new_message("deviceState", valid=True)
    ds.deviceState.deviceType = log.InitData.DeviceType.pc
    self.pm.send("deviceState", ds)

    # driverMonitoringState
    dm = messaging.new_message("driverMonitoringState", valid=True)
    dm.driverMonitoringState.isRHD = is_rhd
    self.pm.send("driverMonitoringState", dm)

    # carControl
    cc = messaging.new_message("carControl", valid=True)
    cc.carControl.latActive = False
    self.pm.send("carControl", cc)

    # liveDelay
    ld = messaging.new_message("liveDelay", valid=True)
    ld.liveDelay.lateralDelay = 0.0
    self.pm.send("liveDelay", ld)

  def close(self) -> None:
    """Clean up VisionIPC server."""
    # VisionIpcServer doesn't have an explicit close, but we can del it
    del self.vipc_server

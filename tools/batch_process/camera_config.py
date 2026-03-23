import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExternalCameraConfig:
  """Camera intrinsics and distortion for an external camera.

  The intrinsics are specified at a reference resolution (e.g. 1080p).
  They can be scaled to the actual source video resolution via scale_to().
  """
  image_width: int
  image_height: int
  focal_length_x: float
  focal_length_y: float
  principal_point_x: float
  principal_point_y: float
  distortion_r2: float = 0.0
  distortion_r4: float = 0.0
  distortion_r6: float = 0.0
  distortion_t1: float = 0.0
  distortion_t2: float = 0.0
  name: str = ""

  @classmethod
  def from_json(cls, path: str | Path) -> "ExternalCameraConfig":
    with open(path) as f:
      data = json.load(f)
    intr = data["intrinsics"]
    return cls(
      image_width=data["imageWidth"],
      image_height=data["imageHeight"],
      focal_length_x=intr["focalLengthX"],
      focal_length_y=intr["focalLengthY"],
      principal_point_x=intr["principalPointX"],
      principal_point_y=intr["principalPointY"],
      distortion_r2=intr.get("distortionR2", 0.0),
      distortion_r4=intr.get("distortionR4", 0.0),
      distortion_r6=intr.get("distortionR6", 0.0),
      distortion_t1=intr.get("distortionT1", 0.0),
      distortion_t2=intr.get("distortionT2", 0.0),
      name=data.get("name", ""),
    )

  def scale_to(self, target_width: int, target_height: int) -> "ExternalCameraConfig":
    """Return a new config with intrinsics scaled to a different resolution."""
    sx = target_width / self.image_width
    sy = target_height / self.image_height
    return ExternalCameraConfig(
      image_width=target_width,
      image_height=target_height,
      focal_length_x=self.focal_length_x * sx,
      focal_length_y=self.focal_length_y * sy,
      principal_point_x=self.principal_point_x * sx,
      principal_point_y=self.principal_point_y * sy,
      distortion_r2=self.distortion_r2,
      distortion_r4=self.distortion_r4,
      distortion_r6=self.distortion_r6,
      distortion_t1=self.distortion_t1,
      distortion_t2=self.distortion_t2,
      name=self.name,
    )

  @property
  def camera_matrix(self) -> np.ndarray:
    """3x3 camera matrix K."""
    return np.array([
      [self.focal_length_x, 0.0, self.principal_point_x],
      [0.0, self.focal_length_y, self.principal_point_y],
      [0.0, 0.0, 1.0],
    ])

  @property
  def dist_coeffs(self) -> np.ndarray:
    """OpenCV distortion coefficients [k1, k2, p1, p2, k3]."""
    return np.array([
      self.distortion_r2,
      self.distortion_r4,
      self.distortion_t1,
      self.distortion_t2,
      self.distortion_r6,
    ])

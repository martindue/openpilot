from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Protocol


class SpeedSource(Protocol):
  def get_speed(self, elapsed_seconds: float) -> float:
    """Return vehicle speed in m/s at the given elapsed time from start of recording."""
    ...


class ConstantSpeed:
  """Fallback speed source returning a fixed value."""

  def __init__(self, speed_ms: float = 0.0):
    self._speed = speed_ms

  def get_speed(self, elapsed_seconds: float) -> float:
    return self._speed


class GvaReader:
  """Reads a SmartEye *_gva.csv sensor log and provides interpolated speed.

  The CSV has ~100 Hz rows with columns including:
    - FILETIME_100ns_UTC: Windows FILETIME timestamp (100 ns ticks since 1601-01-01)
    - Speed (kph): GPS-derived speed in km/h
    - AP can.speed: CAN bus speed in km/h (fallback)
    - Roll, Pitch, Yaw: orientation (deg)
    - Acceleration X/Y/Z (g), Angular velocity X/Y/Z (deg/s)
  """

  def __init__(self, csv_path: str | Path):
    self.csv_path = Path(csv_path)
    usecols = [
      "FILETIME_100ns_UTC",
      "Speed (kph)",
      "AP can.speed",
      "Roll", "Pitch", "Yaw",
      "Acceleration X (g)", "Acceleration Y (g)", "Acceleration Z (g)",
      "Angular velocity X (degree/s)", "Angular velocity Y (degree/s)", "Angular velocity Z (degree/s)",
    ]
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)

    filetime = df["FILETIME_100ns_UTC"].values.astype(np.float64)
    self._elapsed = (filetime - filetime[0]) / 1e7  # seconds from start

    # Primary speed: GPS.  Fallback: CAN bus speed.
    speed_kph = df["Speed (kph)"]
    if speed_kph.isna().all() and "AP can.speed" in df.columns:
      speed_kph = df["AP can.speed"]
    self._speed_ms = speed_kph.fillna(0.0).values.astype(np.float64) / 3.6

    # Store IMU/orientation for potential future use
    self._roll = df.get("Roll", pd.Series(dtype=float)).fillna(0.0).values.astype(np.float64)
    self._pitch = df.get("Pitch", pd.Series(dtype=float)).fillna(0.0).values.astype(np.float64)
    self._yaw = df.get("Yaw", pd.Series(dtype=float)).fillna(0.0).values.astype(np.float64)

    self.duration = self._elapsed[-1] if len(self._elapsed) > 0 else 0.0

  def get_speed(self, elapsed_seconds: float) -> float:
    """Interpolated speed in m/s at the given elapsed time."""
    return float(np.interp(elapsed_seconds, self._elapsed, self._speed_ms))

  def get_orientation(self, elapsed_seconds: float) -> tuple[float, float, float]:
    """Interpolated (roll, pitch, yaw) in degrees."""
    r = float(np.interp(elapsed_seconds, self._elapsed, self._roll))
    p = float(np.interp(elapsed_seconds, self._elapsed, self._pitch))
    y = float(np.interp(elapsed_seconds, self._elapsed, self._yaw))
    return r, p, y


def load_speed_source(recording_dir: str | Path, default_speed_kmh: float = 0.0) -> SpeedSource:
  """Auto-discover *_gva.csv in a recording directory, or fall back to constant speed."""
  recording_dir = Path(recording_dir)
  gva_files = sorted(recording_dir.glob("*_gva.csv"))
  if gva_files:
    return GvaReader(gva_files[0])
  return ConstantSpeed(default_speed_kmh / 3.6)

"""Output collectors: serialize per-frame model outputs to JSON, CSV, and rlog."""

import json
import csv
import numpy as np
from pathlib import Path

from openpilot.selfdrive.modeld.constants import ModelConstants, Plan


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
      return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
      return int(obj)
    return super().default(obj)


def extract_frame_data(frame_id: int, elapsed_time: float, v_ego: float,
                       model_output: dict[str, np.ndarray]) -> dict:
  """Extract structured data from a single frame's model output."""
  out = {
    "frame_id": frame_id,
    "elapsed_time": elapsed_time,
    "v_ego": v_ego,
  }

  # Plan: best trajectory (position, velocity, acceleration)
  if 'plan' in model_output:
    plan = model_output['plan'][0]  # shape: (IDX_N, PLAN_WIDTH)
    out["plan"] = {
      "position": plan[:, Plan.POSITION].tolist(),       # (33, 3) x,y,z
      "velocity": plan[:, Plan.VELOCITY].tolist(),
      "acceleration": plan[:, Plan.ACCELERATION].tolist(),
      "orientation": plan[:, Plan.T_FROM_CURRENT_EULER].tolist(),
      "orientation_rate": plan[:, Plan.ORIENTATION_RATE].tolist(),
    }

  # Lane lines: 4 lines × IDX_N points × 2 (y, z)
  if 'lane_lines' in model_output:
    ll = model_output['lane_lines'][0]  # (4, 33, 2)
    ll_std = model_output.get('lane_lines_stds', np.zeros_like(ll))[0]
    ll_prob = model_output.get('lane_lines_prob', np.zeros(8))[0]
    out["lane_lines"] = []
    for i in range(4):
      out["lane_lines"].append({
        "y": ll[i, :, 0].tolist(),
        "z": ll[i, :, 1].tolist(),
        "x": ModelConstants.X_IDXS,
      })
    out["lane_line_probs"] = ll_prob[1::2].tolist() if len(ll_prob) >= 8 else ll_prob.tolist()
    out["lane_line_stds"] = ll_std[:, 0, 0].tolist()

  # Road edges: 2 edges × IDX_N points × 2 (y, z)
  if 'road_edges' in model_output:
    re = model_output['road_edges'][0]  # (2, 33, 2)
    re_std = model_output.get('road_edges_stds', np.zeros_like(re))[0]
    out["road_edges"] = []
    for i in range(2):
      out["road_edges"].append({
        "y": re[i, :, 0].tolist(),
        "z": re[i, :, 1].tolist(),
        "x": ModelConstants.X_IDXS,
      })
    out["road_edge_stds"] = re_std[:, 0, 0].tolist()

  # Leads
  if 'lead' in model_output:
    leads = model_output['lead'][0]  # (3, 6, 4)  3 leads × traj × (x,y,v,a)
    lead_prob = model_output.get('lead_prob', np.zeros(3))[0]
    out["leads"] = []
    for i in range(min(3, leads.shape[0])):
      out["leads"].append({
        "xyva": leads[i].tolist(),
        "prob": float(lead_prob[i]) if i < len(lead_prob) else 0.0,
      })

  # Pose (camera odometry)
  if 'pose' in model_output:
    pose = model_output['pose'][0]  # (6,) trans_x,y,z, rot_x,y,z
    out["pose"] = {
      "trans": pose[:3].tolist(),
      "rot": pose[3:].tolist(),
    }

  # Meta
  if 'meta' in model_output:
    meta = model_output['meta'][0]
    out["meta"] = {
      "engaged_prob": float(meta[0]),
    }

  if 'desire_state' in model_output:
    out["desire_state"] = model_output['desire_state'][0].reshape(-1).tolist()

  return out


class OutputCollector:
  """Accumulates per-frame model outputs and saves to various formats."""

  def __init__(self):
    self.frames: list[dict] = []

  def add_frame(self, frame_id: int, elapsed_time: float, v_ego: float,
                model_output: dict[str, np.ndarray]) -> dict:
    """Extract and store data for one frame. Returns the extracted dict."""
    data = extract_frame_data(frame_id, elapsed_time, v_ego, model_output)
    self.frames.append(data)
    return data

  def save_json(self, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
      json.dump(self.frames, f, cls=NumpyEncoder, indent=1)

  def save_csv(self, path: str | Path) -> None:
    """Save a flattened CSV with key per-frame metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
      "frame_id", "elapsed_time", "v_ego",
      # Plan: position at ~1s, ~3s, ~5s (indices 10, 18, 23 roughly)
      "plan_x_1s", "plan_y_1s", "plan_x_3s", "plan_y_3s",
      # Lane line lateral positions at ego (first x_idx)
      "ll_inner_left_y", "ll_inner_right_y",
      "ll_inner_left_prob", "ll_inner_right_prob",
      # Road edge lateral positions at ego
      "re_left_y", "re_right_y",
      # Lead vehicle
      "lead0_x", "lead0_v", "lead0_prob",
      # Pose trans
      "pose_trans_x", "pose_trans_y", "pose_trans_z",
      # Meta
      "engaged_prob",
    ]

    with open(path, 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
      writer.writeheader()

      for frame in self.frames:
        row = {
          "frame_id": frame["frame_id"],
          "elapsed_time": f"{frame['elapsed_time']:.4f}",
          "v_ego": f"{frame['v_ego']:.3f}",
        }

        # Plan positions
        plan = frame.get("plan", {})
        pos = plan.get("position", [])
        if len(pos) > 23:
          row["plan_x_1s"] = f"{pos[10][0]:.3f}"
          row["plan_y_1s"] = f"{pos[10][1]:.3f}"
          row["plan_x_3s"] = f"{pos[18][0]:.3f}"
          row["plan_y_3s"] = f"{pos[18][1]:.3f}"

        # Lane lines: index 1 = inner left, index 2 = inner right
        ll = frame.get("lane_lines", [])
        ll_probs = frame.get("lane_line_probs", [])
        if len(ll) >= 4:
          row["ll_inner_left_y"] = f"{ll[1]['y'][0]:.3f}"
          row["ll_inner_right_y"] = f"{ll[2]['y'][0]:.3f}"
        if len(ll_probs) >= 2:
          row["ll_inner_left_prob"] = f"{ll_probs[0]:.3f}"
          row["ll_inner_right_prob"] = f"{ll_probs[1]:.3f}"

        # Road edges
        re = frame.get("road_edges", [])
        if len(re) >= 2:
          row["re_left_y"] = f"{re[0]['y'][0]:.3f}"
          row["re_right_y"] = f"{re[1]['y'][0]:.3f}"

        # Leads
        leads = frame.get("leads", [])
        if leads:
          xyva = leads[0].get("xyva", [[]])
          if xyva and len(xyva[0]) >= 3:
            row["lead0_x"] = f"{xyva[0][0]:.3f}"
            row["lead0_v"] = f"{xyva[0][2]:.3f}"
          row["lead0_prob"] = f"{leads[0].get('prob', 0.0):.3f}"

        # Pose
        pose = frame.get("pose", {})
        trans = pose.get("trans", [])
        if len(trans) >= 3:
          row["pose_trans_x"] = f"{trans[0]:.6f}"
          row["pose_trans_y"] = f"{trans[1]:.6f}"
          row["pose_trans_z"] = f"{trans[2]:.6f}"

        # Meta
        meta = frame.get("meta", {})
        row["engaged_prob"] = f"{meta.get('engaged_prob', 0.0):.4f}"

        writer.writerow(row)

  def flush_json_incremental(self, path: str | Path, batch_size: int = 500) -> None:
    """For long videos: flush accumulated frames to disk periodically."""
    if len(self.frames) < batch_size:
      return
    self.save_json(path)

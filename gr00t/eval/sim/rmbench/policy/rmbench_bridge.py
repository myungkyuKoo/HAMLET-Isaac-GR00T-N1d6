"""Shared bridge: RMBench (RoboTwin) eval  <->  our GR00T-N1.6 / RLDX-1 ZMQ policy servers.

RMBench eval runs in the RMBench conda env; our models run in their own envs as ZMQ/msgpack
PolicyServers (port 5555). This module is a self-contained client (zmq + msgpack + numpy only)
wire-compatible with BOTH gr00t/policy/server_client.py and rldx/policy/server_client.py
(identical `__ndarray_class__`/`as_npy` encoding).

Obs/action mapping (see ~/workspace/RMBench/analysis.md §4.3/§4.4):
  RMBench obs -> server obs keys:
    head_camera.rgb  -> video.front_view        (decode-as-is; no BGR/RGB swap)
    left_camera.rgb  -> video.left_wrist_view
    right_camera.rgb -> video.right_wrist_view
    joint_action.vector [L_arm6, L_grip, R_arm6, R_grip] (14)
        -> state.joint_position [L_arm6, R_arm6] (12)  +  state.gripper_position [L_grip, R_grip] (2)
  server action -> RMBench take_action(qpos):
    action.joint_position (T,12)=[L_arm6,R_arm6] + action.gripper_close (T,2)=[L_grip,R_grip]
        -> per-step vector [L_arm6, L_grip, R_arm6, R_grip] (14)

Memory: each episode uses a fresh session_id; reset_memory=[True] on the first get_action of the
episode so the server clears that session's memory cache (no-op for non-memory models).
"""
import io
import uuid
from typing import Any

import msgpack
import numpy as np
import zmq


class _Msg:
    @staticmethod
    def to_bytes(d: Any) -> bytes:
        return msgpack.packb(d, default=_Msg._enc)

    @staticmethod
    def from_bytes(b: bytes) -> Any:
        return msgpack.unpackb(b, object_hook=_Msg._dec)

    @staticmethod
    def _enc(o):
        if isinstance(o, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, o, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return o

    @staticmethod
    def _dec(o):
        if isinstance(o, dict) and "__ndarray_class__" in o:
            return np.load(io.BytesIO(o["as_npy"]), allow_pickle=False)
        return o


class _Client:
    def __init__(self, host: str, port: int, timeout_ms: int = 180000):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")

    def get_action(self, observation: dict, options: dict):
        req = {"endpoint": "get_action", "data": {"observation": observation, "options": options}}
        self.socket.send(_Msg.to_bytes(req))
        msg = self.socket.recv()
        if msg == b"ERROR":
            raise RuntimeError("Policy server returned ERROR.")
        resp = _Msg.from_bytes(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Policy server error: {resp['error']}")
        return resp


def _vector_to_state(vec: np.ndarray):
    """RMBench vector [L_arm6, L_grip, R_arm6, R_grip] -> (joint12=[L_arm6,R_arm6], grip2=[L_grip,R_grip])."""
    vec = np.asarray(vec, np.float32).reshape(-1)
    jp = np.concatenate([vec[0:6], vec[7:13]]).astype(np.float32)
    grip = np.array([vec[6], vec[13]], np.float32)
    return jp, grip


def _action_to_vector(jp_row: np.ndarray, gp_row: np.ndarray) -> np.ndarray:
    """server (jp12=[L_arm6,R_arm6], gp2=[L_grip,R_grip]) -> take_action vector [L_arm6,L_grip,R_arm6,R_grip]."""
    return np.concatenate([jp_row[0:6], gp_row[0:1], jp_row[6:12], gp_row[1:2]]).astype(np.float32)


class ServerBridgeModel:
    """Implements the RMBench model contract (update_obs / get_action_chunk / reset / set_language)
    by forwarding to a remote GR00T/RLDX policy server."""

    def __init__(self, host="127.0.0.1", port=5555, video_deltas=(0,), exec_chunk=16):
        self.client = _Client(host, port)
        self.video_deltas = list(video_deltas)
        self.exec_chunk = int(exec_chunk)
        self.reset()

    def reset(self):
        self.hist_front, self.hist_left, self.hist_right = [], [], []
        self.session_id = f"rmbench_{uuid.uuid4().hex[:10]}"
        self._first = True
        self.instruction = ""
        self._cur_vec = None

    def set_language(self, instr: str):
        self.instruction = instr or ""

    @staticmethod
    def _img(rgb) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(rgb, np.uint8))

    def update_obs(self, front, left, right, vector):
        self.hist_front.append(self._img(front))
        self.hist_left.append(self._img(left))
        self.hist_right.append(self._img(right))
        self._cur_vec = np.asarray(vector, np.float32).reshape(-1)

    def _build_obs(self) -> dict:
        last = len(self.hist_front) - 1
        idxs = [max(0, last + d) for d in self.video_deltas]
        fv = np.stack([self.hist_front[i] for i in idxs])[None, ...]
        lv = np.stack([self.hist_left[i] for i in idxs])[None, ...]
        rv = np.stack([self.hist_right[i] for i in idxs])[None, ...]
        jp, grip = _vector_to_state(self._cur_vec)
        return {
            "video.front_view": fv,
            "video.left_wrist_view": lv,
            "video.right_wrist_view": rv,
            "state.joint_position": jp[None, None, :],
            "state.gripper_position": grip[None, None, :],
            "annotation.human.action.task_description": [self.instruction],
        }

    def get_action_chunk(self):
        """Returns a list of up to exec_chunk take_action vectors (14-d each)."""
        options = {"session_ids": [self.session_id], "reset_memory": [self._first]}
        self._first = False
        resp = self.client.get_action(self._build_obs(), options)
        actions = resp[0] if isinstance(resp, (list, tuple)) else resp
        jp = np.asarray(actions["action.joint_position"], np.float32)
        gp = np.asarray(actions["action.gripper_close"], np.float32)
        if jp.ndim == 3:
            jp = jp[0]
        if gp.ndim == 3:
            gp = gp[0]
        if gp.ndim == 1:
            gp = gp[:, None]
        H = min(self.exec_chunk, jp.shape[0])
        return [_action_to_vector(jp[k], gp[k]) for k in range(H)]


def encode_obs(observation: dict):
    """RMBench obs dict -> (front_rgb, left_rgb, right_rgb, joint_vector14)."""
    obs = observation["observation"]
    front = obs["head_camera"]["rgb"]
    left = obs["left_camera"]["rgb"]
    right = obs["right_camera"]["rgb"]
    vector = observation["joint_action"]["vector"]
    return front, left, right, vector


def run_eval(TASK_ENV, model: ServerBridgeModel, observation: dict):
    """Standard RMBench eval step: (first) set instruction -> update obs -> get chunk -> apply each."""
    if len(model.hist_front) == 0:
        model.set_language(TASK_ENV.get_instruction())
    front, left, right, vector = encode_obs(observation)
    model.update_obs(front, left, right, vector)
    for action in model.get_action_chunk():
        TASK_ENV.take_action(action)              # action_type='qpos' (default)
        observation = TASK_ENV.get_obs()
        front, left, right, vector = encode_obs(observation)
        model.update_obs(front, left, right, vector)

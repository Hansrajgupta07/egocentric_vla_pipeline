"""
_libero_sim.py
==============
Shared simulation core re-used by 3_run_libero.py and 8_compare_policies.py.
Do NOT run this file directly — import it from the other scripts.
"""

# Re-export everything needed by comparison scripts
from importlib import import_module
import sys, os, types

# We can't import 3_run_libero directly (name starts with digit).
# So we duplicate the shared classes/functions here cleanly.

import numpy as np
import torch
import cv2

TASK_DESCRIPTION = "Pick up the Rubik's cube and place it inside the bowl"
IMG_SIZE         = 224
MAX_STEPS        = 120
DEFAULT_CUBE     = (0.75, 0.50)
DEFAULT_BOWL     = (0.25, 0.50)


# ---------------------------------------------------------------------------
class SimulatedFrankaWorkspace:
    """Simulated Franka Panda Tabletop Manipulation Environment."""
    def __init__(self,
                 cube_pos=(0.733, 0.354, 0.05),
                 bowl_pos=(0.243, 0.352, 0.05)):
        self.cube_pos_init = np.array(cube_pos, dtype=np.float32)
        self.bowl_pos_init = np.array(bowl_pos, dtype=np.float32)
        self.reset()

    def reset(self):
        self.eef_pos       = np.array([0.82, 1.01, 0.0], dtype=np.float32)
        self.gripper_state = 0.0
        self.cube_pos      = self.cube_pos_init.copy()
        self.bowl_pos      = self.bowl_pos_init.copy()
        self.cube_held     = False
        self.step_count    = 0
        return self.get_observation()

    def get_observation(self):
        return {
            "hand_pos": self.eef_pos.copy(),
            "cube_pos": self.cube_pos[:2].copy(),
            "bowl_pos": self.bowl_pos[:2].copy(),
            "gripper":  self.gripper_state,
        }

    def get_state_vector(self):
        hx, hy, hz = self.eef_pos
        cx, cy     = self.cube_pos[:2]
        bx, by     = self.bowl_pos[:2]
        return np.array([hx, hy, hz, cx, cy, bx, by], dtype=np.float32)

    def step(self, action_pred):
        dx, dy, dz, grip_action = action_pred
        self.step_count += 1
        self.eef_pos[0] += float(dx)
        self.eef_pos[1] += float(dy)
        self.eef_pos[2] += float(dz)
        self.gripper_state = 1.0 if float(grip_action) > 0.5 else 0.0

        dist_to_cube = np.linalg.norm(self.eef_pos - self.cube_pos)
        if self.gripper_state > 0.5 and dist_to_cube < 0.08:
            self.cube_held = True
        if self.cube_held:
            self.cube_pos = self.eef_pos.copy() - np.array([0, 0, 0.03])

        dist_to_bowl = np.linalg.norm(self.cube_pos[:2] - self.bowl_pos[:2])
        if self.gripper_state < 0.5 and self.cube_held:
            self.cube_held = False
            if dist_to_bowl < 0.12:
                self.cube_pos = self.bowl_pos.copy()

        success = bool((not self.cube_held) and (
            np.linalg.norm(self.cube_pos[:2] - self.bowl_pos[:2]) < 0.08
        ))
        done = bool(success or (self.step_count >= MAX_STEPS))
        return self.get_observation(), success, done


# ---------------------------------------------------------------------------
def render_workspace_frame(eef_pos, cube_pos, bowl_pos, gripper_state,
                            cube_held=False, img_size=IMG_SIZE):
    """Top-down synthetic workspace render → 224×224 BGR uint8."""
    frame = np.ones((img_size, img_size, 3), dtype=np.uint8) * 245
    cv2.rectangle(frame, (5, 5), (img_size-5, img_size-5), (210, 190, 160), -1)
    cv2.rectangle(frame, (5, 5), (img_size-5, img_size-5), (150, 130, 110), 2)

    def ws2px(x, y):
        px = int(np.clip(x, 0, 1) * (img_size - 30) + 15)
        py = int(np.clip(y, 0, 1) * (img_size - 30) + 15)
        return (px, py)

    bp = ws2px(bowl_pos[0], bowl_pos[1])
    cv2.circle(frame, bp, 22, (20, 140, 255), -1)
    cv2.circle(frame, bp, 22, (10, 80, 200), 2)
    cv2.circle(frame, bp, 10, (30, 100, 200), 2)
    cv2.putText(frame, "1", (bp[0]-4, bp[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cp = ws2px(cube_pos[0], cube_pos[1])
    cube_col = (50, 50, 220) if not cube_held else (50, 220, 50)
    cv2.rectangle(frame, (cp[0]-12, cp[1]-12), (cp[0]+12, cp[1]+12), cube_col, -1)
    cv2.rectangle(frame, (cp[0]-12, cp[1]-12), (cp[0]+12, cp[1]+12), (0,0,0), 1)
    cv2.putText(frame, "0", (cp[0]-4, cp[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    ep = ws2px(eef_pos[0], eef_pos[1])
    eef_col = (0, 40, 200) if gripper_state > 0.5 else (0, 180, 20)
    cv2.circle(frame, ep, 14, eef_col, -1)
    cv2.circle(frame, ep, 14, (0, 0, 0), 1)
    cv2.putText(frame, "G" if gripper_state > 0.5 else "O",
                (ep[0]-5, ep[1]+5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255,255,255), 1)

    cv2.putText(frame, f"EEF ({eef_pos[0]:.2f},{eef_pos[1]:.2f})",
                (8, img_size-28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,60,60), 1)
    cv2.putText(frame, f"Grip: {'CLOSED' if gripper_state > 0.5 else 'OPEN'}",
                (8, img_size-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,60,60), 1)
    return frame


def frame_to_tensor(frame_bgr, device):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(rgb, dtype=torch.float32).permute(2,0,1) / 255.0
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
def run_rollout_mlp(env, model, verbose=True):
    obs    = env.reset()
    traj_x, traj_y, traj_z, grippers = [], [], [], []
    success = False
    for step in range(MAX_STEPS):
        state = env.get_state_vector()
        traj_x.append(state[0]); traj_y.append(state[1])
        traj_z.append(state[2]); grippers.append(obs["gripper"])

        state_t = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            raw = model(state_t).squeeze().numpy()
        dx, dy, dz = raw[0], raw[1], raw[2]
        grip_prob  = float(1 / (1 + np.exp(-raw[3])))
        obs, success, done = env.step((dx, dy, dz, grip_prob))

        if verbose and (step % 20 == 0 or done):
            print(f"  [MLP] Step {step:03d} | EEF ({state[0]:.3f},{state[1]:.3f}) "
                  f"| Grip {'CLOSED' if grip_prob > 0.5 else 'OPEN'}")
        if done:
            break
    return success, env.step_count, traj_x, traj_y, traj_z, grippers


def run_rollout_smolvla(env, policy, device, verbose=True):
    obs     = env.reset()
    traj_x, traj_y, traj_z, grippers = [], [], [], []
    success = False
    for step in range(MAX_STEPS):
        state = env.get_state_vector()
        traj_x.append(state[0]); traj_y.append(state[1])
        traj_z.append(state[2]); grippers.append(obs["gripper"])

        frame    = render_workspace_frame(state[:3], state[3:5], state[5:7],
                                          obs["gripper"], env.cube_held)
        img_t    = frame_to_tensor(frame, device)
        state_t  = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        obs_dict = {
            "observation.images.phone": img_t,
            "observation.state":        state_t,
            "task":                     [TASK_DESCRIPTION],
        }
        with torch.no_grad():
            action = policy.select_action(obs_dict).squeeze(0).cpu().numpy()

        dx, dy, dz = action[0], action[1], action[2]
        grip_prob  = float(1 / (1 + np.exp(-action[3])))
        obs, success, done = env.step((dx, dy, dz, grip_prob))

        if verbose and (step % 20 == 0 or done):
            print(f"  [VLA] Step {step:03d} | EEF ({state[0]:.3f},{state[1]:.3f}) "
                  f"| Grip {'CLOSED' if grip_prob > 0.5 else 'OPEN'}")
        if done:
            break
    return success, env.step_count, traj_x, traj_y, traj_z, grippers

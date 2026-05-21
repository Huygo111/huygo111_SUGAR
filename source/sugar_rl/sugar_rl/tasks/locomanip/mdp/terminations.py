from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from isaaclab.utils.math import quat_mul, quat_conjugate
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from sugar_rl.tasks.locomanip.mdp.commands import MotionCommand
from sugar_rl.tasks.locomanip.mdp.rewards import _get_body_indexes

from sugar_rl.tasks.locomanip.mdp.rewards import quat_error_magnitude

from isaaclab.utils.math import subtract_frame_transforms


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_obj_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.norm(command.obj_ref_pos_w - command.obj_pos_w, dim=-1)
    return error > threshold

def bad_obj_ori(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.obj_ref_quat_w, command.obj_quat_w)
    return error > threshold


def trajectory_complete(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Terminate when trajectory reaches maximum length.
    
    Note: Uses time_steps + 1 because termination_manager.compute() runs before
    command_manager.compute() which increments time_steps.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    max_timesteps = command.motion.time_step_total_permotion[command.motion_id]
    return (command.time_steps + 1) >= max_timesteps


def rollout_window_timeout(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    # Terminate when rollout window length is reached or trajectory ends.
    
    command: MotionCommand = env.command_manager.get_term(command_name)

    max_timesteps = command.motion.time_step_total_permotion[command.motion_id]

    steps_since_start = command.time_steps + 1 - command.rollout_start_timesteps
    
    window_timeout = steps_since_start >= command.cfg.rollout_window_length
    trajectory_end = (command.time_steps + 1) >= max_timesteps
    
    return window_timeout | trajectory_end




def eval_timeout(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    timeout = (command.time_steps + 1) >= command.cfg.eval_max_time
    
    return timeout

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, matrix_from_quat, subtract_frame_transforms, combine_frame_transforms
from sugar_rl.tasks.locomanip.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)

def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)

def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_obj_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.obj_pos_w - command.obj_ref_pos_w), dim=-1)
    return torch.exp(-error / std**2)

def motion_global_obj_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.obj_quat_w, command.obj_ref_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_global_obj_linear_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.obj_lin_vel_w - command.obj_ref_lin_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_obj_angular_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.obj_ang_vel_w - command.obj_ref_ang_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def obj2body_pos_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    obj2body_pos, _ = subtract_frame_transforms(
        command.robot_body_pos_w,
        command.robot_body_quat_w,
        command.obj_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.obj_quat_w[:, None, :].repeat(1, num_bodies, 1),
    )

    ref_obj2body_pos, _ = subtract_frame_transforms(
        command.body_pos_w,
        command.body_quat_w,
        command.obj_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.obj_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
    )

    error = torch.mean(torch.sum(torch.square(obj2body_pos - ref_obj2body_pos), dim=-1), dim=-1)
    return torch.exp(-error / std**2)


def obj2body_ori_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, obj2body_ori = subtract_frame_transforms(
        command.robot_body_pos_w,
        command.robot_body_quat_w,
        command.obj_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.obj_quat_w[:, None, :].repeat(1, num_bodies, 1),
    )

    _, ref_obj2body_ori = subtract_frame_transforms(
        command.body_pos_w,
        command.body_quat_w,
        command.obj_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.obj_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
    )

    error = torch.mean((quat_error_magnitude(obj2body_ori, ref_obj2body_ori) ** 2), dim=-1)
    return torch.exp(-error / std**2)



def hands_contact(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    left_hand_sensor_cfg: SceneEntityCfg, 
    right_hand_sensor_cfg: SceneEntityCfg, 
    command_name: str
) -> torch.Tensor:
    
    # Only single object is supported.
    command: MotionCommand = env.command_manager.get_term(command_name)

    left_hand_contact_sensor: ContactSensor = env.scene.sensors[left_hand_sensor_cfg.name]
    right_hand_contact_sensor: ContactSensor = env.scene.sensors[right_hand_sensor_cfg.name]
    left_hand_forces = left_hand_contact_sensor.data.force_matrix_w_history  # [num_envs, history(3), body_ids(1), filter_num(1), 3]
    right_hand_forces = right_hand_contact_sensor.data.force_matrix_w_history


    left_curr_contact_force = torch.max(
        torch.norm(left_hand_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs
    right_curr_contact_force = torch.max(
        torch.norm(right_hand_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs
    is_contact = (left_curr_contact_force > threshold) * (right_curr_contact_force  > threshold)
    contact_label = command.contact_label
    
    is_consistent = (is_contact == contact_label).float()
    
    return is_consistent

def foot_contact(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    left_foot_sensor_cfg: SceneEntityCfg, 
    right_foot_sensor_cfg: SceneEntityCfg, 
    command_name: str
) -> torch.Tensor:
    # Only single object is supported.
    command: MotionCommand = env.command_manager.get_term(command_name)

    left_foot_contact_sensor: ContactSensor = env.scene.sensors[left_foot_sensor_cfg.name]
    right_foot_contact_sensor: ContactSensor = env.scene.sensors[right_foot_sensor_cfg.name]
    left_foot_forces = left_foot_contact_sensor.data.force_matrix_w_history  # [num_envs, history(3), body_ids(1), filter_num(1), 3]
    right_foot_forces = right_foot_contact_sensor.data.force_matrix_w_history

    left_curr_contact_force = torch.max(
        torch.norm(left_foot_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs
    right_curr_contact_force = torch.max(
        torch.norm(right_foot_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs

    is_contact = (left_curr_contact_force > threshold) | (right_curr_contact_force  > threshold)
    contact_label = command.contact_label
    
    is_consistent = (is_contact == contact_label).float()
    
    return is_consistent

def sit_contact(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    left_hip_sensor_cfg: SceneEntityCfg, 
    right_hip_sensor_cfg: SceneEntityCfg, 
    pelvis_sensor_cfg: SceneEntityCfg, 
    command_name: str
) -> torch.Tensor:
    # Only single object is supported.
    command: MotionCommand = env.command_manager.get_term(command_name)

    left_hip_contact_sensor: ContactSensor = env.scene.sensors[left_hip_sensor_cfg.name]
    right_hip_contact_sensor: ContactSensor = env.scene.sensors[right_hip_sensor_cfg.name]
    pelvis_contact_sensor: ContactSensor = env.scene.sensors[pelvis_sensor_cfg.name]
    left_hip_forces = left_hip_contact_sensor.data.force_matrix_w_history  # [num_envs, history(3), body_ids(1), filter_num(1), 3]
    right_hip_forces = right_hip_contact_sensor.data.force_matrix_w_history
    pelvis_forces = pelvis_contact_sensor.data.force_matrix_w_history  

    left_curr_contact_force = torch.max(
        torch.norm(left_hip_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs
    right_curr_contact_force = torch.max(
        torch.norm(right_hip_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs
    pelvis_curr_contact_force = torch.max(
        torch.norm(pelvis_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs

    is_contact = (left_curr_contact_force > threshold) | (right_curr_contact_force  > threshold) | (pelvis_curr_contact_force > threshold)
    contact_label = command.contact_label
    
    is_consistent = (is_contact == contact_label).float()
    
    return is_consistent

def feet_air_time_min_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # [num_envs, num_feet]  True if this foot just touched ground
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    # [num_envs, num_feet]  air time of the just-finished step
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # penalty: air time shorter than threshold
    # (negative or zero)
    penalty = (last_air_time - threshold).clamp_max(0.0)

    # only apply once per step
    penalty = penalty * first_contact

    # sum over feet
    penalty = torch.sum(penalty, dim=1)

    env_ready_mask = env.episode_length_buf > 50
    
    return penalty * env_ready_mask

def motion_joint_pos_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
    return torch.exp(-error / std**2)

def motion_joint_vel_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
    return torch.exp(-error / std**2)


def avoid_feet_obj_contact(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    left_foot_sensor_cfg: SceneEntityCfg, 
    right_foot_sensor_cfg: SceneEntityCfg, 
    command_name: str
) -> torch.Tensor:
    # Only single object is supported.
    command: MotionCommand = env.command_manager.get_term(command_name)

    left_foot_contact_sensor: ContactSensor = env.scene.sensors[left_foot_sensor_cfg.name]
    right_foot_contact_sensor: ContactSensor = env.scene.sensors[right_foot_sensor_cfg.name]
    left_foot_forces = left_foot_contact_sensor.data.force_matrix_w_history  # [num_envs, history(3), body_ids(1), filter_num(1), 3]
    right_foot_forces = right_foot_contact_sensor.data.force_matrix_w_history

    left_curr_contact_force = torch.max(
        torch.norm(left_foot_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs
    right_curr_contact_force = torch.max(
        torch.norm(right_foot_forces[:, :, 0, 0, :], dim=-1),
        dim=1
    )[0]    # num_envs
    is_contact = (left_curr_contact_force > threshold) | (right_curr_contact_force  > threshold)
    contact_label = torch.zeros_like(command.contact_label).bool()
    
    is_consistent = (is_contact == contact_label).float()
    
    return is_consistent


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward
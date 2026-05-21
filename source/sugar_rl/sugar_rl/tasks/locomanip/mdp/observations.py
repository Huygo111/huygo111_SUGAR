from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv

from sugar_rl.tasks.locomanip.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )   
    return pos_b.view(env.num_envs, -1)

def obj_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

def ref_obj_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_ref_pos_w,
        command.obj_ref_quat_w,
    )   
    return pos_b.view(env.num_envs, -1)

def ref_obj_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_ref_pos_w,
        command.obj_ref_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    quat_inv_robot_anchor = command.robot_anchor_quat_w.clone()
    quat_inv_robot_anchor[:, :3] = -quat_inv_robot_anchor[:, :3]
    obj_lin_vel_b = torch.bmm(
        matrix_from_quat(quat_inv_robot_anchor),
        command.obj_lin_vel_w.unsqueeze(-1)
    ).squeeze(-1)
    return obj_lin_vel_b.view(env.num_envs, -1)


def obj_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    quat_inv_robot_anchor = command.robot_anchor_quat_w.clone()
    quat_inv_robot_anchor[:, :3] = -quat_inv_robot_anchor[:, :3]
    obj_ang_vel_b = torch.bmm(
        matrix_from_quat(quat_inv_robot_anchor),
        command.obj_ang_vel_w.unsqueeze(-1)
    ).squeeze(-1)
    return obj_ang_vel_b.view(env.num_envs, -1)


def ref_obj_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    quat_inv_robot_anchor = command.robot_anchor_quat_w.clone()
    quat_inv_robot_anchor[:, :3] = -quat_inv_robot_anchor[:, :3]
    ref_obj_lin_vel_b = torch.bmm(
        matrix_from_quat(quat_inv_robot_anchor),
        command.obj_ref_lin_vel_w.unsqueeze(-1)
    ).squeeze(-1)
    return ref_obj_lin_vel_b.view(env.num_envs, -1)


def ref_obj_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    quat_inv_robot_anchor = command.robot_anchor_quat_w.clone()
    quat_inv_robot_anchor[:, :3] = -quat_inv_robot_anchor[:, :3]
    ref_obj_ang_vel_b = torch.bmm(
        matrix_from_quat(quat_inv_robot_anchor),
        command.obj_ref_ang_vel_w.unsqueeze(-1)
    ).squeeze(-1)
    return ref_obj_ang_vel_b.view(env.num_envs, -1)



def obj_motion_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Focus on object movement direction, not target position.
    # Motion target in object space.
    pos, _ = subtract_frame_transforms(
        command.obj_pos_w,
        command.obj_quat_w,
        command.obj_ref_pos_w,
        command.obj_ref_quat_w,
    )
    return pos.view(env.num_envs, -1)

def obj_motion_ori(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Focus on object movement direction, not target position.
    # Motion target in object space.
    _, ori = subtract_frame_transforms(
        command.obj_pos_w,
        command.obj_quat_w,
        command.obj_ref_pos_w,
        command.obj_ref_quat_w,
    )   
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def joint_pos_vel_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames (joint pos + vel).
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * num_joints * 2)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos_vel_future.reshape(env.num_envs, -1)

def teacher_joint_pos_vel_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames (joint pos + vel) from teacher_motion.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * num_joints * 2)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.teacher_joint_pos_vel_future.reshape(env.num_envs, -1)

def motion_anchor_pos_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of reference anchor positions in current robot base frame.
    
    Each anchor position (including current frame) is transformed to the **current** robot base frame.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened anchor positions
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future anchor positions in world frame: (num_envs, n_frames, 3)
    future_anchor_pos_w = command.motion.body_pos_w[batch_motion_ids, future_timesteps, command.motion_anchor_body_index] + env.scene.env_origins.unsqueeze(1)
    future_anchor_quat_w = command.motion.body_quat_w[batch_motion_ids, future_timesteps, command.motion_anchor_body_index]  # (num_envs, n_frames, 4)
    
    # Get current robot anchor pose (same for all future frames)
    robot_anchor_pos_w = command.robot_anchor_pos_w  # (num_envs, 3)
    robot_anchor_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # Expand to match future frames dimension
    robot_anchor_pos_w = robot_anchor_pos_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 3)
    robot_anchor_quat_w = robot_anchor_quat_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future anchors to current robot base frame
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        robot_anchor_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
        future_anchor_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        future_anchor_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
    )
    
    # Reshape back to (num_envs, n_frames, 3) and flatten to (num_envs, n_frames * 3)
    return pos_b.reshape(env.num_envs, n_frames, 3).reshape(env.num_envs, -1)

def teacher_motion_anchor_pos_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of reference anchor positions in current robot base frame from teacher_motion.
    
    Each anchor position (including current frame) is transformed to the **current** robot base frame.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened anchor positions
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future anchor positions in world frame: (num_envs, n_frames, 3)
    future_anchor_pos_w = command.teacher_motion.body_pos_w[batch_motion_ids, future_timesteps, command.motion_anchor_body_index] + env.scene.env_origins.unsqueeze(1)
    future_anchor_quat_w = command.teacher_motion.body_quat_w[batch_motion_ids, future_timesteps, command.motion_anchor_body_index]  # (num_envs, n_frames, 4)
    
    # Get current robot anchor pose (same for all future frames)
    robot_anchor_pos_w = command.robot_anchor_pos_w  # (num_envs, 3)
    robot_anchor_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # Expand to match future frames dimension
    robot_anchor_pos_w = robot_anchor_pos_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 3)
    robot_anchor_quat_w = robot_anchor_quat_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future anchors to current robot base frame
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        robot_anchor_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
        future_anchor_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        future_anchor_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
    )
    
    # Reshape back to (num_envs, n_frames, 3) and flatten to (num_envs, n_frames * 3)
    return pos_b.reshape(env.num_envs, n_frames, 3).reshape(env.num_envs, -1)

def motion_anchor_ori_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of reference anchor orientations in current robot base frame.
    
    Each anchor orientation (including current frame) is transformed to the **current** robot base frame.
    Returns first 2 columns of rotation matrix.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 6) - flattened anchor orientations (rotation matrix first 2 cols)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future anchor positions and orientations in world frame
    future_anchor_pos_w = command.motion.body_pos_w[batch_motion_ids, future_timesteps, command.motion_anchor_body_index] + env.scene.env_origins.unsqueeze(1)  # (num_envs, n_frames, 3)
    future_anchor_quat_w = command.motion.body_quat_w[batch_motion_ids, future_timesteps, command.motion_anchor_body_index]  # (num_envs, n_frames, 4)
    
    # Get current robot anchor pose (same for all future frames)
    robot_anchor_pos_w = command.robot_anchor_pos_w  # (num_envs, 3)
    robot_anchor_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # Expand to match future frames dimension
    robot_anchor_pos_w = robot_anchor_pos_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 3)
    robot_anchor_quat_w = robot_anchor_quat_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future anchors to current robot base frame
    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        robot_anchor_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
        future_anchor_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        future_anchor_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
    )
    
    # Convert to rotation matrix and take first 2 columns
    mat = matrix_from_quat(ori_b)  # (num_envs * n_frames, 3, 3)
    mat_2cols = mat[..., :2]  # (num_envs * n_frames, 3, 2)

    # Reshape to (num_envs, n_frames, 6) and flatten to (num_envs, n_frames * 6)
    return mat_2cols.reshape(env.num_envs, n_frames, 6).reshape(env.num_envs, -1)

def teacher_motion_anchor_ori_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of reference anchor orientations in current robot base frame from teacher_motion.
    
    Each anchor orientation (including current frame) is transformed to the **current** robot base frame.
    Returns first 2 columns of rotation matrix.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 6) - flattened anchor orientations (rotation matrix first 2 cols)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future anchor positions and orientations in world frame
    future_anchor_pos_w = command.teacher_motion.body_pos_w[batch_motion_ids, future_timesteps, command.motion_anchor_body_index] + env.scene.env_origins.unsqueeze(1)  # (num_envs, n_frames, 3)
    future_anchor_quat_w = command.teacher_motion.body_quat_w[batch_motion_ids, future_timesteps, command.motion_anchor_body_index]  # (num_envs, n_frames, 4)
    
    # Get current robot anchor pose (same for all future frames)
    robot_anchor_pos_w = command.robot_anchor_pos_w  # (num_envs, 3)
    robot_anchor_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # Expand to match future frames dimension
    robot_anchor_pos_w = robot_anchor_pos_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 3)
    robot_anchor_quat_w = robot_anchor_quat_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future anchors to current robot base frame
    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        robot_anchor_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
        future_anchor_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        future_anchor_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
    )
    
    # Convert to rotation matrix and take first 2 columns
    mat = matrix_from_quat(ori_b)  # (num_envs * n_frames, 3, 3)
    mat_2cols = mat[..., :2]  # (num_envs * n_frames, 3, 2)
    
    # Reshape to (num_envs, n_frames, 6) and flatten to (num_envs, n_frames * 6)
    return mat_2cols.reshape(env.num_envs, n_frames, 6).reshape(env.num_envs, -1)

def obj_motion_pos_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of object motion positions in current object frame.
    
    Each future reference position is transformed to the **current** object frame.
    Returns where the object should move to in its own coordinate system.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened object motion positions
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future reference object positions in world frame: (num_envs, n_frames, 3)
    future_obj_ref_pos_w = command.motion.obj_pos[batch_motion_ids, future_timesteps] + env.scene.env_origins.unsqueeze(1)
    future_obj_ref_quat_w = command.motion.obj_quat[batch_motion_ids, future_timesteps]  # (num_envs, n_frames, 4)
    
    # Get current object pose (same for all future frames)
    obj_pos_w = command.obj_pos_w  # (num_envs, 3)
    obj_quat_w = command.obj_quat_w  # (num_envs, 4)
    
    # Expand to match future frames dimension
    obj_pos_w = obj_pos_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 3)
    obj_quat_w = obj_quat_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future references to current object frame
    pos_o, _ = subtract_frame_transforms(
        obj_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        obj_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
        future_obj_ref_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        future_obj_ref_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
    )
    
    # Reshape back to (num_envs, n_frames, 3) and flatten to (num_envs, n_frames * 3)
    return pos_o.reshape(env.num_envs, n_frames, 3).reshape(env.num_envs, -1)

def teacher_obj_motion_pos_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of object motion positions in current object frame from teacher_motion.
    
    Each future reference position is transformed to the **current** object frame.
    Returns where the object should move to in its own coordinate system.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened object motion positions
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future reference object positions in world frame: (num_envs, n_frames, 3)
    future_obj_ref_pos_w = command.teacher_motion.obj_pos[batch_motion_ids, future_timesteps] + env.scene.env_origins.unsqueeze(1)
    future_obj_ref_quat_w = command.teacher_motion.obj_quat[batch_motion_ids, future_timesteps]  # (num_envs, n_frames, 4)
    
    # Get current object pose (same for all future frames)
    obj_pos_w = command.obj_pos_w  # (num_envs, 3)
    obj_quat_w = command.obj_quat_w  # (num_envs, 4)
    
    # Expand to match future frames dimension
    obj_pos_w = obj_pos_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 3)
    obj_quat_w = obj_quat_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future references to current object frame
    pos_o, _ = subtract_frame_transforms(
        obj_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        obj_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
        future_obj_ref_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        future_obj_ref_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
    )
    
    # Reshape back to (num_envs, n_frames, 3) and flatten to (num_envs, n_frames * 3)
    return pos_o.reshape(env.num_envs, n_frames, 3).reshape(env.num_envs, -1)

def obj_motion_ori_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of object motion orientations in current object frame.
    
    Each future reference orientation is transformed to the **current** object frame.
    Returns how the object should rotate in its own coordinate system.
    Returns first 2 columns of rotation matrix.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 6) - flattened object motion orientations (rotation matrix first 2 cols)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future reference object positions and orientations in world frame
    future_obj_ref_pos_w = command.motion.obj_pos[batch_motion_ids, future_timesteps] + env.scene.env_origins.unsqueeze(1)  # (num_envs, n_frames, 3)
    future_obj_ref_quat_w = command.motion.obj_quat[batch_motion_ids, future_timesteps]  # (num_envs, n_frames, 4)
    
    # Get current object pose (same for all future frames)
    obj_pos_w = command.obj_pos_w  # (num_envs, 3)
    obj_quat_w = command.obj_quat_w  # (num_envs, 4)
    
    # Expand to match future frames dimension
    obj_pos_w = obj_pos_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 3)
    obj_quat_w = obj_quat_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future references to current object frame
    _, ori_o = subtract_frame_transforms(
        obj_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        obj_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
        future_obj_ref_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        future_obj_ref_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
    )
    
    # Convert to rotation matrix and take first 2 columns
    mat = matrix_from_quat(ori_o)  # (num_envs * n_frames, 3, 3)
    mat_2cols = mat[..., :2]  # (num_envs * n_frames, 3, 2)
    
    # Reshape to (num_envs, n_frames, 6) and flatten to (num_envs, n_frames * 6)
    return mat_2cols.reshape(env.num_envs, n_frames, 6).reshape(env.num_envs, -1)

def teacher_obj_motion_ori_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of object motion orientations in current object frame from teacher_motion.
    
    Each future reference orientation is transformed to the **current** object frame.
    Returns how the object should rotate in its own coordinate system.
    Returns first 2 columns of rotation matrix.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 6) - flattened object motion orientations (rotation matrix first 2 cols)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future reference object positions and orientations in world frame
    future_obj_ref_pos_w = command.teacher_motion.obj_pos[batch_motion_ids, future_timesteps] + env.scene.env_origins.unsqueeze(1)  # (num_envs, n_frames, 3)
    future_obj_ref_quat_w = command.teacher_motion.obj_quat[batch_motion_ids, future_timesteps]  # (num_envs, n_frames, 4)
    
    # Get current object pose (same for all future frames)
    obj_pos_w = command.obj_pos_w  # (num_envs, 3)
    obj_quat_w = command.obj_quat_w  # (num_envs, 4)
    
    # Expand to match future frames dimension
    obj_pos_w = obj_pos_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 3)
    obj_quat_w = obj_quat_w.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future references to current object frame
    _, ori_o = subtract_frame_transforms(
        obj_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        obj_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
        future_obj_ref_pos_w.reshape(-1, 3),  # Flatten to (num_envs * n_frames, 3)
        future_obj_ref_quat_w.reshape(-1, 4),  # Flatten to (num_envs * n_frames, 4)
    )
    
    # Convert to rotation matrix and take first 2 columns
    mat = matrix_from_quat(ori_o)  # (num_envs * n_frames, 3, 3)
    mat_2cols = mat[..., :2]  # (num_envs * n_frames, 3, 2)
    
    # Reshape to (num_envs, n_frames, 6) and flatten to (num_envs, n_frames * 6)
    return mat_2cols.reshape(env.num_envs, n_frames, 6).reshape(env.num_envs, -1)

def ref_obj_lin_vel_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of reference object linear velocities in current robot base frame.
    
    Each future reference velocity is transformed to the **current** robot base frame.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened object linear velocities
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future reference object linear velocities in world frame: (num_envs, n_frames, 3)
    future_obj_ref_lin_vel_w = command.motion.obj_lin_vel[batch_motion_ids, future_timesteps]
    
    # Get current robot anchor orientation (same for all future frames)
    robot_anchor_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # Invert quaternion for coordinate transformation
    quat_inv = robot_anchor_quat_w.clone()
    quat_inv[:, :3] = -quat_inv[:, :3]
    
    # Expand to match future frames dimension
    quat_inv = quat_inv.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future velocities to current robot base frame
    # Reshape for batch matrix multiplication
    rot_mat = matrix_from_quat(quat_inv.reshape(-1, 4))  # (num_envs * n_frames, 3, 3)
    vel_w = future_obj_ref_lin_vel_w.reshape(-1, 3, 1)  # (num_envs * n_frames, 3, 1)
    vel_b = torch.bmm(rot_mat, vel_w).squeeze(-1)  # (num_envs * n_frames, 3)
    
    # Reshape back to (num_envs, n_frames, 3) and flatten to (num_envs, n_frames * 3)
    return vel_b.reshape(env.num_envs, n_frames, 3).reshape(env.num_envs, -1)

def teacher_ref_obj_lin_vel_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of reference object linear velocities in current robot base frame from teacher_motion.
    
    Each future reference velocity is transformed to the **current** robot base frame.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened object linear velocities
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future reference object linear velocities in world frame: (num_envs, n_frames, 3)
    future_obj_ref_lin_vel_w = command.teacher_motion.obj_lin_vel[batch_motion_ids, future_timesteps]
    
    # Get current robot anchor orientation (same for all future frames)
    robot_anchor_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # Invert quaternion for coordinate transformation
    quat_inv = robot_anchor_quat_w.clone()
    quat_inv[:, :3] = -quat_inv[:, :3]
    
    # Expand to match future frames dimension
    quat_inv = quat_inv.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future velocities to current robot base frame
    # Reshape for batch matrix multiplication
    rot_mat = matrix_from_quat(quat_inv.reshape(-1, 4))  # (num_envs * n_frames, 3, 3)
    vel_w = future_obj_ref_lin_vel_w.reshape(-1, 3, 1)  # (num_envs * n_frames, 3, 1)
    vel_b = torch.bmm(rot_mat, vel_w).squeeze(-1)  # (num_envs * n_frames, 3)
    
    # Reshape back to (num_envs, n_frames, 3) and flatten to (num_envs, n_frames * 3)
    return vel_b.reshape(env.num_envs, n_frames, 3).reshape(env.num_envs, -1)

def ref_obj_ang_vel_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of reference object angular velocities in current robot base frame.
    
    Each future reference velocity is transformed to the **current** robot base frame.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened object angular velocities
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future reference object angular velocities in world frame: (num_envs, n_frames, 3)
    future_obj_ref_ang_vel_w = command.motion.obj_ang_vel[batch_motion_ids, future_timesteps]
    
    # Get current robot anchor orientation (same for all future frames)
    robot_anchor_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # Invert quaternion for coordinate transformation
    quat_inv = robot_anchor_quat_w.clone()
    quat_inv[:, :3] = -quat_inv[:, :3]
    
    # Expand to match future frames dimension
    quat_inv = quat_inv.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future velocities to current robot base frame
    # Reshape for batch matrix multiplication
    rot_mat = matrix_from_quat(quat_inv.reshape(-1, 4))  # (num_envs * n_frames, 3, 3)
    vel_w = future_obj_ref_ang_vel_w.reshape(-1, 3, 1)  # (num_envs * n_frames, 3, 1)
    vel_b = torch.bmm(rot_mat, vel_w).squeeze(-1)  # (num_envs * n_frames, 3)
    
    # Reshape back to (num_envs, n_frames, 3) and flatten to (num_envs, n_frames * 3)
    return vel_b.reshape(env.num_envs, n_frames, 3).reshape(env.num_envs, -1)

def teacher_ref_obj_ang_vel_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get current + future frames of reference object angular velocities in current robot base frame from teacher_motion.
    
    Each future reference velocity is transformed to the **current** robot base frame.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened object angular velocities
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    batch_motion_ids, future_timesteps = command.get_future_index()
    n_frames = future_timesteps.shape[-1]
    
    # Get future reference object angular velocities in world frame: (num_envs, n_frames, 3)
    future_obj_ref_ang_vel_w = command.teacher_motion.obj_ang_vel[batch_motion_ids, future_timesteps]
    
    # Get current robot anchor orientation (same for all future frames)
    robot_anchor_quat_w = command.robot_anchor_quat_w  # (num_envs, 4)
    
    # Invert quaternion for coordinate transformation
    quat_inv = robot_anchor_quat_w.clone()
    quat_inv[:, :3] = -quat_inv[:, :3]
    
    # Expand to match future frames dimension
    quat_inv = quat_inv.unsqueeze(1).expand(-1, n_frames, -1)  # (num_envs, n_frames, 4)
    
    # Transform all future velocities to current robot base frame
    # Reshape for batch matrix multiplication
    rot_mat = matrix_from_quat(quat_inv.reshape(-1, 4))  # (num_envs * n_frames, 3, 3)
    vel_w = future_obj_ref_ang_vel_w.reshape(-1, 3, 1)  # (num_envs * n_frames, 3, 1)
    vel_b = torch.bmm(rot_mat, vel_w).squeeze(-1)  # (num_envs * n_frames, 3)
    
    # Reshape back to (num_envs, n_frames, 3) and flatten to (num_envs, n_frames * 3)
    return vel_b.reshape(env.num_envs, n_frames, 3).reshape(env.num_envs, -1)


def obj2body_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get object position in each body frame.
    
    Computes the position of the object relative to each tracked body.
    
    Returns:
        torch.Tensor: Shape (num_envs, num_bodies * 3) - flattened object positions in body frames
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_body_pos_w,
        command.robot_body_quat_w,
        command.obj_pos_w[:, None, :].expand(-1, num_bodies, -1),
        command.obj_quat_w[:, None, :].expand(-1, num_bodies, -1),
    )
    
    return pos_b.reshape(env.num_envs, -1)


def obj2body_ori(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get object orientation in each body frame (as 6D rotation).
    
    Computes the orientation of the object relative to each tracked body,
    returns first two columns of rotation matrix.
    
    Returns:
        torch.Tensor: Shape (num_envs, num_bodies * 6) - flattened 6D orientations
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_body_pos_w,
        command.robot_body_quat_w,
        command.obj_pos_w[:, None, :].expand(-1, num_bodies, -1),
        command.obj_quat_w[:, None, :].expand(-1, num_bodies, -1),
    )
    
    # Convert quaternion to rotation matrix and take first 2 columns
    mat = matrix_from_quat(ori_b)  # (num_envs, num_bodies, 3, 3)
    mat_2cols = mat[..., :2]  # (num_envs, num_bodies, 3, 2)
    
    return mat_2cols.reshape(env.num_envs, -1)



def generated_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get generated command from generator.
    
    Returns:
        torch.Tensor: Shape (num_envs, future_frames * 3) - flattened predicted anchor angular velocity
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    generated_command = command.generated_command
    
    return generated_command['action'].reshape(env.num_envs, -1)


def root_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.root_lin_vel_b.reshape(env.num_envs, -1)

def root_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.root_ang_vel_b.reshape(env.num_envs, -1)


def contact_label(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.contact_label.reshape(env.num_envs, -1)


def project_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=command.device).expand(env.num_envs, 3)
    project_gravity = quat_apply(quat_inv(command.robot_base_quat_w), gravity_w) 
    return project_gravity.view(env.num_envs, -1)


def joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos.reshape(env.num_envs, -1)


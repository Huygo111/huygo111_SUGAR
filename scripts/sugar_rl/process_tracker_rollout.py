import argparse
import glob
import os
import numpy as np
import torch
import numpy as np
from tqdm import tqdm
import zarr
import pickle
import matplotlib.pyplot as plt

def matrix_from_quat_np(quaternions: np.ndarray) -> np.ndarray:
    # w, x, y, z
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / np.sum(quaternions**2, axis=-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    q_conj = q.copy()
    q_conj[..., 1:] *= -1
    return q_conj

def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return quat_conjugate_np(q) / np.clip(np.sum(q**2, axis=-1, keepdims=True), eps, None)

def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if q1.shape != q2.shape:
        raise ValueError(f"Shape mismatch: {q1.shape} != {q2.shape}")
    
    orig_shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(orig_shape)

def quat_apply_np(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    orig_shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec) * 2
    res = vec + quat[:, 0:1] * t + np.cross(xyz, t)
    return res.reshape(orig_shape)

def subtract_frame_transforms_np(t01, q01, t02=None, q02=None):
    q10 = quat_inv_np(q01)
    q12 = quat_mul_np(q10, q02) if q02 is not None else q10
    
    diff_t = (t02 - t01) if t02 is not None else -t01
    t12 = quat_apply_np(q10, diff_t)
    return t12, q12


def extract_motion_id(npz_path: str) -> int:
    """Extract motion_id from filename like motion_{motion_id}_env_{env_id}_t{t_start}-{t_end}_idx_{id}.npz."""
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    return int(basename.split('_')[1])

def extract_env_id(npz_path: str) -> int:
    """Extract env_id from filename like motion_{motion_id}_env_{env_id}_t{t_start}-{t_end}_idx_{id}.npz."""
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    return int(basename.split('_')[3])

def extract_t_start(npz_path: str) -> int:
    """Extract t_start from filename like motion_{motion_id}_env_{env_id}_t{t_start}-{t_end}_idx_{id}.npz."""
    basename = os.path.splitext(os.path.basename(npz_path))[0]
    t_range = basename.split('_')[4]  # 't0-99'
    t_start = int(t_range[1:].split('-')[0])  # Remove 't' prefix and get start
    return t_start

def collect_npz_files(input_dir: str) -> list:
    """Collect all NPZ files from input directory."""
    pattern = os.path.join(input_dir, "*.npz")
    files = sorted(glob.glob(pattern))
    
    # Also check for motion_*_env_*.npz pattern in subdirectories
    if len(files) == 0:
        pattern = os.path.join(input_dir, "**", "*.npz")
        files = sorted(glob.glob(pattern, recursive=True))
    
    return files

def process_data_to_dict(npz_path):
    data = np.load(npz_path,allow_pickle=True)

    # 50Hz -> 10Hz 
    stride = 5

    T_raw = data['body_pos_w'].shape[0] # t, nb, 3
    nb = data['body_pos_w'].shape[1]

    action_full = np.concatenate([
        data['ref_joint_pos'],          # T, 29
        data['ref_root_lin_vel_b'],   # T, 3
        data['ref_root_ang_vel_b'],   # T, 3
        data['ref_contact_label'][:, None], # T, 1
    ], axis=-1)  # T, 36

    episode_data = {
        # state
        'obj_pos_b': data['obj_pos_b'][::stride], # t,3
        'obj_ori_b': matrix_from_quat_np(data['obj_quat_b'])[::stride,:,:2].reshape(-1,6), # t,6
        'joint_pos': data['joint_pos'][::stride],   # t,29
        'project_gravity': data['project_gravity'][::stride], # t,3
        # command(optional)
        'target_obj_pos_b': data['target_obj_pos_b'][::stride],   # t,3
        'target_obj_ori_b': matrix_from_quat_np(data['target_obj_quat_b'])[::stride,:,:2].reshape(-1,6),  # t,6
        'action': action_full[::stride], 
        'last_action': np.concatenate([action_full[::stride][0:1], action_full[::stride][:-1]], axis=0), # t,36
    }


    rollout_start = int(data['rollout_start'])
    rollout_end = int(data['rollout_end'])
    original_max_timestep = int(data['original_max_timestep'])
    
    start_padded = False
    end_padded = False
    
    if rollout_start == 0:
        for key in episode_data.keys():
            first_frame = episode_data[key][0:1]
            episode_data[key] = np.concatenate([first_frame, episode_data[key]], axis=0)
        start_padded = True
    
    if rollout_end >= original_max_timestep - 1:
        for key in episode_data.keys():
            last_frame = episode_data[key][-1:]
            padding = np.repeat(last_frame, 7, axis=0)
            episode_data[key] = np.concatenate([episode_data[key], padding], axis=0)
        end_padded = True


    return episode_data, start_padded, end_padded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    npz_file_paths = collect_npz_files(f'{args.data_dir}/raw_npz/trajectory_complete')

    zroot = zarr.open(f'{args.data_dir}/il_dataset', mode='w', zarr_version=2)
    data_group = zroot.create_group('data')
    meta_group = zroot.create_group('meta')

    all_episode_data = []
    episode_ends = []
    current_end = 0
    
    start_padding_count = 0
    end_padding_count = 0

    print(f"Processing {len(npz_file_paths)} files...")
    for idx,npz_path in enumerate(tqdm(npz_file_paths)):


        e_data, start_padded, end_padded = process_data_to_dict(npz_path)
        all_episode_data.append(e_data)
        
        if start_padded:
            start_padding_count += 1
        if end_padded:
            end_padding_count += 1
        
        current_end += e_data['action'].shape[0]
        episode_ends.append(current_end)

    keys = all_episode_data[0].keys()
    for key in keys:
        combined_data = np.concatenate([d[key] for d in all_episode_data], axis=0)
                
        shape = combined_data.shape

        if combined_data.ndim == 2:
            chunks = (1000, shape[1])
        elif combined_data.ndim == 3:
            chunks = (1000, shape[1], shape[2])
        else:
            raise ValueError(f"Unsupported ndim: {combined_data.ndim}")

        data_group.create_dataset(
            name=key,
            data=combined_data,
            chunks=chunks,
            overwrite=True
        )

    meta_group.create_dataset(
        name='episode_ends',
        data=np.array(episode_ends, dtype=np.int64),
        overwrite=True
    )

    print(f"Successfully saved to {args.data_dir}/zarr")
    print(f"\nPadding Statistics:")
    print(f"  Start padding (t=0): {start_padding_count}/{len(npz_file_paths)} trajectories")
    print(f"  End padding (t=max): {end_padding_count}/{len(npz_file_paths)} trajectories")


if __name__ == "__main__":
    main()
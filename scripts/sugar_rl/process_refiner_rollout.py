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


def find_first_stable_frame(obj_lin_vel, obj_ang_vel, window_size=25, lin_vel_thresh=0.05, ang_vel_thresh=0.1):
    T = len(obj_lin_vel)
    
    for t in range(T - window_size):
        window_lin_vel = obj_lin_vel[t:t + window_size]
        window_ang_vel = obj_ang_vel[t:t + window_size]
        
        lin_vel_norms = np.linalg.norm(window_lin_vel, axis=1)
        ang_vel_norms = np.linalg.norm(window_ang_vel, axis=1)
        
        if np.all(lin_vel_norms < lin_vel_thresh) and np.all(ang_vel_norms < ang_vel_thresh):
            return t
    
    return 0


def stabilize_initial_frames(data, stable_frame_t):
    if stable_frame_t <= 0:
        return data
    
    data_modified = {key: np.copy(val) for key, val in data.items()}
    
    obj_pos_at_t = data['obj_pos_w'][stable_frame_t].copy()
    obj_quat_at_t = data['obj_quat_w'][stable_frame_t].copy()
    
    for i in range(stable_frame_t):
        data_modified['obj_pos_w'][i] = obj_pos_at_t
        data_modified['obj_quat_w'][i] = obj_quat_at_t
        data_modified['obj_lin_vel_w'][i] = np.zeros(3)
        data_modified['obj_ang_vel_w'][i] = np.zeros(3)
    
    return data_modified


def process_data_to_rl_dataset(npz_path, data_dir, contact_label_type, 
                                        lin_vel_thresh=0.05, ang_vel_thresh=0.1,
                                        stabilize_initial_frames_flag=True):
    data = np.load(npz_path, allow_pickle=True)
    data = {key: data[key] for key in data.files}
    
    motion_id = extract_motion_id(npz_path)
    env_id = extract_env_id(npz_path)
    t_start = extract_t_start(npz_path)
    data_subdir = f'{data_dir}/rl_dataset/data_{motion_id:03d}_{env_id:03d}_t{t_start}'
    os.makedirs(data_subdir, exist_ok=True)
    
    stable_frame_t = 0
    if stabilize_initial_frames_flag:
        print('stabilize initial frame!')
        stable_frame_t = find_first_stable_frame(
            data['obj_lin_vel_w'], 
            data['obj_ang_vel_w'],
            window_size=25,
            lin_vel_thresh=lin_vel_thresh,
            ang_vel_thresh=ang_vel_thresh
        )
        
        if stable_frame_t > 0:
            data = stabilize_initial_frames(data, stable_frame_t)
    


    robot_50hz = {
        'joint_pos': data['joint_pos'],   # t,29
        'joint_vel': data['joint_vel'],   # t,29
        'body_pos_w': data['body_pos_w'],   # t, nb, 3
        'body_quat_w': data['body_quat_w'],  # t, nb, 4
        'body_lin_vel_w': data['body_lin_vel_w'],   # t, nb, 3
        'body_ang_vel_w': data['body_ang_vel_w'],   # t, nb
    }
    object_motion_global_50hz = {
        'obj_trans': data['obj_pos_w'],   # t,3
        'obj_rot': matrix_from_quat_np(data['obj_quat_w']),
        'obj_lin_vel': data['obj_lin_vel_w'],   # t,3
        'obj_ang_vel': data['obj_ang_vel_w'],   # t,3
    }
    contact_label_50hz = data[contact_label_type]   # t,

    np.savez(f'{data_subdir}/robot_50hz.npz', **robot_50hz)
    with open(f'{data_subdir}/obj_motion_global_50hz.pkl', 'wb') as f:
        pickle.dump(object_motion_global_50hz, f)
    np.save(f'{data_subdir}/contact_labels_50hz.npy', contact_label_50hz)
    
    return motion_id, env_id, t_start, stable_frame_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True, choices=['CarryBox','KickBox', 'PickBottle', 'PushBox', 'SitChair', 'StandBottle'])
    parser.add_argument("--lin_vel_thresh", type=float, default=0.05)
    parser.add_argument("--ang_vel_thresh", type=float, default=0.1)

    args = parser.parse_args()
    npz_file_paths = collect_npz_files(f'{args.data_dir}/raw_npz/trajectory_complete')

    if args.task_name == 'SitChair':
        stabilize_initial_frames_flag = True
        contact_label_type = 'sit_contact_label'
    elif args.task_name == 'StandBottle' or args.task_name == 'PickBottle' or args.task_name == 'CarryBox' or args.task_name == 'PushBox':
        stabilize_initial_frames_flag = False
        contact_label_type = 'hands_contact_label'
    elif args.task_name == 'KickBox':
        stabilize_initial_frames_flag = False
        contact_label_type = 'foot_contact_label'
    else:
        raise ValueError(f"Unsupported task name: {args.task_name}")


    stable_frame_records = []

    print(f"Processing {len(npz_file_paths)} files...")
    for idx, npz_path in enumerate(tqdm(npz_file_paths)):
        motion_id, env_id, t_start, stable_frame_t = process_data_to_rl_dataset(
            npz_path, 
            args.data_dir, 
            contact_label_type,
            lin_vel_thresh=args.lin_vel_thresh,
            ang_vel_thresh=args.ang_vel_thresh,
            stabilize_initial_frames_flag=stabilize_initial_frames_flag
        )
        stable_frame_records.append((motion_id, env_id, t_start, stable_frame_t))
    
    output_txt_path = os.path.join(args.data_dir, 'stable_frame_records.txt')
    with open(output_txt_path, 'w') as f:
        f.write("motion_id\tenv_id\tt_start\tstable_frame_t\n")
        f.write("-" * 50 + "\n")
        for motion_id, env_id, t_start, stable_frame_t in stable_frame_records:
            f.write(f"{motion_id:03d}\t\t{env_id:03d}\t\t{t_start}\t\t{stable_frame_t}\n")
        
        total = len(stable_frame_records)
        modified = sum(1 for _, _, _, t in stable_frame_records if t > 0)
        f.write("-" * 50 + "\n")
        f.write(f"Total trajectories: {total}\n")
        f.write(f"Modified trajectories (t > 0): {modified}\n")
    
    print(f"\nStable frame records saved to: {output_txt_path}")
    print(f"Total: {total}, Modified: {modified}")



if __name__ == "__main__":
    main()
from typing import Dict
import torch
import numpy as np
import copy
from sugar_il.common.pytorch_util import dict_apply
from sugar_il.common.streaming_replay_buffer import StreamingReplayBuffer
from sugar_il.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from sugar_il.model.common.normalizer import LinearNormalizer
from sugar_il.dataset.base_dataset import BaseLowdimDataset
import torch.nn.functional as F

import numpy as np

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """ (w, -x, -y, -z)"""
    res = q.copy()
    res[..., 1:] *= -1
    return res

def quat_inv(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # q_inv = q_conjugate / |q|^2
    return quat_conjugate(q) / np.maximum(np.sum(q**2, axis=-1, keepdims=True), eps)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if q1.shape != q2.shape:
        raise ValueError(f"Expected input shape mismatch: {q1.shape} != {q2.shape}")
    
    shape = q1.shape
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

    return np.stack([w, x, y, z], axis=-1).reshape(shape)

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    orig_shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec) * 2
    res = vec + quat[:, 0:1] * t + np.cross(xyz, t)
    
    return res.reshape(orig_shape)

def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray, t02: np.ndarray = None, q02: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
        
    if t02 is not None:
        diff_t = t02 - t01
    else:
        diff_t = -t01
        
    t12 = quat_apply(q10, diff_t)
    
    return t12, q12

class GeneratorDataset(BaseLowdimDataset):
    def __init__(self,
            zarr_paths,
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            image_size=(518, 518),
            use_last_action=None,
            use_target=None,
            ):
        
        super().__init__()

        assert use_last_action is not None, "Must specify use_last_action for GeneratorDataset"
        assert use_target is not None, "Must specify use_target for GeneratorDataset"

        self.image_size = image_size
        
        # Initialize storage lists
        self.replay_buffers = []
        self.train_masks = []
        self.samplers = []
        self.sampler_lens = []
        
        # Process each zarr file
        for zarr_path in zarr_paths:
            # Create replay buffer
            replay_buffer = StreamingReplayBuffer.copy_from_path(
                zarr_path, keys=[
                    'obj_pos_b',
                    'obj_ori_b',
                    'joint_pos',
                    'project_gravity',
                    'target_obj_pos_b',
                    'target_obj_ori_b',
                    'action',
                    'last_action',
                    ])
            self.replay_buffers.append(replay_buffer)
            
            # Create train mask
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes,
                val_ratio=val_ratio,
                seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=max_train_episodes,
                seed=seed)
            self.train_masks.append(train_mask)
            # Create sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=train_mask,
                # key_first_k=dict(right_cam_img=n_obs_steps, rgbm=n_obs_steps)
                )
            self.samplers.append(sampler)
            
            # Record sampler length
            self.sampler_lens.append(len(sampler))

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.use_last_action = use_last_action
        self.use_target = use_target

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.samplers = []
        val_set.train_masks = []
        val_set.sampler_lens = []
        
        for i, replay_buffer in enumerate(self.replay_buffers):
            # Create validation set sampler
            sampler = SequenceSampler(
                replay_buffer=replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=~self.train_masks[i],
                # key_first_k=dict(right_cam_img=self.n_obs_steps, rgbm=self.n_obs_steps)
                )
            val_set.samplers.append(sampler)
            val_set.train_masks.append(~self.train_masks[i])
            val_set.sampler_lens.append(len(sampler))
            
        return val_set


    def _sample_to_data(self, sample):
        T_slice = slice(self.n_obs_steps)

        # obs
        obj_pos_b = sample['obj_pos_b'].astype(np.float32)
        obj_ori_b = sample['obj_ori_b'].astype(np.float32)
        last_action = sample['last_action'].astype(np.float32)
        joint_pos = sample['joint_pos'].astype(np.float32)
        project_gravity = sample['project_gravity'].astype(np.float32)

        # command
        target_obj_pos_b = sample['target_obj_pos_b'].astype(np.float32)
        target_obj_ori_b = sample['target_obj_ori_b'].astype(np.float32)


        # action
        action = sample['action'].astype(np.float32)

        data = {
            'obs': {
                'obj_pos_b': obj_pos_b[T_slice],
                'obj_ori_b': obj_ori_b[T_slice],
            },
            'action': action
        }
        if self.use_last_action:
            data['obs']['last_action'] = last_action[T_slice]
        else:
            data['obs']['joint_pos'] = joint_pos[T_slice]
            data['obs']['project_gravity'] = project_gravity[T_slice]

        if self.use_target:
            data['obs']['target_obj_pos_b'] = target_obj_pos_b[T_slice]
            data['obs']['target_obj_ori_b'] = target_obj_ori_b[T_slice]


        return data

    def get_normalizer(self, mode='limits', **kwargs):
        # Merge all data
        actions = []
        last_action = []
        obj_pos_b_list = []
        obj_ori_b_list = []
        joint_pos_list = []
        project_gravity_list = []
        target_obj_pos_b_list = []
        target_obj_ori_b_list = []

        for rb in self.replay_buffers:
            actions.append(rb['action'])
            last_action.append(rb['last_action'])
            obj_pos_b_list.append(rb['obj_pos_b'])
            obj_ori_b_list.append(rb['obj_ori_b'])
            joint_pos_list.append(rb['joint_pos'])
            project_gravity_list.append(rb['project_gravity'])
            target_obj_pos_b_list.append(rb['target_obj_pos_b'])
            target_obj_ori_b_list.append(rb['target_obj_ori_b'])
        data = {
            'action': np.concatenate(actions, axis=0),
            'obj_pos_b': np.concatenate(obj_pos_b_list, axis=0),
            'obj_ori_b': np.concatenate(obj_ori_b_list, axis=0),
            'joint_pos': np.concatenate(joint_pos_list, axis=0),
            'project_gravity': np.concatenate(project_gravity_list, axis=0),
            'target_obj_pos_b': np.concatenate(target_obj_pos_b_list, axis=0),
            'target_obj_ori_b': np.concatenate(target_obj_ori_b_list, axis=0),
            'last_action': np.concatenate(last_action, axis=0),
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find corresponding sampler
        curr_idx = idx
        for i, length in enumerate(self.sampler_lens):
            if curr_idx < length:
                sample = self.samplers[i].sample_sequence(curr_idx)
                break
            curr_idx -= length
            
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def __len__(self):
        return sum(self.sampler_lens)

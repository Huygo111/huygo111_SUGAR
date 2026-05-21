import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops.layers.torch import Rearrange

from sugar_il.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)


class GeneratorStateObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            feature_dim: int,
            use_last_action: bool,
            use_target: bool,
        ):
        super().__init__()

        self.use_last_action = use_last_action
        self.use_target = use_target


        # Create state_net to process robot arm and dexterous hand states
        self.obj_state_net = nn.Sequential(
            nn.Linear(9, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        if self.use_last_action:
            self.robot_state_net = nn.Sequential(
                nn.Linear(36, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        else:
            self.robot_state_net = nn.Sequential(
                nn.Linear(32, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        
        if self.use_target:
            self.target_state_net = nn.Sequential(
                nn.Linear(9, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, feature_dim),
                nn.LayerNorm(feature_dim)
            )



        self.shape_meta = shape_meta
        self.feature_dim = feature_dim

        logger.info(
            "Number of parameters in obs encoder: %e", sum(p.numel() for p in self.parameters())
        )

    def forward_obj_state(self, obj_state):
        # state_data: B,T,13
        B, T = obj_state.shape[:2]
        obj_state = obj_state.reshape(B*T, -1)
        obj_state_feature = self.obj_state_net(obj_state)  # (B*T, feature_dim)
        obj_state_feature = obj_state_feature.reshape(B, T, obj_state_feature.shape[-1])  # (B, T, feature_dim)
        return obj_state_feature

    def forward_robot_state(self, robot_state):
        # state_data: B,T,13
        B, T = robot_state.shape[:2]
        robot_state = robot_state.reshape(B*T, -1)
        robot_state_feature = self.robot_state_net(robot_state)  # (B*T, feature_dim)
        robot_state_feature = robot_state_feature.reshape(B, T, robot_state_feature.shape[-1])  # (B, T, feature_dim)
        return robot_state_feature

    def forward_target_state(self, target_state):
        # state_data: B,T,13
        B, T = target_state.shape[:2]
        target_state = target_state.reshape(B*T, -1)
        target_state_feature = self.target_state_net(target_state)  # (B*T, feature_dim)
        target_state_feature = target_state_feature.reshape(B, T, target_state_feature.shape[-1])  # (B, T, feature_dim)
        return target_state_feature


    def forward(self, obs_dict, training=True):
        """
        Input:
        obs_dict = {
            'rgbm': (B,T,4,H,W),      # Head camera RGBM image
            'right_cam_img': (B,T,3,H,W), # Wrist camera RGB image  
            'right_state': (B,T,13)    # Robot arm state
        }
        Output:
        embeddings: (B,T*(num_patches*2+1),feature_dim) # Concatenate all features along sequence length dimension
                                                       # head and wrist each output T*num_patches features
                                                       # state outputs T features
        """


        obj_embedding = self.forward_obj_state(
            torch.cat([
                obs_dict['obj_pos_b'],  # B, T, 3
                obs_dict['obj_ori_b'],  # B, T, 6
            ],dim=-1)
        )  # B, T, 9 -> B, T, 256
        if self.use_last_action:
            robot_embedding = self.forward_robot_state(
                    obs_dict['last_action'], # B, T, 36
            )  # B, T, 36 -> B, T, 256
        else:
            robot_embedding = self.forward_robot_state(
                torch.cat([
                    obs_dict['joint_pos'],  # B, T, 29
                    obs_dict['project_gravity'],  # B, T, 3
                ],dim=-1)
            )  # B, T, 32 -> B, T, 256

        if self.use_target:
            target_embedding = self.forward_target_state(
                torch.cat([
                    obs_dict['target_obj_pos_b'],  # B, T, 3
                    obs_dict['target_obj_ori_b'],  # B, T, 6
                ],dim=-1)
            )  # B, T, 9 -> B, T, 256

            return torch.cat([
                obj_embedding,
                robot_embedding,
                target_embedding
            ],dim=1)
        else:
            return torch.cat([
                obj_embedding,
                robot_embedding,
            ],dim=1)


    @torch.no_grad()
    def output_shape(self):
        if self.use_target:
            return (1, self.shape_meta['obs']['n_obs_steps']*3, self.feature_dim), [self.shape_meta['obs']['n_obs_steps'], self.shape_meta['obs']['n_obs_steps'], self.shape_meta['obs']['n_obs_steps']]
        else:
            return (1, self.shape_meta['obs']['n_obs_steps']*2, self.feature_dim), [self.shape_meta['obs']['n_obs_steps'], self.shape_meta['obs']['n_obs_steps']]

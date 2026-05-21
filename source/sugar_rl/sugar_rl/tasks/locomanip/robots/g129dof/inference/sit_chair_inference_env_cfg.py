from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import sugar_rl.tasks.locomanip.mdp as mdp
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
from sugar_rl.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG

from sugar_rl.assets.objects.objects import CHAIR_CFG 
import torch

from .base_inference_env_cfg import BaseRobotSceneCfg, BaseCommandsCfg, BaseActionsCfg, BaseObservationsCfg, BaseEventCfg, BaseRewardsCfg, BaseTerminationsCfg

@configclass
class RobotSceneCfg(BaseRobotSceneCfg):
    obj: RigidObjectCfg = CHAIR_CFG.replace(prim_path="{ENV_REGEX_NS}/Obj")

@configclass
class CommandsCfg(BaseCommandsCfg):
    pass

@configclass
class ActionsCfg(BaseActionsCfg):
    pass

@configclass
class ObservationsCfg(BaseObservationsCfg):
    pass

@configclass
class EventCfg(BaseEventCfg):
    pass

@configclass
class RewardsCfg(BaseRewardsCfg):
    pass
@configclass
class TerminationsCfg(BaseTerminationsCfg):
    pass

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
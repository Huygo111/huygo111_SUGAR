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

from .base_refiner_env_cfg import BaseRobotSceneCfg, BaseCommandsCfg, BaseActionsCfg, BaseObservationsCfg, BaseEventCfg, BaseRewardsCfg, BaseTerminationsCfg

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
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        params={"velocity_range": {
        "x": (-0.3, 0.3),
        "y": (-0.3, 0.3),
        "z": (-0.15, 0.15),
        "roll": (-0.3, 0.3),
        "pitch": (-0.3, 0.3),
        "yaw": (-0.5, 0.5),
    }},
    )
    push_robot_sit = EventTerm(
        func=mdp.push_by_setting_velocity_on_contact,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-1.0, 1.0),
            "pitch": (-1.0, 1.0),
            "yaw": (-1.0, 1.0),
            },
    },
    )

@configclass
class RewardsCfg(BaseRewardsCfg):
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!.*hip.*)(?!.*pelvis.*).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    hoi_contact = RewTerm(
        func=mdp.sit_contact,
        weight=1.0,
        params={
            "left_hip_sensor_cfg": SceneEntityCfg(
                "left_hip_forces",
            ),
            "right_hip_sensor_cfg": SceneEntityCfg(
                "right_hip_forces",
            ),
            "pelvis_sensor_cfg": SceneEntityCfg(
                "pelvis_forces",
            ),
            "command_name": "motion",
            "threshold": 0.1,
        },
    )
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


class RobotRolloutPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        
        self.commands.motion.rollout_traj = True
        self.events.push_robot=None
        self.events.push_object=None
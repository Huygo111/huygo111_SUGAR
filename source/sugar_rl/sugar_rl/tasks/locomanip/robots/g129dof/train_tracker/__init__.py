import gymnasium as gym

# carry box
gym.register(
    id="Sugar-G129dof-CarryBox-Tracker",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carry_box_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.carry_box_tracker_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)


gym.register(
    id="Sugar-G129dof-CarryBox-Tracker-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carry_box_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.carry_box_tracker_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)

# kick box
gym.register(
    id="Sugar-G129dof-KickBox-Tracker",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kick_box_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.kick_box_tracker_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-KickBox-Tracker-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kick_box_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.kick_box_tracker_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)


# pick bottle
gym.register(
    id="Sugar-G129dof-PickBottle-Tracker",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_bottle_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.pick_bottle_tracker_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-PickBottle-Tracker-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_bottle_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.pick_bottle_tracker_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)

# push box
gym.register(
    id="Sugar-G129dof-PushBox-Tracker",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_box_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.push_box_tracker_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-PushBox-Tracker-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_box_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.push_box_tracker_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)


# sit chair
gym.register(
    id="Sugar-G129dof-SitChair-Tracker",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sit_chair_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sit_chair_tracker_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-SitChair-Tracker-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sit_chair_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sit_chair_tracker_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)

# stand bottle
gym.register(
    id="Sugar-G129dof-StandBottle-Tracker",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_bottle_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stand_bottle_tracker_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-StandBottle-Tracker-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_bottle_tracker_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stand_bottle_tracker_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_bcppo_cfg:BCPPORunnerCfg",
    },
)



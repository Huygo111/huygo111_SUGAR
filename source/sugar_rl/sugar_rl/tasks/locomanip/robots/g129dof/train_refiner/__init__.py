import gymnasium as gym


# carry box
gym.register(
    id="Sugar-G129dof-CarryBox-Refiner",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carry_box_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.carry_box_refiner_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-CarryBox-Refiner-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carry_box_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.carry_box_refiner_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# kick box
gym.register(
    id="Sugar-G129dof-KickBox-Refiner",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kick_box_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.kick_box_refiner_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-KickBox-Refiner-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kick_box_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.kick_box_refiner_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# pick bottle
gym.register(
    id="Sugar-G129dof-PickBottle-Refiner",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_bottle_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.pick_bottle_refiner_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-PickBottle-Refiner-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_bottle_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.pick_bottle_refiner_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# push box
gym.register(
    id="Sugar-G129dof-PushBox-Refiner",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_box_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.push_box_refiner_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-PushBox-Refiner-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_box_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.push_box_refiner_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


# sit chair
gym.register(
    id="Sugar-G129dof-SitChair-Refiner",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sit_chair_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sit_chair_refiner_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-SitChair-Refiner-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sit_chair_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sit_chair_refiner_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# stand bottle
gym.register(
    id="Sugar-G129dof-StandBottle-Refiner",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_bottle_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stand_bottle_refiner_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-StandBottle-Refiner-Rollout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_bottle_refiner_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stand_bottle_refiner_env_cfg:RobotRolloutPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
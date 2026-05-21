import gymnasium as gym

gym.register(
    id="Sugar-G129dof-CarryBox-Inference",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carry_box_inference_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.carry_box_inference_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Sugar-G129dof-KickBox-Inference",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kick_box_inference_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.kick_box_inference_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Sugar-G129dof-SitChair-Inference",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sit_chair_inference_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sit_chair_inference_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Sugar-G129dof-StandBottle-Inference",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_bottle_inference_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stand_bottle_inference_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Sugar-G129dof-PushBox-Inference",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_box_inference_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.push_box_inference_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Sugar-G129dof-PickBottle-Inference",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_bottle_inference_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.pick_bottle_inference_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"sugar_rl.tasks.locomanip.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
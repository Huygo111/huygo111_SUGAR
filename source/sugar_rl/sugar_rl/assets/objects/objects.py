from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
import isaaclab.sim as sim_utils


BIGBOX_CFG = RigidObjectCfg(
    spawn=UsdFileCfg(
        usd_path="descriptions/objects/big_box/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.75),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)


CHAIR_CFG = RigidObjectCfg(
    spawn=UsdFileCfg(
        usd_path="descriptions/objects/chair/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)

BOTTLE_CFG = RigidObjectCfg(
    spawn=UsdFileCfg(
        usd_path="descriptions/objects/bottle/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            angular_damping=0.2
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.75),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)

SMALLBOX_CFG = RigidObjectCfg(
    spawn=UsdFileCfg(
        usd_path="descriptions/objects/small_box/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)
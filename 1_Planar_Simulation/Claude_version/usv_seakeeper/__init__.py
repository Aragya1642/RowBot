"""USV Sea-Keeper: a 1-D/3-DOF unmanned surface vessel simulator with a
pluggable controller interface and Gymnasium environments.

Quick start
-----------
>>> from usv_seakeeper import SpeedHoldEnv, PIDSpeedController, rollout
>>> env = SpeedHoldEnv(preset="coastal_chop")
>>> res = rollout(env, PIDSpeedController(), seed=0)
>>> print(res.summary())                       # doctest: +SKIP
"""
from .config import (
    G, RHO, WaveConfig, VesselConfig, SensorConfig, SimConfig,
    SpeedTaskConfig, StationTaskConfig, RewardConfig, RandomizeConfig,
    PRESETS, get_preset,
)
from .waves import WaveField, WaveQuery
from .sim import USVSim, USVState, SensorModel
from .observations import ObsConfig, ObservationBuilder
from .controllers import (
    Controller, ControlObs, PIDGains, ZeroController, ConstantController,
    CallableController, PIDSpeedController, CascadePositionController,
    PreviewFeedforwardController, PolicyController,
)
from .envs import SpeedHoldEnv, StationKeepEnv, register_envs, GYMNASIUM_AVAILABLE
from .rollout import rollout, evaluate, compare, comparison_table, RolloutResult
from .render import (SeaScope, animate_rollout, animate_comparison,
                     save_frame, LiveViewer)

__version__ = "0.1.0"

__all__ = [
    "G", "RHO", "WaveConfig", "VesselConfig", "SensorConfig", "SimConfig",
    "SpeedTaskConfig", "StationTaskConfig", "RewardConfig", "RandomizeConfig",
    "PRESETS", "get_preset",
    "WaveField", "WaveQuery", "USVSim", "USVState", "SensorModel",
    "ObsConfig", "ObservationBuilder",
    "Controller", "ControlObs", "PIDGains", "ZeroController",
    "ConstantController", "CallableController", "PIDSpeedController",
    "CascadePositionController", "PreviewFeedforwardController",
    "PolicyController",
    "SpeedHoldEnv", "StationKeepEnv", "register_envs", "GYMNASIUM_AVAILABLE",
    "rollout", "evaluate", "compare", "comparison_table", "RolloutResult",
    "SeaScope", "animate_rollout", "animate_comparison", "save_frame",
    "LiveViewer",
    "__version__",
]

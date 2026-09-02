"""

"""

from typing import Literal, Optional

# Global Constants
G = 9.81        # Gravitaional acceleration [m/s^2]
RHO = 1025.0    # Seawater density [kg/m^3]

# --------------------------------------------------------------------------
# Waves
# --------------------------------------------------------------------------
@dataclass
class WaveConfig:
    """
    Sea State descriptions

    Three modes:
        - Calm -- Flat ocean
        - Regular -- Sinusoidal Wave
        - Irregular -- Wave generated using JONSWAP or other algorithm
    """

    mode: Literal["calm", "regular", "irregular"] = "calm"

    # regular
    amplitude: float = 0.5  # [m]
    period: float = 4.0     # [s]

    # irregular
    # TODO

    # Seed for irregular waves
    seed: Optional[int] = 0 # None -> nondeterministic phases

    # Direction of travel relative to the waves, vessel moves towards +x
    #   +1 -> head sea  (waves run towards the vessel, the vessel gets bogged down)
    #   -1 -> following sea (waves overtake the vessel, the vessel gets a boost)
    heading: Literal[1, -1] = 1

    def steepness(self) -> float:
        if self.mode == "calm":
            pass
        elif self.mode == "regular":
            pass


# --------------------------------------------------------------------------
# Vessels
# --------------------------------------------------------------------------
@dataclass
class VesselConfig:
    """
    Vessel, Actuator, and Hydrodynamic Parameters
    """
    mass: float = 350.0             # [kg]
    length: float = 2.6             # [m]
    beam: float = 1.0               # [m]
    block_coefficient: float = 0.7  # TODO: Claude says it's for draft estimation - I need to verify

    # TODO: Surge

    # TODO: Heave/Pitch

    # Actuator
    thrust_max: float = 500.0       # [N]
    slew_rate: float = 2000.0       # [N/s]
    # TODO: Depth of prop, diameter of prop, longitudinal pos. of prop






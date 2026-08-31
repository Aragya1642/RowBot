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



"""System model module: scenario, geometry, channels."""
from .geometry import build_array_positions, feed_positions
from .rhs import build_phase_matrix, holographic_beamforming_matrix
from .channel import (
    near_field_array_response,
    generate_channel_vector,
    generate_target_response,
)
from .scenario import Scenario, generate_scenario

__all__ = [
    "build_array_positions",
    "feed_positions",
    "build_phase_matrix",
    "holographic_beamforming_matrix",
    "near_field_array_response",
    "generate_channel_vector",
    "generate_target_response",
    "Scenario",
    "generate_scenario",
]

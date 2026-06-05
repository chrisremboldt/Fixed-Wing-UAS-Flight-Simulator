"""
Intruder Management System

Manages multiple intruder aircraft that spawn randomly for DAA testing.
Intruders follow predefined flight patterns and can be configured for various scenarios.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from .state import AircraftState, ControlInputs
from .aircraft import AircraftConfig
from .dynamics import FlightDynamics, SimulationConfig
from .environment import Environment
from .frames import Quaternion, body_quaternion_to_threejs, ned_position_to_threejs


class IntruderType(Enum):
    """Types of intruder flight patterns."""
    CRUISE = "cruise"           # Straight and level flight
    CLIMB = "climb"            # Climbing flight
    DESCENT = "descent"        # Descending flight
    TURNING = "turning"        # Level turning flight
    CROSSING = "crossing"      # Crossing perpendicular to ownship


class IntruderBehavior(Enum):
    """Intruder behavior patterns."""
    COOPERATIVE = "cooperative"    # Maintains altitude and heading
    NONCOOPERATIVE = "noncooperative"  # May change altitude/heading unpredictably
    AGGRESSIVE = "aggressive"     # May turn towards ownship


@dataclass
class IntruderConfig:
    """Configuration for intruder spawning and behavior."""

    # Spawning parameters
    spawn_rate: float = 0.1  # Probability per second of spawning new intruder
    max_intruders: int = 5   # Maximum number of simultaneous intruders
    spawn_distance_min: float = 500.0  # Minimum spawn distance (m)
    spawn_distance_max: float = 2000.0  # Maximum spawn distance (m)
    spawn_altitude_min: float = 50.0   # Minimum spawn altitude (m)
    spawn_altitude_max: float = 300.0  # Maximum spawn altitude (m)

    # Flight parameters
    cruise_speed: float = 35.0  # m/s (increased for faster movement)
    speed_variation: float = 8.0  # m/s variation (increased range)
    altitude_change_rate: float = 2.0  # m/s vertical speed for maneuvers

    # Behavior
    behavior_distribution: Dict[IntruderBehavior, float] = field(default_factory=lambda: {
        IntruderBehavior.COOPERATIVE: 0.6,
        IntruderBehavior.NONCOOPERATIVE: 0.3,
        IntruderBehavior.AGGRESSIVE: 0.1
    })

    # Lifetime
    min_lifetime: float = 30.0  # Minimum time intruder stays active (s)
    max_lifetime: float = 120.0  # Maximum time intruder stays active (s)


@dataclass
class IntruderState:
    """State of a single intruder aircraft."""

    id: int
    aircraft: AircraftConfig
    dynamics: FlightDynamics
    behavior: IntruderBehavior
    intruder_type: IntruderType
    spawn_time: float
    lifetime: float
    last_maneuver_time: float = 0.0
    maneuver_interval: float = 10.0  # How often to perform maneuvers
    target_altitude: Optional[float] = None
    target_heading: Optional[float] = None

    def is_expired(self, current_time: float) -> bool:
        """Check if intruder should be removed."""
        return current_time - self.spawn_time > self.lifetime

    def should_maneuver(self, current_time: float) -> bool:
        """Check if intruder should perform a maneuver."""
        if self.behavior == IntruderBehavior.COOPERATIVE:
            return False  # Cooperative intruders maintain course

        return current_time - self.last_maneuver_time > self.maneuver_interval


class IntruderManager:
    """
    Manages multiple intruder aircraft for DAA testing scenarios.
    """

    def __init__(self, config: IntruderConfig, environment: Environment):
        self.config = config
        self.environment = environment
        self.intruders: List[IntruderState] = []
        self.next_id = 0
        self.random = random.Random(42)  # For reproducible behavior

    def should_spawn_intruder(self, dt: float, ownship_state: AircraftState) -> bool:
        """Determine if a new intruder should spawn."""
        if len(self.intruders) >= self.config.max_intruders:
            return False

        # Probabilistic spawning based on spawn rate
        spawn_probability = 1.0 - np.exp(-self.config.spawn_rate * dt)
        return self.random.random() < spawn_probability

    def spawn_intruder(self, ownship_state: AircraftState) -> IntruderState:
        """Create a new intruder aircraft."""

        # Select behavior based on distribution
        behavior = self._select_behavior()

        # Generate random spawn position
        spawn_pos = self._generate_spawn_position(ownship_state)

        # Create aircraft configuration (simplified for intruders)
        aircraft = self._create_intruder_aircraft()

        # Create initial state
        initial_state = self._create_initial_state(spawn_pos, ownship_state)

        # Create dynamics
        sim_config = SimulationConfig(dt=0.01)  # Same as main sim
        dynamics = FlightDynamics(aircraft, self.environment, sim_config)
        dynamics.reset(initial_state)

        # Determine intruder type and lifetime
        intruder_type = self._select_intruder_type()
        lifetime = self.random.uniform(self.config.min_lifetime, self.config.max_lifetime)

        intruder = IntruderState(
            id=self.next_id,
            aircraft=aircraft,
            dynamics=dynamics,
            behavior=behavior,
            intruder_type=intruder_type,
            spawn_time=ownship_state.time,
            lifetime=lifetime,
            last_maneuver_time=ownship_state.time
        )

        self.next_id += 1
        self.intruders.append(intruder)

        # Print spawn info for debugging
        distance = np.linalg.norm(spawn_pos[:2] - ownship_state.position[:2])
        altitude = -spawn_pos[2]
        speed = np.linalg.norm(intruder.dynamics.state.velocity_body)
        print(f"🎯 Spawned intruder {intruder.id} at {distance:.0f}m distance, {altitude:.0f}m altitude, {speed:.1f} m/s ({speed * 3.6:.1f} km/h) (type: {intruder_type.value}, behavior: {behavior.value})")

        return intruder

    def update_intruders(self, dt: float, ownship_state: AircraftState):
        """Update all intruders for one timestep."""

        # Remove expired intruders
        self.intruders = [i for i in self.intruders if not i.is_expired(ownship_state.time)]

        # Update each intruder's controls and step simulation
        for intruder in self.intruders:
            controls = self._generate_controls(intruder, ownship_state)
            intruder.dynamics.step(controls)

            # Check for maneuvers
            if intruder.should_maneuver(ownship_state.time):
                self._perform_maneuver(intruder, ownship_state)

    def get_intruder_states(self) -> List[Dict[str, Any]]:
        """Get current states of all intruders for visualization/logging."""
        states = []
        for intruder in self.intruders:
            state = intruder.dynamics.state
            pos_x, pos_y, pos_z = ned_position_to_threejs(state.position)
            qw, qx, qy, qz = body_quaternion_to_threejs(state.quaternion)

            states.append({
                'id': intruder.id,
                'type': 'intruder',
                'time': state.time,
                'position': {'x': pos_x, 'y': pos_y, 'z': pos_z},
                'quaternion': {'w': qw, 'x': qx, 'y': qy, 'z': qz},
                'velocity': state.velocity_body.tolist(),
                'airspeed': intruder.dynamics.forces_moments.airspeed,
                'behavior': intruder.behavior.value,
                'intruder_type': intruder.intruder_type.value
            })

        return states

    def _select_behavior(self) -> IntruderBehavior:
        """Select intruder behavior based on configured distribution."""
        r = self.random.random()
        cumulative = 0.0
        for behavior, probability in self.config.behavior_distribution.items():
            cumulative += probability
            if r <= cumulative:
                return behavior
        return IntruderBehavior.COOPERATIVE  # Default fallback

    def _generate_spawn_position(self, ownship_state: AircraftState) -> np.ndarray:
        """Generate a random spawn position around the ownship."""

        # Random distance and angle
        distance = self.random.uniform(self.config.spawn_distance_min, self.config.spawn_distance_max)
        angle = self.random.uniform(0, 2 * np.pi)

        # Position in horizontal plane around ownship
        spawn_x = ownship_state.p_north + distance * np.cos(angle)
        spawn_y = ownship_state.p_east + distance * np.sin(angle)

        # Random altitude within limits
        spawn_z = -self.random.uniform(self.config.spawn_altitude_min, self.config.spawn_altitude_max)

        return np.array([spawn_x, spawn_y, spawn_z])

    def _create_intruder_aircraft(self) -> AircraftConfig:
        """Create a simplified aircraft config for intruders."""
        # Use same config as default but could be customized
        from .aircraft import AircraftConfig
        config = AircraftConfig(name="Intruder UAS")
        # Could add variations here if desired
        return config

    def _create_initial_state(self, spawn_pos: np.ndarray, ownship_state: AircraftState) -> AircraftState:
        """Create initial state for new intruder."""

        # Random speed within limits
        speed = self.config.cruise_speed + self.random.uniform(-self.config.speed_variation, self.config.speed_variation)
        speed = max(20.0, min(50.0, speed))  # Clamp to reasonable limits (increased for faster intruders)

        # Random heading (biased towards crossing ownship path)
        ownship_heading = ownship_state.psi
        heading_variation = self.random.uniform(-np.pi/2, np.pi/2)
        heading = ownship_heading + heading_variation

        # Level attitude; forward speed is along body +X
        quaternion = Quaternion.from_euler(0.0, 0.0, heading)

        return AircraftState(
            position=spawn_pos,
            velocity_body=np.array([speed, 0.0, 0.0]),
            quaternion=quaternion,
            omega_body=np.zeros(3),
            time=ownship_state.time
        )

    def _select_intruder_type(self) -> IntruderType:
        """Select the type of flight pattern for the intruder."""
        types = list(IntruderType)
        return self.random.choice(types)

    def _generate_controls(self, intruder: IntruderState, ownship_state: AircraftState) -> ControlInputs:
        """Generate control inputs for intruder based on its type and behavior."""

        state = intruder.dynamics.state

        # Base throttle for cruise speed
        target_speed = self.config.cruise_speed
        current_speed = intruder.dynamics.forces_moments.airspeed
        throttle = 0.5 + 0.1 * (target_speed - current_speed) / target_speed
        throttle = np.clip(throttle, 0.0, 1.0)

        # Default controls (straight and level)
        elevator = 0.0
        aileron = 0.0
        rudder = 0.0

        # Altitude control if target altitude is set
        if intruder.target_altitude is not None:
            alt_error = intruder.target_altitude - state.altitude
            elevator = -0.01 * alt_error  # Simple proportional control
            elevator = np.clip(elevator, -0.2, 0.2)

            # Check if altitude target reached
            if abs(alt_error) < 10.0:
                intruder.target_altitude = None

        # Heading control if target heading is set
        if intruder.target_heading is not None:
            heading_error = intruder.target_heading - state.psi
            # Normalize to [-pi, pi]
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi

            rudder = 0.1 * heading_error  # Simple proportional control
            rudder = np.clip(rudder, -0.3, 0.3)

            # Check if heading target reached
            if abs(heading_error) < np.radians(5.0):
                intruder.target_heading = None

        return ControlInputs(
            elevator=elevator,
            aileron=aileron,
            rudder=rudder,
            throttle=throttle
        )

    def _perform_maneuver(self, intruder: IntruderState, ownship_state: AircraftState):
        """Perform a random maneuver based on intruder behavior."""

        intruder.last_maneuver_time = ownship_state.time

        if intruder.behavior == IntruderBehavior.COOPERATIVE:
            return  # No maneuvers

        # Random maneuver selection
        maneuver_type = self.random.choice(['altitude_change', 'heading_change', 'both'])

        if maneuver_type in ['altitude_change', 'both']:
            # Change altitude
            alt_change = self.random.uniform(-50.0, 50.0)
            intruder.target_altitude = intruder.dynamics.state.altitude + alt_change

        if maneuver_type in ['heading_change', 'both']:
            # Change heading
            heading_change = self.random.uniform(-np.pi/3, np.pi/3)  # ±60 degrees

            if intruder.behavior == IntruderBehavior.AGGRESSIVE:
                # Aggressive intruders may turn towards ownship
                if self.random.random() < 0.3:  # 30% chance
                    # Calculate bearing to ownship
                    dx = ownship_state.p_north - intruder.dynamics.state.p_north
                    dy = ownship_state.p_east - intruder.dynamics.state.p_east
                    bearing_to_ownship = np.arctan2(dy, dx)
                    intruder.target_heading = bearing_to_ownship
                else:
                    intruder.target_heading = intruder.dynamics.state.psi + heading_change
            else:
                intruder.target_heading = intruder.dynamics.state.psi + heading_change

        # Update intruder type based on maneuver
        if intruder.target_altitude is not None and intruder.target_heading is not None:
            intruder.intruder_type = IntruderType.TURNING
        elif intruder.target_altitude is not None:
            intruder.intruder_type = IntruderType.CLIMB if intruder.target_altitude > intruder.dynamics.state.altitude else IntruderType.DESCENT
        else:
            intruder.intruder_type = IntruderType.CROSSING

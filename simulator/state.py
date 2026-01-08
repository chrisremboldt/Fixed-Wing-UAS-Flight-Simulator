"""
Aircraft State Representation

The state vector contains 13 states for full 6-DOF rigid body simulation:
- Position (3): North, East, Down in NED frame
- Velocity (3): u, v, w in body frame
- Attitude (4): Quaternion (w, x, y, z)
- Angular rates (3): p, q, r in body frame

Plus derived quantities computed from the state for convenience.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .frames import Quaternion, wind_angles


@dataclass
class AircraftState:
    """
    Complete aircraft state for 6-DOF simulation.
    
    All values are in SI units (m, m/s, rad, rad/s).
    """
    
    # Position in NED frame (m)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Velocity in body frame (m/s) - [u, v, w]
    velocity_body: np.ndarray = field(default_factory=lambda: np.array([20.0, 0.0, 0.0]))
    
    # Attitude quaternion (body to NED)
    quaternion: Quaternion = field(default_factory=lambda: Quaternion(1.0, 0.0, 0.0, 0.0))
    
    # Angular velocity in body frame (rad/s) - [p, q, r]
    omega_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Time (s)
    time: float = 0.0
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity_body = np.asarray(self.velocity_body, dtype=np.float64)
        self.omega_body = np.asarray(self.omega_body, dtype=np.float64)
    
    @property
    def p_north(self) -> float:
        """North position (m)."""
        return self.position[0]
    
    @property
    def p_east(self) -> float:
        """East position (m)."""
        return self.position[1]
    
    @property
    def p_down(self) -> float:
        """Down position (m). Negative = above origin."""
        return self.position[2]
    
    @property
    def altitude(self) -> float:
        """Altitude above origin (m). Positive = up."""
        return -self.position[2]
    
    @property
    def u(self) -> float:
        """Forward velocity in body frame (m/s)."""
        return self.velocity_body[0]
    
    @property
    def v(self) -> float:
        """Rightward velocity in body frame (m/s)."""
        return self.velocity_body[1]
    
    @property
    def w(self) -> float:
        """Downward velocity in body frame (m/s)."""
        return self.velocity_body[2]
    
    @property
    def p(self) -> float:
        """Roll rate (rad/s)."""
        return self.omega_body[0]
    
    @property
    def q(self) -> float:
        """Pitch rate (rad/s)."""
        return self.omega_body[1]
    
    @property
    def r(self) -> float:
        """Yaw rate (rad/s)."""
        return self.omega_body[2]
    
    @property
    def euler_angles(self) -> tuple:
        """(phi, theta, psi) - roll, pitch, yaw in radians."""
        return self.quaternion.to_euler()
    
    @property
    def phi(self) -> float:
        """Roll angle (rad)."""
        return self.euler_angles[0]
    
    @property
    def theta(self) -> float:
        """Pitch angle (rad)."""
        return self.euler_angles[1]
    
    @property
    def psi(self) -> float:
        """Yaw/heading angle (rad)."""
        return self.euler_angles[2]
    
    @property
    def velocity_ned(self) -> np.ndarray:
        """Velocity in NED frame (m/s)."""
        return self.quaternion.rotate_vector(self.velocity_body)
    
    @property
    def groundspeed(self) -> float:
        """Horizontal speed over ground (m/s)."""
        v_ned = self.velocity_ned
        return np.sqrt(v_ned[0]**2 + v_ned[1]**2)
    
    @property
    def climb_rate(self) -> float:
        """Vertical speed, positive up (m/s)."""
        return -self.velocity_ned[2]
    
    def get_wind_angles(self, wind_body: Optional[np.ndarray] = None) -> tuple:
        """
        Compute airspeed, alpha, beta accounting for wind.
        
        Args:
            wind_body: Wind velocity in body frame (m/s), optional
            
        Returns:
            (V_air, alpha, beta)
        """
        v_air = self.velocity_body.copy()
        if wind_body is not None:
            v_air -= wind_body
        return wind_angles(v_air)
    
    def to_array(self) -> np.ndarray:
        """
        Convert state to flat array for integration.
        
        Returns:
            13-element array: [pos(3), vel(3), quat(4), omega(3)]
        """
        return np.concatenate([
            self.position,
            self.velocity_body,
            self.quaternion.to_array(),
            self.omega_body
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, time: float = 0.0) -> 'AircraftState':
        """
        Create state from flat array.
        
        Args:
            arr: 13-element array [pos(3), vel(3), quat(4), omega(3)]
            time: Current simulation time
            
        Returns:
            AircraftState instance
        """
        return cls(
            position=arr[0:3],
            velocity_body=arr[3:6],
            quaternion=Quaternion.from_array(arr[6:10]),
            omega_body=arr[10:13],
            time=time
        )
    
    def copy(self) -> 'AircraftState':
        """Create a deep copy of this state."""
        return AircraftState(
            position=self.position.copy(),
            velocity_body=self.velocity_body.copy(),
            quaternion=Quaternion(
                self.quaternion.w,
                self.quaternion.x,
                self.quaternion.y,
                self.quaternion.z
            ),
            omega_body=self.omega_body.copy(),
            time=self.time
        )


@dataclass
class ControlInputs:
    """
    Aircraft control surface deflections and throttle.
    
    All deflections in radians, throttle normalized [0, 1].
    Sign conventions (typical):
        - Elevator: positive = trailing edge down = nose up
        - Aileron: positive = right aileron down = roll right
        - Rudder: positive = trailing edge left = yaw left
    """
    
    elevator: float = 0.0    # δe (rad), positive = pitch up
    aileron: float = 0.0     # δa (rad), positive = roll right
    rudder: float = 0.0      # δr (rad), positive = yaw left (nose left)
    throttle: float = 0.5    # 0 to 1
    
    def clip(self, max_deflection: float = np.radians(25), 
             max_throttle: float = 1.0) -> 'ControlInputs':
        """
        Return clipped control inputs within limits.
        
        Args:
            max_deflection: Maximum control surface deflection (rad)
            max_throttle: Maximum throttle (typically 1.0)
            
        Returns:
            New ControlInputs with clipped values
        """
        return ControlInputs(
            elevator=np.clip(self.elevator, -max_deflection, max_deflection),
            aileron=np.clip(self.aileron, -max_deflection, max_deflection),
            rudder=np.clip(self.rudder, -max_deflection, max_deflection),
            throttle=np.clip(self.throttle, 0.0, max_throttle)
        )


@dataclass 
class ForcesAndMoments:
    """
    Aggregated forces and moments acting on the aircraft.
    
    All in body frame, SI units (N, N·m).
    Forces: [Fx, Fy, Fz]
    Moments: [L, M, N] (roll, pitch, yaw)
    """
    
    # Total forces in body frame (N)
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Total moments about CG in body frame (N·m)
    moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Individual force components for debugging
    aero_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    aero_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    thrust_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    thrust_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gravity_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Aerodynamic state for logging
    alpha: float = 0.0
    beta: float = 0.0
    airspeed: float = 0.0
    dynamic_pressure: float = 0.0
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.force = np.asarray(self.force, dtype=np.float64)
        self.moment = np.asarray(self.moment, dtype=np.float64)
        self.aero_force = np.asarray(self.aero_force, dtype=np.float64)
        self.aero_moment = np.asarray(self.aero_moment, dtype=np.float64)
        self.thrust_force = np.asarray(self.thrust_force, dtype=np.float64)
        self.thrust_moment = np.asarray(self.thrust_moment, dtype=np.float64)
        self.gravity_force = np.asarray(self.gravity_force, dtype=np.float64)


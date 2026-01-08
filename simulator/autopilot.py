"""
Basic Autopilot Controllers

Provides PID-based autopilot modes for:
- Altitude hold
- Heading hold  
- Airspeed hold
- Waypoint following

These are useful for setting up DAA test scenarios where the
ownship needs to maintain a stable flight path.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from .state import AircraftState, ControlInputs


@dataclass
class PIDController:
    """Simple PID controller with anti-windup."""
    
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    
    # Output limits
    min_output: float = -1.0
    max_output: float = 1.0
    
    # Anti-windup: integral limits
    max_integral: float = 10.0
    
    # State
    _integral: float = 0.0
    _last_error: Optional[float] = None
    _last_time: float = 0.0
    
    def reset(self):
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = None
        self._last_time = 0.0
    
    def update(self, error: float, dt: float) -> float:
        """
        Compute control output.
        
        Args:
            error: Current error (setpoint - measured)
            dt: Time step
            
        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0
        
        # Proportional
        p_term = self.kp * error
        
        # Integral with anti-windup
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.max_integral, self.max_integral)
        i_term = self.ki * self._integral
        
        # Derivative
        if self._last_error is not None:
            derivative = (error - self._last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        self._last_error = error
        
        # Sum and clip
        output = p_term + i_term + d_term
        return np.clip(output, self.min_output, self.max_output)


@dataclass
class AltitudeHold:
    """Altitude hold autopilot using pitch control."""
    
    target_altitude: float = 100.0  # m
    
    # Nested controllers
    altitude_controller: PIDController = field(default_factory=lambda: PIDController(
        kp=0.02, ki=0.002, kd=0.05, min_output=-0.3, max_output=0.3
    ))
    climb_rate_controller: PIDController = field(default_factory=lambda: PIDController(
        kp=0.1, ki=0.01, kd=0.0, min_output=-0.3, max_output=0.3
    ))
    
    # Limits
    max_climb_rate: float = 5.0  # m/s
    
    def update(self, state: AircraftState, dt: float) -> float:
        """
        Compute elevator command for altitude hold.
        
        Args:
            state: Current aircraft state
            dt: Time step
            
        Returns:
            Elevator command (rad)
        """
        # Outer loop: altitude -> desired climb rate
        alt_error = self.target_altitude - state.altitude
        desired_climb_rate = self.altitude_controller.update(alt_error, dt)
        desired_climb_rate = np.clip(desired_climb_rate * 20, -self.max_climb_rate, self.max_climb_rate)
        
        # Inner loop: climb rate -> elevator
        climb_rate_error = desired_climb_rate - state.climb_rate
        elevator = self.climb_rate_controller.update(climb_rate_error, dt)
        
        return elevator
    
    def reset(self):
        """Reset controller state."""
        self.altitude_controller.reset()
        self.climb_rate_controller.reset()


@dataclass
class HeadingHold:
    """Heading hold autopilot using coordinated turn."""
    
    target_heading: float = 0.0  # rad, from North
    
    # Controllers
    heading_controller: PIDController = field(default_factory=lambda: PIDController(
        kp=1.0, ki=0.05, kd=0.2, min_output=-0.5, max_output=0.5
    ))
    roll_controller: PIDController = field(default_factory=lambda: PIDController(
        kp=2.0, ki=0.1, kd=0.5, min_output=-0.44, max_output=0.44
    ))
    
    # Limits
    max_bank: float = 0.52  # rad (~30 deg)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def update(self, state: AircraftState, dt: float) -> Tuple[float, float]:
        """
        Compute aileron and rudder commands for heading hold.
        
        Args:
            state: Current aircraft state
            dt: Time step
            
        Returns:
            (aileron, rudder) commands (rad)
        """
        # Heading error
        heading_error = self._normalize_angle(self.target_heading - state.psi)
        
        # Outer loop: heading -> desired bank angle
        desired_bank = self.heading_controller.update(heading_error, dt)
        desired_bank = np.clip(desired_bank, -self.max_bank, self.max_bank)
        
        # Inner loop: bank -> aileron
        bank_error = desired_bank - state.phi
        aileron = self.roll_controller.update(bank_error, dt)
        
        # Coordinated turn: rudder proportional to aileron
        rudder = 0.3 * aileron
        
        return aileron, rudder
    
    def reset(self):
        """Reset controller state."""
        self.heading_controller.reset()
        self.roll_controller.reset()


@dataclass
class AirspeedHold:
    """Airspeed hold autopilot using throttle."""
    
    target_airspeed: float = 25.0  # m/s
    
    controller: PIDController = field(default_factory=lambda: PIDController(
        kp=0.05, ki=0.01, kd=0.02, min_output=0.0, max_output=1.0
    ))
    
    def update(self, airspeed: float, dt: float) -> float:
        """
        Compute throttle command for airspeed hold.
        
        Args:
            airspeed: Current airspeed (m/s)
            dt: Time step
            
        Returns:
            Throttle command [0, 1]
        """
        error = self.target_airspeed - airspeed
        return self.controller.update(error, dt)
    
    def reset(self):
        """Reset controller state."""
        self.controller.reset()


@dataclass
class Waypoint:
    """A waypoint with position and optional altitude/speed."""
    north: float        # m
    east: float         # m
    altitude: Optional[float] = None  # m, None = maintain current
    airspeed: Optional[float] = None  # m/s, None = maintain current
    
    @property
    def position_2d(self) -> np.ndarray:
        """2D position (North, East)."""
        return np.array([self.north, self.east])


@dataclass
class WaypointFollower:
    """
    Waypoint following autopilot.
    
    Flies to waypoints in sequence, combining altitude, heading, and speed holds.
    """
    
    waypoints: List[Waypoint] = field(default_factory=list)
    current_waypoint_index: int = 0
    
    # Arrival radius (m)
    arrival_radius: float = 50.0
    
    # Sub-controllers
    altitude_hold: AltitudeHold = field(default_factory=AltitudeHold)
    heading_hold: HeadingHold = field(default_factory=HeadingHold)
    airspeed_hold: AirspeedHold = field(default_factory=AirspeedHold)
    
    # Default values when waypoint doesn't specify
    default_altitude: float = 100.0
    default_airspeed: float = 25.0
    
    @property
    def current_waypoint(self) -> Optional[Waypoint]:
        """Get current target waypoint."""
        if 0 <= self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if all waypoints have been reached."""
        return self.current_waypoint_index >= len(self.waypoints)
    
    def add_waypoint(self, north: float, east: float, 
                     altitude: Optional[float] = None,
                     airspeed: Optional[float] = None):
        """Add a waypoint to the list."""
        self.waypoints.append(Waypoint(north, east, altitude, airspeed))
    
    def update(self, state: AircraftState, airspeed: float, dt: float) -> ControlInputs:
        """
        Compute control inputs to follow waypoints.
        
        Args:
            state: Current aircraft state
            airspeed: Current airspeed
            dt: Time step
            
        Returns:
            Control inputs
        """
        wp = self.current_waypoint
        
        if wp is None:
            # No waypoints, maintain current state
            return ControlInputs(
                elevator=0.0,
                aileron=0.0,
                rudder=0.0,
                throttle=0.5
            )
        
        # Check arrival
        current_pos_2d = np.array([state.p_north, state.p_east])
        distance = np.linalg.norm(wp.position_2d - current_pos_2d)
        
        if distance < self.arrival_radius:
            self.current_waypoint_index += 1
            wp = self.current_waypoint
            if wp is None:
                return ControlInputs(throttle=0.5)
        
        # Compute heading to waypoint
        delta = wp.position_2d - current_pos_2d
        target_heading = np.arctan2(delta[1], delta[0])
        
        # Update controllers
        self.heading_hold.target_heading = target_heading
        self.altitude_hold.target_altitude = wp.altitude if wp.altitude else self.default_altitude
        self.airspeed_hold.target_airspeed = wp.airspeed if wp.airspeed else self.default_airspeed
        
        elevator = self.altitude_hold.update(state, dt)
        aileron, rudder = self.heading_hold.update(state, dt)
        throttle = self.airspeed_hold.update(airspeed, dt)
        
        return ControlInputs(
            elevator=elevator,
            aileron=aileron,
            rudder=rudder,
            throttle=throttle
        )
    
    def reset(self):
        """Reset to start of waypoint list."""
        self.current_waypoint_index = 0
        self.altitude_hold.reset()
        self.heading_hold.reset()
        self.airspeed_hold.reset()


@dataclass
class BasicAutopilot:
    """
    Combined autopilot with altitude, heading, and airspeed hold.
    
    Simple interface for DAA testing.
    """
    
    altitude_hold: AltitudeHold = field(default_factory=AltitudeHold)
    heading_hold: HeadingHold = field(default_factory=HeadingHold)
    airspeed_hold: AirspeedHold = field(default_factory=AirspeedHold)
    
    # Modes
    altitude_hold_enabled: bool = True
    heading_hold_enabled: bool = True
    airspeed_hold_enabled: bool = True
    
    def set_targets(self, altitude: float = None, heading: float = None, 
                    airspeed: float = None):
        """Set target values for autopilot."""
        if altitude is not None:
            self.altitude_hold.target_altitude = altitude
        if heading is not None:
            self.heading_hold.target_heading = heading
        if airspeed is not None:
            self.airspeed_hold.target_airspeed = airspeed
    
    def update(self, state: AircraftState, airspeed: float, dt: float) -> ControlInputs:
        """
        Compute control inputs.
        
        Args:
            state: Current aircraft state
            airspeed: Current airspeed
            dt: Time step
            
        Returns:
            Control inputs
        """
        elevator = 0.0
        aileron = 0.0
        rudder = 0.0
        throttle = 0.5
        
        if self.altitude_hold_enabled:
            elevator = self.altitude_hold.update(state, dt)
        
        if self.heading_hold_enabled:
            aileron, rudder = self.heading_hold.update(state, dt)
        
        if self.airspeed_hold_enabled:
            throttle = self.airspeed_hold.update(airspeed, dt)
        
        return ControlInputs(
            elevator=elevator,
            aileron=aileron,
            rudder=rudder,
            throttle=throttle
        )
    
    def reset(self):
        """Reset all controllers."""
        self.altitude_hold.reset()
        self.heading_hold.reset()
        self.airspeed_hold.reset()


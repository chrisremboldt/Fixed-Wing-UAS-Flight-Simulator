"""
6-DOF Rigid Body Dynamics

Implements the equations of motion for a rigid body aircraft:
- Translational dynamics (Newton's 2nd law in body frame)
- Rotational dynamics (Euler's equations)
- Kinematic equations (quaternion propagation)

Uses RK4 integration for numerical stability.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional
from enum import Enum, auto

from .state import AircraftState, ControlInputs, ForcesAndMoments
from .aircraft import AircraftConfig
from .environment import Environment
from .aerodynamics import (
    compute_aero_state, 
    compute_aero_forces_moments,
    compute_thrust_forces_moments,
    compute_gravity_force
)
from .frames import Quaternion, quaternion_derivative


class CrashType(Enum):
    """Types of crash/failure conditions."""
    NONE = auto()
    GROUND_COLLISION = auto()
    STRUCTURAL_FAILURE = auto()  # Exceeded g-limits
    OVERSPEED = auto()           # Exceeded Vne
    STALL_SPIN = auto()          # Unrecoverable stall/spin
    UNDERSPEED = auto()          # Below stall speed too long


@dataclass
class CrashState:
    """Crash detection state."""
    crashed: bool = False
    crash_type: CrashType = CrashType.NONE
    crash_time: float = 0.0
    crash_message: str = ""
    
    # Impact data
    impact_velocity: float = 0.0
    impact_g_force: float = 0.0
    
    # Warning states (pre-crash)
    stall_warning: bool = False
    overspeed_warning: bool = False
    terrain_warning: bool = False
    
    # Stall timer for detecting unrecoverable stall
    time_in_stall: float = 0.0


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    
    # Integration timestep (s)
    dt: float = 0.01  # 100 Hz
    
    # Maximum simulation time (s)
    max_time: float = 300.0
    
    # Enable/disable physics components (for validation)
    enable_gravity: bool = True
    enable_aerodynamics: bool = True
    enable_thrust: bool = True
    enable_moments: bool = True
    
    # Ground collision altitude (m above reference)
    ground_altitude: float = 0.0
    
    # State limits for stability
    max_angular_rate: float = 10.0  # rad/s
    
    # Crash detection settings
    enable_crash_detection: bool = True
    max_g_load: float = 6.0           # Maximum load factor before structural failure
    max_negative_g: float = -2.0      # Minimum (negative) load factor
    stall_duration_limit: float = 5.0  # Seconds in stall before crash
    terrain_warning_altitude: float = 50.0  # Altitude for terrain warning


def state_derivative(
    state: AircraftState,
    controls: ControlInputs,
    config: AircraftConfig,
    environment: Environment,
    sim_config: SimulationConfig
) -> Tuple[np.ndarray, ForcesAndMoments]:
    """
    Compute the time derivative of the state vector.
    
    This is the core physics function that computes:
    - All forces and moments
    - Linear and angular accelerations
    - Position and attitude rates
    
    Args:
        state: Current aircraft state
        controls: Control inputs
        config: Aircraft configuration
        environment: Environment model
        sim_config: Simulation settings
        
    Returns:
        (state_derivative, forces_and_moments)
        state_derivative is a 13-element array matching state.to_array()
    """
    # Get rotation matrices
    R_body_to_ned = state.quaternion.to_dcm()
    R_ned_to_body = R_body_to_ned.T
    
    # Get atmospheric properties
    atmo = environment.get_atmosphere(state.altitude)
    
    # Get wind in body frame
    wind_body = environment.get_wind_body(
        state.position, 
        state.time, 
        R_ned_to_body
    )
    
    # Get gravity in body frame
    gravity_body = environment.get_gravity_body(state.altitude, R_ned_to_body)
    
    # Initialize forces and moments
    fm = ForcesAndMoments()
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    
    # === GRAVITY ===
    if sim_config.enable_gravity:
        F_gravity = compute_gravity_force(config.mass_properties.mass, gravity_body)
        fm.gravity_force = F_gravity.copy()
        total_force += F_gravity
    
    # === AERODYNAMICS ===
    if sim_config.enable_aerodynamics:
        aero_state = compute_aero_state(state, wind_body, atmo, config)
        F_aero, M_aero, aero_state = compute_aero_forces_moments(
            aero_state, controls, config
        )
        fm.aero_force = F_aero.copy()
        fm.aero_moment = M_aero.copy()
        fm.alpha = aero_state.alpha
        fm.beta = aero_state.beta
        fm.airspeed = aero_state.airspeed
        fm.dynamic_pressure = aero_state.q_bar
        
        total_force += F_aero
        if sim_config.enable_moments:
            total_moment += M_aero
    else:
        # Still compute airspeed for display
        v_air = state.velocity_body - wind_body
        fm.airspeed = np.linalg.norm(v_air)
    
    # === THRUST ===
    if sim_config.enable_thrust:
        F_thrust, M_thrust = compute_thrust_forces_moments(
            fm.airspeed, controls, atmo, config
        )
        fm.thrust_force = F_thrust.copy()
        fm.thrust_moment = M_thrust.copy()
        
        total_force += F_thrust
        if sim_config.enable_moments:
            total_moment += M_thrust
    
    # Store totals
    fm.force = total_force.copy()
    fm.moment = total_moment.copy()
    
    # === EQUATIONS OF MOTION ===
    
    mass = config.mass_properties.mass
    I = config.mass_properties.inertia_tensor
    I_inv = config.mass_properties.inertia_inverse
    omega = state.omega_body
    
    # --- Translational dynamics (body frame) ---
    # m * dv/dt = F - ω × (m*v)
    # dv/dt = F/m - ω × v
    v_body = state.velocity_body
    dv_body = total_force / mass - np.cross(omega, v_body)
    
    # --- Rotational dynamics (Euler's equations) ---
    # I * dω/dt = M - ω × (I*ω)
    # dω/dt = I^(-1) * (M - ω × (I*ω))
    if sim_config.enable_moments:
        domega = I_inv @ (total_moment - np.cross(omega, I @ omega))
    else:
        domega = np.zeros(3)
    
    # --- Kinematic equations ---
    
    # Position rate: dp_ned/dt = R_body_to_ned * v_body
    dp_ned = R_body_to_ned @ v_body
    
    # Quaternion rate: dq/dt = 0.5 * Ω(ω) * q
    dq = quaternion_derivative(state.quaternion, omega)
    
    # Assemble state derivative
    # Order: [pos(3), vel(3), quat(4), omega(3)]
    state_dot = np.concatenate([dp_ned, dv_body, dq, domega])
    
    return state_dot, fm


def rk4_step(
    state: AircraftState,
    controls: ControlInputs,
    config: AircraftConfig,
    environment: Environment,
    sim_config: SimulationConfig,
    dt: float
) -> Tuple[AircraftState, ForcesAndMoments]:
    """
    Perform one RK4 integration step.
    
    Args:
        state: Current state
        controls: Control inputs
        config: Aircraft configuration
        environment: Environment model
        sim_config: Simulation settings
        dt: Timestep (s)
        
    Returns:
        (new_state, forces_and_moments at start of step)
    """
    y0 = state.to_array()
    t0 = state.time
    
    # k1
    k1, fm = state_derivative(state, controls, config, environment, sim_config)
    
    # k2
    y_temp = y0 + 0.5 * dt * k1
    state_temp = AircraftState.from_array(y_temp, t0 + 0.5 * dt)
    k2, _ = state_derivative(state_temp, controls, config, environment, sim_config)
    
    # k3
    y_temp = y0 + 0.5 * dt * k2
    state_temp = AircraftState.from_array(y_temp, t0 + 0.5 * dt)
    k3, _ = state_derivative(state_temp, controls, config, environment, sim_config)
    
    # k4
    y_temp = y0 + dt * k3
    state_temp = AircraftState.from_array(y_temp, t0 + dt)
    k4, _ = state_derivative(state_temp, controls, config, environment, sim_config)
    
    # Combine
    y_new = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Create new state
    new_state = AircraftState.from_array(y_new, t0 + dt)
    
    # Normalize quaternion to maintain unit length
    new_state.quaternion._normalize()
    
    # Clamp angular rates for stability
    max_rate = sim_config.max_angular_rate
    new_state.omega_body = np.clip(new_state.omega_body, -max_rate, max_rate)
    
    return new_state, fm


class FlightDynamics:
    """
    Main flight dynamics simulation engine.
    
    Manages simulation state and provides step/run interface.
    Includes crash detection for ground collision, structural limits, and stall.
    """
    
    def __init__(
        self,
        aircraft: AircraftConfig,
        environment: Optional[Environment] = None,
        sim_config: Optional[SimulationConfig] = None
    ):
        self.aircraft = aircraft
        self.environment = environment or Environment()
        self.sim_config = sim_config or SimulationConfig()
        
        # Current state
        self.state = AircraftState()
        
        # Current controls
        self.controls = ControlInputs()
        
        # Latest forces/moments (for debugging)
        self.forces_moments = ForcesAndMoments()
        
        # Crash detection state
        self.crash_state = CrashState()
        
        # History (optional, for analysis)
        self.history: list = []
        self.record_history = False
        
        # Previous state for computing accelerations
        self._prev_velocity = np.zeros(3)
    
    def reset(self, initial_state: Optional[AircraftState] = None):
        """Reset simulation to initial conditions."""
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            # Default: level flight at 20 m/s, 100m altitude
            self.state = AircraftState(
                position=np.array([0.0, 0.0, -100.0]),  # 100m altitude
                velocity_body=np.array([20.0, 0.0, 0.0]),
                quaternion=Quaternion.from_euler(0.0, 0.0, 0.0),
                omega_body=np.zeros(3),
                time=0.0
            )
        
        self.controls = ControlInputs()
        self.forces_moments = ForcesAndMoments()
        self.crash_state = CrashState()
        self.history = []
        self._prev_velocity = self.state.velocity_body.copy()
    
    def _check_crash_conditions(self) -> None:
        """
        Check for crash/failure conditions.
        
        Updates self.crash_state with current warnings and crash status.
        """
        if not self.sim_config.enable_crash_detection:
            return
        
        if self.crash_state.crashed:
            return  # Already crashed
        
        dt = self.sim_config.dt
        mass = self.aircraft.mass_properties.mass
        g = 9.81
        
        # === GROUND COLLISION ===
        if self.state.altitude <= self.sim_config.ground_altitude:
            impact_velocity = np.linalg.norm(self.state.velocity_body)
            self.crash_state.crashed = True
            self.crash_state.crash_type = CrashType.GROUND_COLLISION
            self.crash_state.crash_time = self.state.time
            self.crash_state.impact_velocity = impact_velocity
            self.crash_state.crash_message = f"Ground collision at {impact_velocity:.1f} m/s"
            return
        
        # === TERRAIN WARNING ===
        self.crash_state.terrain_warning = (
            self.state.altitude < self.sim_config.terrain_warning_altitude and
            self.state.climb_rate < 0
        )
        
        # === G-FORCE / STRUCTURAL LIMITS ===
        # Compute load factor from acceleration
        dv = self.state.velocity_body - self._prev_velocity
        accel = dv / dt if dt > 0 else np.zeros(3)
        
        # Load factor: n = (L + T_vertical) / W ≈ -az/g + 1 for vertical
        # Simplified: use total acceleration magnitude relative to gravity
        total_accel = np.linalg.norm(self.forces_moments.force) / mass
        
        # More accurate: load factor from lift
        # In steady level flight, n = 1. In a 60° bank turn, n = 2.
        # n = -F_body_z / (m * g) for body-z force (lift opposes weight)
        n_load = -self.forces_moments.force[2] / (mass * g) if mass * g > 0 else 1.0
        
        self.crash_state.impact_g_force = n_load
        
        if n_load > self.sim_config.max_g_load:
            self.crash_state.crashed = True
            self.crash_state.crash_type = CrashType.STRUCTURAL_FAILURE
            self.crash_state.crash_time = self.state.time
            self.crash_state.crash_message = f"Structural failure: {n_load:.1f}g exceeded {self.sim_config.max_g_load}g limit"
            return
        
        if n_load < self.sim_config.max_negative_g:
            self.crash_state.crashed = True
            self.crash_state.crash_type = CrashType.STRUCTURAL_FAILURE
            self.crash_state.crash_time = self.state.time
            self.crash_state.crash_message = f"Structural failure: {n_load:.1f}g exceeded {self.sim_config.max_negative_g}g negative limit"
            return
        
        # === OVERSPEED ===
        airspeed = self.forces_moments.airspeed
        vne = self.aircraft.max_airspeed
        
        self.crash_state.overspeed_warning = airspeed > vne * 0.9
        
        if airspeed > vne * 1.1:  # 10% over Vne = structural failure
            self.crash_state.crashed = True
            self.crash_state.crash_type = CrashType.OVERSPEED
            self.crash_state.crash_time = self.state.time
            self.crash_state.crash_message = f"Overspeed structural failure: {airspeed:.1f} m/s > Vne {vne:.1f} m/s"
            return
        
        # === STALL DETECTION ===
        alpha_deg = np.degrees(self.forces_moments.alpha)
        stall_alpha_deg = np.degrees(self.aircraft.aero.alpha_stall)
        
        # Check if in stall (high alpha, low speed)
        in_stall = (
            abs(alpha_deg) > stall_alpha_deg or 
            airspeed < self.aircraft.min_airspeed
        )
        
        self.crash_state.stall_warning = in_stall
        
        if in_stall:
            self.crash_state.time_in_stall += dt
            
            # Unrecoverable stall after duration limit
            if self.crash_state.time_in_stall > self.sim_config.stall_duration_limit:
                # Check if also low altitude (can't recover)
                if self.state.altitude < 100:
                    self.crash_state.crashed = True
                    self.crash_state.crash_type = CrashType.STALL_SPIN
                    self.crash_state.crash_time = self.state.time
                    self.crash_state.crash_message = f"Unrecoverable stall/spin at {self.state.altitude:.0f}m altitude"
                    return
        else:
            # Reset stall timer if recovered
            self.crash_state.time_in_stall = max(0, self.crash_state.time_in_stall - dt * 2)
    
    def step(self, controls: Optional[ControlInputs] = None) -> AircraftState:
        """
        Advance simulation by one timestep.
        
        Args:
            controls: Control inputs (uses last if None)
            
        Returns:
            New aircraft state
        """
        # Don't simulate if crashed
        if self.crash_state.crashed:
            return self.state
        
        if controls is not None:
            # Clip controls to limits
            self.controls = ControlInputs(
                elevator=np.clip(controls.elevator, 
                                -self.aircraft.max_elevator, 
                                self.aircraft.max_elevator),
                aileron=np.clip(controls.aileron,
                               -self.aircraft.max_aileron,
                               self.aircraft.max_aileron),
                rudder=np.clip(controls.rudder,
                              -self.aircraft.max_rudder,
                              self.aircraft.max_rudder),
                throttle=np.clip(controls.throttle, 0.0, 1.0)
            )
        
        # Save previous velocity for acceleration calculation
        self._prev_velocity = self.state.velocity_body.copy()
        
        # Update environment
        self.environment.update(self.sim_config.dt)
        
        # Perform integration step
        self.state, self.forces_moments = rk4_step(
            self.state,
            self.controls,
            self.aircraft,
            self.environment,
            self.sim_config,
            self.sim_config.dt
        )
        
        # Check crash conditions
        self._check_crash_conditions()
        
        # Handle ground collision (if not crashed, just stop descent)
        if not self.crash_state.crashed and self.state.altitude < self.sim_config.ground_altitude:
            self.state.position[2] = -self.sim_config.ground_altitude
            self.state.velocity_body[2] = min(0.0, self.state.velocity_body[2])
        
        # Record history if enabled
        if self.record_history:
            self.history.append({
                'time': self.state.time,
                'position': self.state.position.copy(),
                'velocity': self.state.velocity_body.copy(),
                'euler': self.state.euler_angles,
                'omega': self.state.omega_body.copy(),
                'airspeed': self.forces_moments.airspeed,
                'alpha': self.forces_moments.alpha,
                'beta': self.forces_moments.beta,
                'forces': self.forces_moments.force.copy(),
                'moments': self.forces_moments.moment.copy(),
                'controls': (
                    self.controls.elevator,
                    self.controls.aileron,
                    self.controls.rudder,
                    self.controls.throttle
                ),
                'crashed': self.crash_state.crashed,
                'crash_type': self.crash_state.crash_type.name if self.crash_state.crashed else None
            })
        
        return self.state
    
    def run(
        self, 
        duration: float,
        control_callback: Optional[Callable[[AircraftState, float], ControlInputs]] = None
    ) -> list:
        """
        Run simulation for a specified duration.
        
        Args:
            duration: Simulation duration (s)
            control_callback: Optional function(state, time) -> ControlInputs
            
        Returns:
            History list of states
        """
        self.record_history = True
        end_time = self.state.time + duration
        
        while self.state.time < end_time:
            if control_callback is not None:
                controls = control_callback(self.state, self.state.time)
                self.step(controls)
            else:
                self.step()
            
            # Stop if crashed
            if self.crash_state.crashed:
                print(f"CRASH: {self.crash_state.crash_message}")
                break
            
            # Safety check
            if np.any(np.isnan(self.state.to_array())):
                print(f"Warning: NaN detected at t={self.state.time:.3f}s")
                break
        
        self.record_history = False
        return self.history
    
    def get_diagnostic_string(self) -> str:
        """Get formatted diagnostic output for debugging."""
        s = self.state
        fm = self.forces_moments
        cs = self.crash_state
        phi, theta, psi = np.degrees(s.euler_angles)
        
        status = ""
        if cs.crashed:
            status = f" | CRASHED: {cs.crash_type.name}"
        elif cs.stall_warning:
            status = " | STALL WARNING"
        elif cs.overspeed_warning:
            status = " | OVERSPEED WARNING"
        elif cs.terrain_warning:
            status = " | TERRAIN WARNING"
        
        return (
            f"t={s.time:.2f}s | "
            f"Alt={s.altitude:.1f}m | "
            f"V={fm.airspeed:.1f}m/s | "
            f"α={np.degrees(fm.alpha):.1f}° β={np.degrees(fm.beta):.1f}° | "
            f"φ={phi:.1f}° θ={theta:.1f}° ψ={psi:.1f}° | "
            f"G={cs.impact_g_force:.1f}"
            f"{status}"
        )


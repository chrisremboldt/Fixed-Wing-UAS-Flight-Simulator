"""
Trim Solver

Finds the control inputs and state that result in steady-state flight.
This is the primary validation tool - if you can't trim, the model is wrong.

Supports:
- Level flight at specified airspeed
- Climbing/descending flight
- Coordinated turns
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .state import AircraftState, ControlInputs, ForcesAndMoments
from .aircraft import AircraftConfig
from .environment import Environment
from .dynamics import state_derivative, SimulationConfig
from .frames import Quaternion


@dataclass
class TrimCondition:
    """Specification of desired trim condition."""
    
    # Target airspeed (m/s)
    airspeed: float = 25.0
    
    # Target altitude (m)
    altitude: float = 100.0
    
    # Climb rate (m/s, positive = climbing)
    climb_rate: float = 0.0
    
    # Bank angle (rad, positive = right bank)
    bank_angle: float = 0.0
    
    # Turn rate (rad/s, positive = right turn) - alternative to bank
    turn_rate: Optional[float] = None
    
    # Heading (rad, from North)
    heading: float = 0.0


@dataclass
class TrimResult:
    """Result of trim solution."""
    
    success: bool
    state: AircraftState
    controls: ControlInputs
    residuals: np.ndarray
    forces_moments: ForcesAndMoments
    iterations: int
    message: str


def compute_trim(
    condition: TrimCondition,
    aircraft: AircraftConfig,
    environment: Optional[Environment] = None,
    sim_config: Optional[SimulationConfig] = None,
    initial_guess: Optional[Dict[str, float]] = None
) -> TrimResult:
    """
    Compute trim state and controls for given condition.
    
    Solves for: alpha, throttle, elevator, (aileron, rudder if turning)
    such that all accelerations are zero.
    
    Args:
        condition: Desired trim condition
        aircraft: Aircraft configuration
        environment: Environment model
        sim_config: Simulation configuration
        initial_guess: Optional initial values for solver
        
    Returns:
        TrimResult with solution or failure info
    """
    environment = environment or Environment()
    sim_config = sim_config or SimulationConfig()
    
    # Determine if we're in a turn
    is_turning = abs(condition.bank_angle) > 0.01 or (
        condition.turn_rate is not None and abs(condition.turn_rate) > 0.01
    )
    
    if is_turning:
        return _trim_coordinated_turn(
            condition, aircraft, environment, sim_config, initial_guess
        )
    else:
        return _trim_straight_flight(
            condition, aircraft, environment, sim_config, initial_guess
        )


def _trim_straight_flight(
    condition: TrimCondition,
    aircraft: AircraftConfig,
    environment: Environment,
    sim_config: SimulationConfig,
    initial_guess: Optional[Dict[str, float]]
) -> TrimResult:
    """
    Trim for straight (non-turning) flight.
    
    Solve for: alpha, throttle, elevator
    Constraints: no angular accelerations, Fx = Fz = 0 in stability axis
    """
    
    V = condition.airspeed
    gamma = np.arcsin(np.clip(condition.climb_rate / V, -1.0, 1.0))  # Flight path angle
    psi = condition.heading
    
    # Initial guess
    if initial_guess:
        x0 = np.array([
            initial_guess.get('alpha', 0.05),
            initial_guess.get('throttle', 0.5),
            initial_guess.get('elevator', 0.0)
        ])
    else:
        x0 = np.array([0.05, 0.5, 0.0])  # alpha, throttle, elevator
    
    def residuals(x):
        alpha, throttle, elevator = x
        
        # Clamp inputs
        alpha = np.clip(alpha, -0.3, 0.5)
        throttle = np.clip(throttle, 0.0, 1.0)
        elevator = np.clip(elevator, -aircraft.max_elevator, aircraft.max_elevator)
        
        # Pitch angle = flight path + alpha
        theta = gamma + alpha
        
        # Build state
        state = AircraftState(
            position=np.array([0.0, 0.0, -condition.altitude]),
            velocity_body=np.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)]),
            quaternion=Quaternion.from_euler(0.0, theta, psi),
            omega_body=np.zeros(3),
            time=0.0
        )
        
        controls = ControlInputs(
            elevator=elevator,
            aileron=0.0,
            rudder=0.0,
            throttle=throttle
        )
        
        # Compute accelerations
        state_dot, fm = state_derivative(state, controls, aircraft, environment, sim_config)
        
        # Extract velocity derivatives (should be zero for trim)
        dv_body = state_dot[3:6]  # du, dv, dw
        domega = state_dot[10:13]  # dp, dq, dr
        
        # Residuals: accelerations should be zero
        # Weight differently: translational vs rotational
        return np.array([
            dv_body[0] * 10.0,  # Forward accel (important for speed maintenance)
            dv_body[2] * 10.0,  # Vertical accel (important for climb rate)
            domega[1] * 5.0,    # Pitch rate change (important for stability)
        ])
    
    # Solve
    result = least_squares(residuals, x0, bounds=(
        [-0.3, 0.0, -aircraft.max_elevator],
        [0.5, 1.0, aircraft.max_elevator]
    ), method='trf', ftol=1e-8)
    
    # Extract solution
    alpha, throttle, elevator = result.x
    theta = np.arcsin(condition.climb_rate / V) + alpha if V > 0 else 0.0
    
    final_state = AircraftState(
        position=np.array([0.0, 0.0, -condition.altitude]),
        velocity_body=np.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)]),
        quaternion=Quaternion.from_euler(0.0, theta, psi),
        omega_body=np.zeros(3),
        time=0.0
    )
    
    final_controls = ControlInputs(
        elevator=elevator,
        aileron=0.0,
        rudder=0.0,
        throttle=throttle
    )
    
    # Compute final forces/moments for output
    _, fm = state_derivative(final_state, final_controls, aircraft, environment, sim_config)
    
    return TrimResult(
        success=result.success,
        state=final_state,
        controls=final_controls,
        residuals=result.fun,
        forces_moments=fm,
        iterations=result.nfev,
        message=result.message if hasattr(result, 'message') else "Converged" if result.success else "Failed"
    )


def _trim_coordinated_turn(
    condition: TrimCondition,
    aircraft: AircraftConfig,
    environment: Environment,
    sim_config: SimulationConfig,
    initial_guess: Optional[Dict[str, float]]
) -> TrimResult:
    """
    Trim for coordinated turn.
    
    In a coordinated turn:
    - Bank angle phi determines turn rate: omega_z = g*tan(phi)/V
    - No sideslip (beta = 0)
    - Lift increases to support the turn: L = W/cos(phi)
    """
    
    V = condition.airspeed
    phi = condition.bank_angle
    psi = condition.heading
    g = 9.80665
    
    # Compute turn rate from bank angle (coordinated turn relation)
    # turn_rate = g * tan(phi) / V
    if condition.turn_rate is not None:
        omega_turn = condition.turn_rate
        # Compute required bank angle
        phi = np.arctan(omega_turn * V / g)
    else:
        omega_turn = g * np.tan(phi) / V if V > 0.1 else 0.0
    
    # In a coordinated turn, the aircraft has a yaw rate (r) in body frame
    # r = omega_turn * cos(theta) for small theta
    
    # Initial guess
    if initial_guess:
        x0 = np.array([
            initial_guess.get('alpha', 0.08),
            initial_guess.get('throttle', 0.6),
            initial_guess.get('elevator', -0.02),
            initial_guess.get('aileron', phi * 0.5),
            initial_guess.get('rudder', 0.0)
        ])
    else:
        # Higher alpha in turn due to increased load factor
        x0 = np.array([0.08, 0.6, -0.02, phi * 0.5, 0.0])
    
    def residuals(x):
        alpha, throttle, elevator, aileron, rudder = x
        
        # Clamp inputs
        alpha = np.clip(alpha, -0.2, 0.5)
        throttle = np.clip(throttle, 0.0, 1.0)
        elevator = np.clip(elevator, -aircraft.max_elevator, aircraft.max_elevator)
        aileron = np.clip(aileron, -aircraft.max_aileron, aircraft.max_aileron)
        rudder = np.clip(rudder, -aircraft.max_rudder, aircraft.max_rudder)
        
        theta = alpha  # Approximate: level turn
        
        # Angular velocity in body frame for coordinated turn
        # p = -omega_turn * sin(theta)
        # q = omega_turn * sin(phi) * cos(theta)  
        # r = omega_turn * cos(phi) * cos(theta)
        p = -omega_turn * np.sin(theta)
        q = omega_turn * np.sin(phi) * np.cos(theta)
        r = omega_turn * np.cos(phi) * np.cos(theta)
        
        state = AircraftState(
            position=np.array([0.0, 0.0, -condition.altitude]),
            velocity_body=np.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)]),
            quaternion=Quaternion.from_euler(phi, theta, psi),
            omega_body=np.array([p, q, r]),
            time=0.0
        )
        
        controls = ControlInputs(
            elevator=elevator,
            aileron=aileron,
            rudder=rudder,
            throttle=throttle
        )
        
        state_dot, fm = state_derivative(state, controls, aircraft, environment, sim_config)
        
        dv_body = state_dot[3:6]
        domega = state_dot[10:13]
        
        # In steady turn, angular accelerations should be zero
        # and velocity should be maintained
        return np.array([
            dv_body[0] * 10.0,   # Forward accel
            dv_body[1] * 10.0,   # Sideslip should be zero (coordinated)
            dv_body[2] * 10.0,   # Vertical accel
            domega[0] * 5.0,     # Roll rate change
            domega[1] * 5.0,     # Pitch rate change
            domega[2] * 5.0,     # Yaw rate change
        ])
    
    result = least_squares(residuals, x0, bounds=(
        [-0.2, 0.0, -aircraft.max_elevator, -aircraft.max_aileron, -aircraft.max_rudder],
        [0.5, 1.0, aircraft.max_elevator, aircraft.max_aileron, aircraft.max_rudder]
    ), method='trf', ftol=1e-8)
    
    alpha, throttle, elevator, aileron, rudder = result.x
    theta = alpha
    
    p = -omega_turn * np.sin(theta)
    q = omega_turn * np.sin(phi) * np.cos(theta)
    r = omega_turn * np.cos(phi) * np.cos(theta)
    
    final_state = AircraftState(
        position=np.array([0.0, 0.0, -condition.altitude]),
        velocity_body=np.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)]),
        quaternion=Quaternion.from_euler(phi, theta, psi),
        omega_body=np.array([p, q, r]),
        time=0.0
    )
    
    final_controls = ControlInputs(
        elevator=elevator,
        aileron=aileron,
        rudder=rudder,
        throttle=throttle
    )
    
    _, fm = state_derivative(final_state, final_controls, aircraft, environment, sim_config)
    
    return TrimResult(
        success=result.success,
        state=final_state,
        controls=final_controls,
        residuals=result.fun,
        forces_moments=fm,
        iterations=result.nfev,
        message="Coordinated turn trim " + ("succeeded" if result.success else "failed")
    )


def validate_turn_rate(airspeed: float, bank_angle: float) -> float:
    """
    Compute expected turn rate for coordinated turn.
    
    This is a physics sanity check - the simulation should match this.
    
    Args:
        airspeed: True airspeed (m/s)
        bank_angle: Bank angle (rad)
        
    Returns:
        Turn rate (rad/s)
    """
    g = 9.80665
    if airspeed < 1.0:
        return 0.0
    return g * np.tan(bank_angle) / airspeed


def validate_stall_speed(aircraft: AircraftConfig, weight: Optional[float] = None) -> float:
    """
    Estimate stall speed at 1g.
    
    Args:
        aircraft: Aircraft configuration
        weight: Weight in N (uses mass*g if not provided)
        
    Returns:
        Stall speed (m/s)
    """
    if weight is None:
        weight = aircraft.mass_properties.mass * 9.80665
    
    rho = 1.225  # Sea level
    CL_max = aircraft.aero.CL_max
    S = aircraft.wing_area
    
    # V_stall = sqrt(2*W / (rho * S * CL_max))
    return np.sqrt(2 * weight / (rho * S * CL_max))


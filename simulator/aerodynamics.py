"""
Aerodynamics Module

Computes aerodynamic forces and moments based on:
- Flight condition (airspeed, α, β)
- Control surface deflections
- Angular rates
- Dynamic pressure

Forces are computed in the wind frame and then rotated to body frame.
Moments are computed directly in body frame about the CG.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .state import AircraftState, ControlInputs
from .aircraft import AircraftConfig
from .environment import AtmosphericProperties
from .frames import wind_angles, wind_to_body_dcm


@dataclass
class AeroState:
    """Aerodynamic state variables for a single computation."""
    
    # True airspeed (m/s)
    airspeed: float = 0.0
    
    # Angles (rad)
    alpha: float = 0.0
    beta: float = 0.0
    
    # Dynamic pressure (Pa)
    q_bar: float = 0.0
    
    # Mach number
    mach: float = 0.0
    
    # Nondimensional rates
    p_hat: float = 0.0  # pb/(2V)
    q_hat: float = 0.0  # qc/(2V)
    r_hat: float = 0.0  # rb/(2V)
    
    # Coefficients (for logging/debugging)
    CL: float = 0.0
    CD: float = 0.0
    CY: float = 0.0
    Cl: float = 0.0
    Cm: float = 0.0
    Cn: float = 0.0


def compute_aero_state(
    state: AircraftState,
    wind_body: np.ndarray,
    atmosphere: AtmosphericProperties,
    config: AircraftConfig
) -> AeroState:
    """
    Compute aerodynamic state variables from flight state.
    
    Args:
        state: Current aircraft state
        wind_body: Wind velocity in body frame (m/s)
        atmosphere: Current atmospheric properties
        config: Aircraft configuration
        
    Returns:
        AeroState with computed values
    """
    # Air-relative velocity in body frame
    v_air_body = state.velocity_body - wind_body
    
    # Compute airspeed, alpha, beta
    V, alpha, beta = wind_angles(v_air_body)
    
    # Dynamic pressure
    q_bar = 0.5 * atmosphere.density * V**2
    
    # Mach number
    mach = V / atmosphere.speed_of_sound if atmosphere.speed_of_sound > 0 else 0.0
    
    # Nondimensional angular rates
    if V > 1.0:  # Avoid division by small numbers
        p_hat = state.p * config.wing_span / (2 * V)
        q_hat = state.q * config.mean_chord / (2 * V)
        r_hat = state.r * config.wing_span / (2 * V)
    else:
        p_hat = q_hat = r_hat = 0.0
    
    return AeroState(
        airspeed=V,
        alpha=alpha,
        beta=beta,
        q_bar=q_bar,
        mach=mach,
        p_hat=p_hat,
        q_hat=q_hat,
        r_hat=r_hat
    )


def compute_aero_forces_moments(
    aero_state: AeroState,
    controls: ControlInputs,
    config: AircraftConfig
) -> Tuple[np.ndarray, np.ndarray, AeroState]:
    """
    Compute aerodynamic forces and moments.
    
    Forces are returned in body frame.
    Moments are about the CG in body frame.
    
    Args:
        aero_state: Current aerodynamic state
        controls: Control surface deflections
        config: Aircraft configuration
        
    Returns:
        (forces_body, moments_body, updated_aero_state)
    """
    aero = config.aero
    
    alpha = aero_state.alpha
    beta = aero_state.beta
    q_bar = aero_state.q_bar
    p_hat = aero_state.p_hat
    q_hat = aero_state.q_hat
    r_hat = aero_state.r_hat
    
    # Get coefficients
    CL = aero.get_CL(alpha, q_hat, controls.elevator)
    CD = aero.get_CD(CL, alpha, beta)
    CY = aero.get_CY(beta, p_hat, r_hat, controls.rudder)
    Cl = aero.get_Cl(beta, p_hat, r_hat, controls.aileron, controls.rudder)
    Cm = aero.get_Cm(alpha, q_hat, controls.elevator)
    Cn = aero.get_Cn(beta, p_hat, r_hat, controls.aileron, controls.rudder)
    
    # Update aero state with coefficients for logging
    aero_state.CL = CL
    aero_state.CD = CD
    aero_state.CY = CY
    aero_state.Cl = Cl
    aero_state.Cm = Cm
    aero_state.Cn = Cn
    
    # Reference values
    S = config.wing_area
    b = config.wing_span
    c = config.mean_chord
    
    # Forces in wind axes (lift is +Z_wind, drag is -X_wind in wind frame)
    # Wind frame: X along velocity, Z perpendicular (in vertical plane), Y to right
    F_wind = np.array([
        -CD * q_bar * S,  # Drag (negative X = opposing motion)
         CY * q_bar * S,  # Side force (Y)
        -CL * q_bar * S   # Lift (negative Z in wind frame = upward relative to velocity)
    ])
    
    # Rotate forces from wind frame to body frame
    R_wind_to_body = wind_to_body_dcm(alpha, beta)
    F_body = R_wind_to_body @ F_wind
    
    # Moments directly in body frame
    M_body = np.array([
        Cl * q_bar * S * b,   # Rolling moment (about X)
        Cm * q_bar * S * c,   # Pitching moment (about Y)
        Cn * q_bar * S * b    # Yawing moment (about Z)
    ])
    
    return F_body, M_body, aero_state


def compute_thrust_forces_moments(
    airspeed: float,
    controls: ControlInputs,
    atmosphere: AtmosphericProperties,
    config: AircraftConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute propulsion forces and moments.
    
    Args:
        airspeed: True airspeed (m/s)
        controls: Control inputs (throttle)
        atmosphere: Current atmospheric properties
        config: Aircraft configuration
        
    Returns:
        (thrust_force_body, thrust_moment_body)
    """
    prop = config.propulsion
    
    # Density ratio for altitude correction
    rho_ratio = atmosphere.density / 1.225
    
    # Compute thrust magnitude
    T = prop.get_thrust(controls.throttle, airspeed, rho_ratio)
    
    # Thrust vector in body frame
    F_thrust = T * prop.thrust_direction
    
    # Moment from thrust offset (if thrust line doesn't pass through CG)
    # M = r × F where r is position of thrust application point relative to CG
    r_thrust = prop.thrust_position - config.mass_properties.cg_position
    M_thrust = np.cross(r_thrust, F_thrust)
    
    # Add propeller torque (reaction to spinning prop)
    # Positive thrust creates negative rolling moment (for clockwise prop from rear)
    M_prop_torque = np.array([-T * prop.propeller_torque_factor, 0.0, 0.0])
    
    M_thrust += M_prop_torque
    
    return F_thrust, M_thrust


def compute_gravity_force(
    mass: float,
    gravity_body: np.ndarray
) -> np.ndarray:
    """
    Compute gravity force in body frame.
    
    Args:
        mass: Aircraft mass (kg)
        gravity_body: Gravity vector in body frame (m/s²)
        
    Returns:
        Gravity force in body frame (N)
    """
    return mass * gravity_body


"""
Aircraft Configuration

Defines the physical properties of the aircraft:
- Mass and inertia
- Aerodynamic reference dimensions
- Aerodynamic coefficients and derivatives
- Propulsion characteristics

This is the data-driven core that allows different aircraft to be simulated.
"""

import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
from pathlib import Path


@dataclass
class MassProperties:
    """Aircraft mass and inertia properties."""
    
    mass: float = 25.0  # kg
    
    # Moments of inertia about body axes through CG (kg·m²)
    Ixx: float = 1.0
    Iyy: float = 2.0
    Izz: float = 2.5
    
    # Products of inertia (typically small for symmetric aircraft)
    Ixz: float = 0.1
    Ixy: float = 0.0
    Iyz: float = 0.0
    
    # CG position relative to reference point (body frame, m)
    cg_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        self.cg_position = np.asarray(self.cg_position, dtype=np.float64)
    
    @property
    def inertia_tensor(self) -> np.ndarray:
        """Full 3x3 inertia tensor."""
        return np.array([
            [self.Ixx, -self.Ixy, -self.Ixz],
            [-self.Ixy, self.Iyy, -self.Iyz],
            [-self.Ixz, -self.Iyz, self.Izz]
        ])
    
    @property
    def inertia_inverse(self) -> np.ndarray:
        """Inverse of inertia tensor (pre-computed for efficiency)."""
        return np.linalg.inv(self.inertia_tensor)


@dataclass
class AeroCoefficients:
    """
    Aerodynamic coefficient derivatives.
    
    These define how forces and moments vary with:
    - α (angle of attack)
    - β (sideslip)
    - Control deflections (δe, δa, δr)
    - Nondimensional rates (p̂, q̂, r̂)
    
    Positive conventions:
    - CL: Lift coefficient, positive up
    - CD: Drag coefficient, positive aft
    - CY: Side force coefficient, positive right
    - Cl: Roll moment, positive right wing down
    - Cm: Pitch moment, positive nose up
    - Cn: Yaw moment, positive nose right
    """
    
    # Lift
    CL0: float = 0.1        # Zero-alpha lift
    CLa: float = 5.0        # Lift curve slope (per rad)
    CLq: float = 5.0        # Lift due to pitch rate (per rad)
    CLde: float = 0.5       # Lift due to elevator (per rad)
    
    # Stall model
    alpha_stall: float = 0.26   # ~15 degrees
    CL_max: float = 1.4         # Maximum CL at stall
    CL_stall_drop: float = 0.3  # CL reduction post-stall
    
    # Drag
    CD0: float = 0.03       # Parasitic drag
    CD_k: float = 0.04      # Induced drag factor (CD = CD0 + k*CL²)
    CDa: float = 0.1        # Drag increase with alpha (per rad²)
    CDb: float = 0.2        # Drag increase with sideslip (per rad²)
    
    # Side force
    CYb: float = -0.5       # Side force due to sideslip (per rad)
    CYdr: float = 0.2       # Side force due to rudder (per rad)
    CYp: float = 0.0        # Side force due to roll rate
    CYr: float = 0.3        # Side force due to yaw rate
    
    # Rolling moment
    Clb: float = -0.1       # Dihedral effect (per rad)
    Clp: float = -0.5       # Roll damping (per rad)
    Clr: float = 0.1        # Roll due to yaw rate
    Clda: float = 0.15      # Aileron effectiveness (per rad)
    Cldr: float = 0.01      # Roll due to rudder
    
    # Pitching moment
    Cm0: float = 0.05       # Zero-alpha pitching moment
    Cma: float = -0.8       # Pitch stiffness (static margin, per rad)
    Cmq: float = -15.0      # Pitch damping (per rad)
    Cmde: float = -1.2      # Elevator effectiveness (per rad)
    
    # Yawing moment
    Cnb: float = 0.1        # Weathercock stability (per rad)
    Cnp: float = -0.03      # Yaw due to roll rate
    Cnr: float = -0.2       # Yaw damping (per rad)
    Cnda: float = 0.01      # Adverse yaw (per rad)
    Cndr: float = -0.1      # Rudder effectiveness (per rad)
    
    def get_CL(self, alpha: float, q_hat: float, delta_e: float) -> float:
        """
        Compute lift coefficient with stall modeling.
        
        Args:
            alpha: Angle of attack (rad)
            q_hat: Nondimensional pitch rate = q*c/(2V)
            delta_e: Elevator deflection (rad)
            
        Returns:
            CL coefficient
        """
        # Linear region
        CL_linear = self.CL0 + self.CLa * alpha + self.CLq * q_hat + self.CLde * delta_e
        
        # Simple stall model: smooth transition after stall angle
        if abs(alpha) <= self.alpha_stall:
            return CL_linear
        else:
            # Post-stall: CL drops off
            alpha_excess = abs(alpha) - self.alpha_stall
            stall_factor = np.exp(-5.0 * alpha_excess)
            CL_post_stall = self.CL_max * stall_factor * np.sign(alpha)
            
            # Smooth blend
            blend = np.tanh(10.0 * alpha_excess)
            return CL_linear * (1 - blend) + CL_post_stall * blend
    
    def get_CD(self, CL: float, alpha: float, beta: float) -> float:
        """
        Compute drag coefficient (parabolic polar + additional terms).
        
        Args:
            CL: Current lift coefficient
            alpha: Angle of attack (rad)
            beta: Sideslip angle (rad)
            
        Returns:
            CD coefficient
        """
        CD_induced = self.CD_k * CL**2
        CD_alpha = self.CDa * alpha**2
        CD_beta = self.CDb * beta**2
        
        return self.CD0 + CD_induced + CD_alpha + CD_beta
    
    def get_CY(self, beta: float, p_hat: float, r_hat: float, 
               delta_r: float) -> float:
        """Compute side force coefficient."""
        return (self.CYb * beta + 
                self.CYp * p_hat + 
                self.CYr * r_hat + 
                self.CYdr * delta_r)
    
    def get_Cl(self, beta: float, p_hat: float, r_hat: float,
               delta_a: float, delta_r: float) -> float:
        """Compute rolling moment coefficient."""
        return (self.Clb * beta +
                self.Clp * p_hat +
                self.Clr * r_hat +
                self.Clda * delta_a +
                self.Cldr * delta_r)
    
    def get_Cm(self, alpha: float, q_hat: float, delta_e: float) -> float:
        """Compute pitching moment coefficient."""
        return (self.Cm0 +
                self.Cma * alpha +
                self.Cmq * q_hat +
                self.Cmde * delta_e)
    
    def get_Cn(self, beta: float, p_hat: float, r_hat: float,
               delta_a: float, delta_r: float) -> float:
        """Compute yawing moment coefficient."""
        return (self.Cnb * beta +
                self.Cnp * p_hat +
                self.Cnr * r_hat +
                self.Cnda * delta_a +
                self.Cndr * delta_r)


@dataclass
class PropulsionProperties:
    """Propulsion system characteristics."""
    
    # Maximum static thrust at sea level (N)
    max_thrust: float = 100.0
    
    # Thrust decrease with airspeed (linear model)
    # T = max_thrust * throttle * (1 - airspeed / V_max_thrust)
    thrust_velocity_factor: float = 0.01  # (per m/s)
    
    # Thrust line offset from CG (body frame, m)
    thrust_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Thrust direction in body frame (unit vector)
    thrust_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    
    # Propeller effects (optional)
    propeller_torque_factor: float = 0.02  # Torque as fraction of thrust
    
    def __post_init__(self):
        self.thrust_position = np.asarray(self.thrust_position, dtype=np.float64)
        self.thrust_direction = np.asarray(self.thrust_direction, dtype=np.float64)
        # Normalize thrust direction
        norm = np.linalg.norm(self.thrust_direction)
        if norm > 1e-6:
            self.thrust_direction /= norm
    
    def get_thrust(self, throttle: float, airspeed: float, 
                   density_ratio: float = 1.0) -> float:
        """
        Compute thrust magnitude.
        
        Args:
            throttle: Throttle setting [0, 1]
            airspeed: True airspeed (m/s)
            density_ratio: ρ/ρ_0 for altitude correction
            
        Returns:
            Thrust force (N)
        """
        throttle = np.clip(throttle, 0.0, 1.0)
        
        # Base thrust
        T = self.max_thrust * throttle
        
        # Airspeed reduction
        velocity_factor = max(0.0, 1.0 - self.thrust_velocity_factor * airspeed)
        T *= velocity_factor
        
        # Density correction (prop efficiency increases with density)
        T *= np.sqrt(density_ratio)
        
        return T


@dataclass
class AircraftConfig:
    """Complete aircraft configuration."""
    
    name: str = "Generic UAS"
    
    # Reference dimensions
    wing_span: float = 3.0      # b (m)
    wing_area: float = 0.8      # S (m²)
    mean_chord: float = 0.27    # c̄ (m)
    
    # Component properties
    mass_properties: MassProperties = field(default_factory=MassProperties)
    aero: AeroCoefficients = field(default_factory=AeroCoefficients)
    propulsion: PropulsionProperties = field(default_factory=PropulsionProperties)
    
    # Control limits (rad)
    max_elevator: float = 0.44  # ~25°
    max_aileron: float = 0.44
    max_rudder: float = 0.44
    
    # Performance limits
    max_airspeed: float = 50.0  # m/s
    min_airspeed: float = 12.0  # m/s (stall speed approx)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'AircraftConfig':
        """Load aircraft configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'AircraftConfig':
        """Create config from dictionary."""
        mass_data = data.get('mass_properties', {})
        aero_data = data.get('aerodynamics', {})
        prop_data = data.get('propulsion', {})
        
        return cls(
            name=data.get('name', 'Unknown'),
            wing_span=data.get('wing_span', 3.0),
            wing_area=data.get('wing_area', 0.8),
            mean_chord=data.get('mean_chord', 0.27),
            mass_properties=MassProperties(**mass_data) if mass_data else MassProperties(),
            aero=AeroCoefficients(**aero_data) if aero_data else AeroCoefficients(),
            propulsion=PropulsionProperties(**prop_data) if prop_data else PropulsionProperties(),
            max_elevator=data.get('max_elevator', 0.44),
            max_aileron=data.get('max_aileron', 0.44),
            max_rudder=data.get('max_rudder', 0.44),
            max_airspeed=data.get('max_airspeed', 50.0),
            min_airspeed=data.get('min_airspeed', 12.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'name': self.name,
            'wing_span': self.wing_span,
            'wing_area': self.wing_area,
            'mean_chord': self.mean_chord,
            'mass_properties': {
                'mass': self.mass_properties.mass,
                'Ixx': self.mass_properties.Ixx,
                'Iyy': self.mass_properties.Iyy,
                'Izz': self.mass_properties.Izz,
                'Ixz': self.mass_properties.Ixz,
            },
            'aerodynamics': {
                'CL0': self.aero.CL0,
                'CLa': self.aero.CLa,
                'CD0': self.aero.CD0,
                'CD_k': self.aero.CD_k,
                'Cma': self.aero.Cma,
                'Cmq': self.aero.Cmq,
                'Cmde': self.aero.Cmde,
                # ... other coefficients
            },
            'propulsion': {
                'max_thrust': self.propulsion.max_thrust,
            }
        }
    
    def save_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


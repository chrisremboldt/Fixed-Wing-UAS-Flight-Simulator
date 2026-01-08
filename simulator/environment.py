"""
Environment Model

Provides atmospheric properties (density, pressure, temperature, speed of sound)
and wind/turbulence models.

Uses the International Standard Atmosphere (ISA 1976) model for properties
varying with altitude.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ISA Constants at sea level
ISA_T0 = 288.15      # Temperature (K)
ISA_P0 = 101325.0    # Pressure (Pa)
ISA_RHO0 = 1.225     # Density (kg/m³)
ISA_A0 = 340.29      # Speed of sound (m/s)
ISA_G0 = 9.80665     # Standard gravity (m/s²)
ISA_R = 287.05       # Specific gas constant for air (J/(kg·K))
ISA_GAMMA = 1.4      # Ratio of specific heats

# Lapse rate in troposphere (K/m)
ISA_LAPSE_RATE = 0.0065

# Tropopause altitude (m)
ISA_TROPOPAUSE = 11000.0


@dataclass
class AtmosphericProperties:
    """Atmospheric properties at a given altitude."""
    density: float          # kg/m³
    pressure: float         # Pa
    temperature: float      # K
    speed_of_sound: float   # m/s
    gravity: float          # m/s²
    

def isa_atmosphere(altitude: float) -> AtmosphericProperties:
    """
    Compute atmospheric properties using ISA 1976 model.
    
    Args:
        altitude: Geometric altitude above sea level (m)
        
    Returns:
        AtmosphericProperties at the given altitude
    """
    # Clamp to valid range
    h = max(0.0, min(altitude, 86000.0))
    
    if h <= ISA_TROPOPAUSE:
        # Troposphere: temperature decreases linearly
        T = ISA_T0 - ISA_LAPSE_RATE * h
        P = ISA_P0 * (T / ISA_T0) ** (ISA_G0 / (ISA_LAPSE_RATE * ISA_R))
    else:
        # Stratosphere (simplified): isothermal layer
        T_tropo = ISA_T0 - ISA_LAPSE_RATE * ISA_TROPOPAUSE
        P_tropo = ISA_P0 * (T_tropo / ISA_T0) ** (ISA_G0 / (ISA_LAPSE_RATE * ISA_R))
        T = T_tropo  # Constant temperature
        P = P_tropo * np.exp(-ISA_G0 * (h - ISA_TROPOPAUSE) / (ISA_R * T))
    
    # Density from ideal gas law
    rho = P / (ISA_R * T)
    
    # Speed of sound
    a = np.sqrt(ISA_GAMMA * ISA_R * T)
    
    # Gravity variation with altitude (approximate)
    g = ISA_G0 * (6371000.0 / (6371000.0 + h)) ** 2
    
    return AtmosphericProperties(
        density=rho,
        pressure=P,
        temperature=T,
        speed_of_sound=a,
        gravity=g
    )


@dataclass
class WindModel:
    """
    Wind and turbulence model.
    
    Provides steady wind plus optional turbulence/gusts.
    Wind is specified in NED frame.
    """
    
    # Steady wind components in NED frame (m/s)
    wind_ned: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Turbulence intensity (0 = none, 1 = severe)
    turbulence_intensity: float = 0.0
    
    # Gust parameters (for discrete gusts)
    gust_active: bool = False
    gust_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gust_duration: float = 0.0
    gust_elapsed: float = 0.0
    
    # Dryden turbulence state (for continuous turbulence)
    _turbulence_state: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _last_time: float = 0.0
    
    def __post_init__(self):
        self.wind_ned = np.asarray(self.wind_ned, dtype=np.float64)
        self.gust_velocity = np.asarray(self.gust_velocity, dtype=np.float64)
        self._turbulence_state = np.asarray(self._turbulence_state, dtype=np.float64)
    
    def get_wind_ned(self, position: np.ndarray, time: float) -> np.ndarray:
        """
        Get total wind velocity in NED frame at given position and time.
        
        Args:
            position: Position in NED frame (m)
            time: Simulation time (s)
            
        Returns:
            Wind velocity in NED frame (m/s)
        """
        wind = self.wind_ned.copy()
        
        # Add turbulence if enabled
        if self.turbulence_intensity > 0:
            wind += self._get_turbulence(position, time)
        
        # Add discrete gust if active
        if self.gust_active and self.gust_elapsed < self.gust_duration:
            # 1-cosine gust shape
            gust_fraction = 0.5 * (1 - np.cos(2 * np.pi * self.gust_elapsed / self.gust_duration))
            wind += self.gust_velocity * gust_fraction
        
        return wind
    
    def _get_turbulence(self, position: np.ndarray, time: float) -> np.ndarray:
        """
        Simple turbulence model using filtered white noise.
        
        For more realism, implement full Dryden or von Kármán model.
        """
        dt = time - self._last_time
        if dt <= 0:
            return self._turbulence_state
        
        self._last_time = time
        
        # Time constant for turbulence (larger = slower changes)
        tau = 2.0
        
        # Turbulence magnitude based on intensity
        sigma = self.turbulence_intensity * 3.0  # m/s at intensity=1
        
        # First-order filter with white noise input
        alpha = np.exp(-dt / tau)
        noise = np.random.randn(3) * sigma * np.sqrt(1 - alpha**2)
        self._turbulence_state = alpha * self._turbulence_state + noise
        
        return self._turbulence_state
    
    def set_steady_wind(self, speed: float, direction_from: float):
        """
        Set steady wind from speed and direction.
        
        Args:
            speed: Wind speed (m/s)
            direction_from: Direction wind is coming FROM, clockwise from North (rad)
        """
        # Wind FROM a direction means it's blowing toward the opposite
        self.wind_ned[0] = -speed * np.cos(direction_from)  # North component
        self.wind_ned[1] = -speed * np.sin(direction_from)  # East component
        self.wind_ned[2] = 0.0  # Vertical wind
    
    def trigger_gust(self, velocity_ned: np.ndarray, duration: float):
        """
        Trigger a discrete gust event.
        
        Args:
            velocity_ned: Peak gust velocity in NED frame (m/s)
            duration: Gust duration (s)
        """
        self.gust_active = True
        self.gust_velocity = np.asarray(velocity_ned)
        self.gust_duration = duration
        self.gust_elapsed = 0.0
    
    def update(self, dt: float):
        """Update wind model state (e.g., gust progression)."""
        if self.gust_active:
            self.gust_elapsed += dt
            if self.gust_elapsed >= self.gust_duration:
                self.gust_active = False
                self.gust_elapsed = 0.0


@dataclass
class Environment:
    """
    Complete environment model combining atmosphere and wind.
    """
    
    # Reference altitude for relative calculations (m MSL)
    reference_altitude: float = 0.0
    
    # Wind model
    wind: WindModel = field(default_factory=WindModel)
    
    # Use constant atmosphere (for simplified sims)
    constant_density: Optional[float] = None
    
    def get_atmosphere(self, altitude: float) -> AtmosphericProperties:
        """
        Get atmospheric properties at altitude.
        
        Args:
            altitude: Altitude above reference (m)
            
        Returns:
            AtmosphericProperties
        """
        if self.constant_density is not None:
            return AtmosphericProperties(
                density=self.constant_density,
                pressure=ISA_P0,
                temperature=ISA_T0,
                speed_of_sound=ISA_A0,
                gravity=ISA_G0
            )
        
        return isa_atmosphere(self.reference_altitude + altitude)
    
    def get_wind_body(self, position: np.ndarray, time: float, 
                      dcm_ned_to_body: np.ndarray) -> np.ndarray:
        """
        Get wind velocity in body frame.
        
        Args:
            position: Position in NED frame (m)
            time: Simulation time (s)
            dcm_ned_to_body: Rotation matrix from NED to body frame
            
        Returns:
            Wind velocity in body frame (m/s)
        """
        wind_ned = self.wind.get_wind_ned(position, time)
        return dcm_ned_to_body @ wind_ned
    
    def get_gravity_body(self, altitude: float, 
                         dcm_ned_to_body: np.ndarray) -> np.ndarray:
        """
        Get gravity vector in body frame.
        
        Args:
            altitude: Altitude above reference (m)
            dcm_ned_to_body: Rotation matrix from NED to body frame
            
        Returns:
            Gravity acceleration in body frame (m/s²)
        """
        atm = self.get_atmosphere(altitude)
        g_ned = np.array([0.0, 0.0, atm.gravity])  # Down is positive in NED
        return dcm_ned_to_body @ g_ned
    
    def update(self, dt: float):
        """Update environment state."""
        self.wind.update(dt)


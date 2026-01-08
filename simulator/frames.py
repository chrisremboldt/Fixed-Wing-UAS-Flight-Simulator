"""
Coordinate Frame Transformations

This module handles all rotations between reference frames:
- NED (North-East-Down): Inertial frame, gravity defines Down
- Body: X forward (nose), Y right (starboard), Z down (belly)  
- Wind/Stability: Aligned with relative airflow

We use quaternions internally to avoid gimbal lock, with utilities
to convert to/from Euler angles for human readability.

Convention: quaternion q = [w, x, y, z] where w is the scalar part
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Quaternion:
    """
    Unit quaternion for attitude representation.
    q = w + xi + yj + zk, stored as [w, x, y, z]
    
    Represents rotation from Body frame to NED frame.
    """
    w: float
    x: float
    y: float
    z: float
    
    def __post_init__(self):
        """Normalize on creation to ensure unit quaternion."""
        self._normalize()
    
    def _normalize(self):
        """Normalize to unit quaternion."""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 1e-10:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
    
    @classmethod
    def from_euler(cls, phi: float, theta: float, psi: float) -> 'Quaternion':
        """
        Create quaternion from Euler angles (3-2-1 rotation sequence).
        
        Args:
            phi: Roll angle (rad) - rotation about body X
            theta: Pitch angle (rad) - rotation about body Y  
            psi: Yaw angle (rad) - rotation about body Z
            
        Returns:
            Quaternion representing the rotation
        """
        # Half angles
        cr = np.cos(phi / 2)
        sr = np.sin(phi / 2)
        cp = np.cos(theta / 2)
        sp = np.sin(theta / 2)
        cy = np.cos(psi / 2)
        sy = np.sin(psi / 2)
        
        # Quaternion from 3-2-1 Euler sequence
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return cls(w, x, y, z)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        """Create quaternion from numpy array [w, x, y, z]."""
        return cls(arr[0], arr[1], arr[2], arr[3])
    
    def to_array(self) -> np.ndarray:
        """Return quaternion as numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])
    
    def to_euler(self) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (3-2-1 sequence).
        
        Returns:
            (phi, theta, psi) - roll, pitch, yaw in radians
            
        Note: Has singularity at theta = ±90° (gimbal lock).
              Use quaternion directly for computations.
        """
        # Roll (phi)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x**2 + self.y**2)
        phi = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (theta) - handle gimbal lock
        sinp = 2 * (self.w * self.y - self.z * self.x)
        sinp = np.clip(sinp, -1.0, 1.0)  # Numerical safety
        theta = np.arcsin(sinp)
        
        # Yaw (psi)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y**2 + self.z**2)
        psi = np.arctan2(siny_cosp, cosy_cosp)
        
        return phi, theta, psi
    
    def to_dcm(self) -> np.ndarray:
        """
        Convert to Direction Cosine Matrix (rotation matrix).
        
        Returns:
            3x3 rotation matrix R_body_to_ned
            v_ned = R @ v_body
        """
        w, x, y, z = self.w, self.x, self.y, self.z
        
        return np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    def conjugate(self) -> 'Quaternion':
        """Return conjugate (inverse for unit quaternions)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication (Hamilton product)."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector from body frame to NED frame.
        
        Args:
            v: 3D vector in body frame
            
        Returns:
            3D vector in NED frame
        """
        # Using DCM is clearer and fast enough
        return self.to_dcm() @ v
    
    def inverse_rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector from NED frame to body frame.
        
        Args:
            v: 3D vector in NED frame
            
        Returns:
            3D vector in body frame
        """
        return self.to_dcm().T @ v


def quaternion_derivative(q: Quaternion, omega_body: np.ndarray) -> np.ndarray:
    """
    Compute quaternion time derivative from angular velocity.
    
    dq/dt = 0.5 * Omega(omega) * q
    
    Args:
        q: Current quaternion (body to NED)
        omega_body: Angular velocity [p, q, r] in body frame (rad/s)
        
    Returns:
        Quaternion derivative as array [dw, dx, dy, dz]
    """
    p, qb, r = omega_body  # Note: qb to avoid confusion with quaternion q
    
    # Omega matrix
    omega_matrix = np.array([
        [0, -p, -qb, -r],
        [p,  0,  r, -qb],
        [qb, -r,  0,  p],
        [r, qb, -p,  0]
    ])
    
    q_arr = q.to_array()
    return 0.5 * omega_matrix @ q_arr


def wind_angles(v_body: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute airspeed, angle of attack, and sideslip from body velocity.
    
    Args:
        v_body: Velocity in body frame [u, v, w] (m/s)
                u = forward, v = right, w = down
                
    Returns:
        (V, alpha, beta):
            V: True airspeed (m/s)
            alpha: Angle of attack (rad)
            beta: Sideslip angle (rad)
    """
    u, v, w = v_body
    
    V = np.sqrt(u**2 + v**2 + w**2)
    
    if V < 1e-6:
        return 0.0, 0.0, 0.0
    
    # Angle of attack: angle in the vertical plane
    # alpha = atan2(w, u) - positive when nose up relative to airflow
    alpha = np.arctan2(w, u)
    
    # Sideslip: angle in the horizontal plane
    # beta = asin(v / V) - positive when wind from right
    beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
    
    return V, alpha, beta


def body_to_wind_dcm(alpha: float, beta: float) -> np.ndarray:
    """
    Rotation matrix from body frame to wind frame.
    
    The wind frame has X aligned with the velocity vector.
    
    Args:
        alpha: Angle of attack (rad)
        beta: Sideslip angle (rad)
        
    Returns:
        3x3 rotation matrix
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    
    # First rotate by -alpha about Y (pitch down to align with wind)
    # Then rotate by beta about Z (yaw into wind)
    return np.array([
        [ ca*cb, sb, sa*cb],
        [-ca*sb, cb, -sa*sb],
        [-sa,    0,  ca]
    ])


def wind_to_body_dcm(alpha: float, beta: float) -> np.ndarray:
    """
    Rotation matrix from wind frame to body frame.
    
    Args:
        alpha: Angle of attack (rad)
        beta: Sideslip angle (rad)
        
    Returns:
        3x3 rotation matrix (transpose of body_to_wind)
    """
    return body_to_wind_dcm(alpha, beta).T


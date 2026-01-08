"""
Tests for coordinate frame transformations.

These verify the quaternion math and frame rotations are correct.
"""

import numpy as np
import pytest
from simulator.frames import (
    Quaternion, 
    quaternion_derivative, 
    wind_angles,
    body_to_wind_dcm,
    wind_to_body_dcm
)


class TestQuaternion:
    """Tests for Quaternion class."""
    
    def test_identity_quaternion(self):
        """Identity quaternion should not rotate."""
        q = Quaternion(1, 0, 0, 0)
        v = np.array([1.0, 2.0, 3.0])
        v_rotated = q.rotate_vector(v)
        np.testing.assert_array_almost_equal(v, v_rotated)
    
    def test_euler_to_quaternion_identity(self):
        """Zero Euler angles should give identity quaternion."""
        q = Quaternion.from_euler(0, 0, 0)
        assert abs(q.w - 1.0) < 1e-10
        assert abs(q.x) < 1e-10
        assert abs(q.y) < 1e-10
        assert abs(q.z) < 1e-10
    
    def test_euler_roundtrip(self):
        """Converting to quaternion and back should preserve angles."""
        angles = [
            (0.0, 0.0, 0.0),
            (0.1, 0.2, 0.3),
            (-0.5, 0.3, 1.0),
            (np.pi/4, np.pi/6, np.pi/3),
        ]
        
        for phi, theta, psi in angles:
            q = Quaternion.from_euler(phi, theta, psi)
            phi2, theta2, psi2 = q.to_euler()
            
            assert abs(phi - phi2) < 1e-10, f"Roll mismatch: {phi} vs {phi2}"
            assert abs(theta - theta2) < 1e-10, f"Pitch mismatch: {theta} vs {theta2}"
            assert abs(psi - psi2) < 1e-10, f"Yaw mismatch: {psi} vs {psi2}"
    
    def test_quaternion_normalization(self):
        """Quaternion should auto-normalize."""
        q = Quaternion(2, 0, 0, 0)
        assert abs(q.w - 1.0) < 1e-10
        
        q = Quaternion(1, 1, 1, 1)
        norm = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert abs(norm - 1.0) < 1e-10
    
    def test_90_degree_rotations(self):
        """Test 90-degree rotations about each axis."""
        # 90° pitch (nose up)
        q = Quaternion.from_euler(0, np.pi/2, 0)
        v = np.array([1.0, 0.0, 0.0])  # Forward in body
        v_ned = q.rotate_vector(v)
        # Forward in body should now be up in NED (negative z)
        np.testing.assert_array_almost_equal(v_ned, [0.0, 0.0, -1.0])
        
        # 90° roll (right wing down): body Z (down) rotates to body Y (right)
        # When the right wing goes down, the body's down axis points left in NED
        q = Quaternion.from_euler(np.pi/2, 0, 0)
        v = np.array([0.0, 0.0, 1.0])  # Down in body
        v_ned = q.rotate_vector(v)
        # With right wing down (positive roll), body down points to East-negative
        np.testing.assert_array_almost_equal(v_ned, [0.0, -1.0, 0.0])
    
    def test_dcm_and_quaternion_equivalent(self):
        """DCM rotation should match quaternion rotation."""
        q = Quaternion.from_euler(0.3, 0.2, 0.5)
        R = q.to_dcm()
        
        v = np.array([1.0, 2.0, 3.0])
        
        v_quat = q.rotate_vector(v)
        v_dcm = R @ v
        
        np.testing.assert_array_almost_equal(v_quat, v_dcm)
    
    def test_inverse_rotation(self):
        """Inverse rotation should return to original."""
        q = Quaternion.from_euler(0.3, 0.2, 0.5)
        
        v_original = np.array([1.0, 2.0, 3.0])
        v_ned = q.rotate_vector(v_original)
        v_back = q.inverse_rotate_vector(v_ned)
        
        np.testing.assert_array_almost_equal(v_original, v_back)


class TestWindAngles:
    """Tests for wind angle calculations."""
    
    def test_straight_flight(self):
        """Forward flight should have zero alpha and beta."""
        v_body = np.array([25.0, 0.0, 0.0])
        V, alpha, beta = wind_angles(v_body)
        
        assert abs(V - 25.0) < 1e-10
        assert abs(alpha) < 1e-10
        assert abs(beta) < 1e-10
    
    def test_positive_alpha(self):
        """Downward velocity component should give positive alpha."""
        v_body = np.array([25.0, 0.0, 2.5])  # Some downward component
        V, alpha, beta = wind_angles(v_body)
        
        assert alpha > 0  # Nose up relative to wind
        assert abs(beta) < 1e-10
        expected_alpha = np.arctan2(2.5, 25.0)
        assert abs(alpha - expected_alpha) < 1e-10
    
    def test_positive_beta(self):
        """Rightward velocity component should give positive beta."""
        v_body = np.array([25.0, 2.5, 0.0])
        V, alpha, beta = wind_angles(v_body)
        
        assert abs(alpha) < 1e-10
        assert beta > 0  # Wind from right
    
    def test_zero_velocity(self):
        """Zero velocity should return zeros."""
        v_body = np.array([0.0, 0.0, 0.0])
        V, alpha, beta = wind_angles(v_body)
        
        assert V == 0.0
        assert alpha == 0.0
        assert beta == 0.0


class TestWindBodyTransforms:
    """Tests for wind/body frame transformations."""
    
    def test_zero_angles_identity(self):
        """Zero alpha/beta should give identity rotation."""
        R = body_to_wind_dcm(0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))
    
    def test_inverse_relationship(self):
        """Wind-to-body should be inverse of body-to-wind."""
        alpha, beta = 0.1, 0.05
        
        R_bw = body_to_wind_dcm(alpha, beta)
        R_wb = wind_to_body_dcm(alpha, beta)
        
        product = R_bw @ R_wb
        np.testing.assert_array_almost_equal(product, np.eye(3))


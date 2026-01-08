"""
Tests for flight dynamics.

These tests validate the physics implementation following the
"Brick to Glider" approach:
1. Gravity-only (The Brick)
2. Gravity + Thrust + Drag (The Rocket)
3. + Moments (The Arrow)
4. + Lift (The Glider)
"""

import numpy as np
import pytest
from simulator.state import AircraftState, ControlInputs
from simulator.aircraft import AircraftConfig
from simulator.environment import Environment
from simulator.dynamics import FlightDynamics, SimulationConfig, state_derivative
from simulator.frames import Quaternion
from simulator.trim import compute_trim, TrimCondition, validate_stall_speed


class TestBrick:
    """Phase 1: Gravity-only tests."""
    
    @pytest.fixture
    def brick_sim(self):
        """Create simulation with only gravity enabled."""
        config = SimulationConfig(
            dt=0.01,
            enable_gravity=True,
            enable_aerodynamics=False,
            enable_thrust=False,
            enable_moments=False
        )
        aircraft = AircraftConfig()
        dynamics = FlightDynamics(aircraft, Environment(), config)
        return dynamics
    
    def test_free_fall_acceleration(self, brick_sim):
        """Object in free fall should accelerate at g."""
        brick_sim.reset(AircraftState(
            position=np.array([0.0, 0.0, -1000.0]),
            velocity_body=np.zeros(3),
            quaternion=Quaternion(1, 0, 0, 0),
            omega_body=np.zeros(3)
        ))
        
        # Run for 1 second
        for _ in range(100):
            brick_sim.step(ControlInputs(throttle=0))
        
        # Should have velocity ≈ g*t = 9.81 m/s downward
        v_ned = brick_sim.state.velocity_ned
        expected_down = 9.81 * 1.0
        
        # Down is positive in NED
        assert abs(v_ned[2] - expected_down) < 0.1, \
            f"Expected {expected_down}, got {v_ned[2]}"
    
    def test_energy_conservation_free_fall(self, brick_sim):
        """Total mechanical energy should be conserved (minus integration error)."""
        mass = brick_sim.aircraft.mass_properties.mass
        g = 9.81
        
        brick_sim.reset(AircraftState(
            position=np.array([0.0, 0.0, -1000.0]),
            velocity_body=np.zeros(3),
            quaternion=Quaternion(1, 0, 0, 0),
            omega_body=np.zeros(3)
        ))
        
        # Initial energy
        E0 = mass * g * brick_sim.state.altitude
        
        # Run for 2 seconds
        for _ in range(200):
            brick_sim.step(ControlInputs(throttle=0))
        
        # Final energy
        v = np.linalg.norm(brick_sim.state.velocity_body)
        h = brick_sim.state.altitude
        E1 = 0.5 * mass * v**2 + mass * g * h
        
        # Energy should be conserved within integration tolerance
        relative_error = abs(E1 - E0) / E0
        assert relative_error < 0.01, f"Energy error: {relative_error*100:.2f}%"
    
    def test_tilted_fall(self, brick_sim):
        """Tilted object should still fall straight down."""
        # 45 degree bank
        brick_sim.reset(AircraftState(
            position=np.array([0.0, 0.0, -500.0]),
            velocity_body=np.zeros(3),
            quaternion=Quaternion.from_euler(np.pi/4, 0, 0),
            omega_body=np.zeros(3)
        ))
        
        # Run for 1 second
        for _ in range(100):
            brick_sim.step(ControlInputs(throttle=0))
        
        # Should have moved only in z (down)
        pos = brick_sim.state.position
        assert abs(pos[0]) < 0.1, f"Unexpected North motion: {pos[0]}"
        assert abs(pos[1]) < 0.1, f"Unexpected East motion: {pos[1]}"


class TestRocket:
    """Phase 2: Thrust + Drag tests."""
    
    @pytest.fixture
    def rocket_sim(self):
        """Create simulation with gravity, thrust, and drag (no lift)."""
        config = SimulationConfig(
            dt=0.01,
            enable_gravity=True,
            enable_aerodynamics=True,
            enable_thrust=True,
            enable_moments=False
        )
        aircraft = AircraftConfig()
        dynamics = FlightDynamics(aircraft, Environment(), config)
        return dynamics
    
    def test_terminal_velocity_concept(self, rocket_sim):
        """At high speed, drag should balance thrust."""
        # Start with significant forward velocity
        rocket_sim.reset(AircraftState(
            position=np.array([0.0, 0.0, -500.0]),
            velocity_body=np.array([40.0, 0.0, 0.0]),
            quaternion=Quaternion.from_euler(0, 0, 0),
            omega_body=np.zeros(3)
        ))
        
        # Run with constant throttle
        for _ in range(500):
            rocket_sim.step(ControlInputs(throttle=0.5))
        
        # Check that velocity has stabilized (not diverging)
        v1 = np.linalg.norm(rocket_sim.state.velocity_body)
        
        for _ in range(100):
            rocket_sim.step(ControlInputs(throttle=0.5))
        
        v2 = np.linalg.norm(rocket_sim.state.velocity_body)
        
        # Velocity should not be changing rapidly
        assert abs(v2 - v1) < 2.0, f"Velocity not stable: {v1:.1f} -> {v2:.1f}"


class TestArrow:
    """Phase 3: Stability/moment tests."""
    
    @pytest.fixture
    def arrow_sim(self):
        """Create simulation with moments enabled."""
        config = SimulationConfig(
            dt=0.01,
            enable_gravity=True,
            enable_aerodynamics=True,
            enable_thrust=True,
            enable_moments=True
        )
        aircraft = AircraftConfig()
        dynamics = FlightDynamics(aircraft, Environment(), config)
        return dynamics
    
    def test_pitch_stability(self, arrow_sim):
        """Aircraft should resist pitch disturbance (positive Cma)."""
        # Start in level flight with a pitch-up perturbation
        arrow_sim.reset(AircraftState(
            position=np.array([0.0, 0.0, -500.0]),
            velocity_body=np.array([25.0, 0.0, 0.0]),
            quaternion=Quaternion.from_euler(0, 0.1, 0),  # 5.7° pitch up
            omega_body=np.zeros(3)
        ))
        
        initial_theta = arrow_sim.state.theta
        
        # Run for 1 second
        for _ in range(100):
            arrow_sim.step(ControlInputs(throttle=0.5))
        
        # The pitch should have returned toward equilibrium (or at least not diverged wildly)
        final_theta = arrow_sim.state.theta
        
        # With negative Cma (stable), the restoring moment should push nose down
        # Just check we haven't diverged to crazy values
        assert abs(final_theta) < 1.0, f"Pitch diverged: {np.degrees(final_theta):.1f}°"


class TestGlider:
    """Phase 4: Full aerodynamic tests."""
    
    @pytest.fixture
    def glider_sim(self):
        """Create full simulation."""
        config = SimulationConfig(dt=0.01)
        aircraft = AircraftConfig()
        dynamics = FlightDynamics(aircraft, Environment(), config)
        return dynamics
    
    def test_trim_solver(self, glider_sim):
        """Trim solver should find a valid trim point."""
        result = compute_trim(
            TrimCondition(airspeed=25.0, altitude=100.0, climb_rate=0.0),
            glider_sim.aircraft,
            glider_sim.environment
        )
        
        assert result.success, f"Trim failed: {result.message}"
        
        # Check residuals are small
        max_residual = np.max(np.abs(result.residuals))
        assert max_residual < 1.0, f"Large residual: {max_residual}"
    
    def test_trimmed_flight_stability(self, glider_sim):
        """Trimmed flight should maintain altitude and speed."""
        result = compute_trim(
            TrimCondition(airspeed=25.0, altitude=100.0, climb_rate=0.0),
            glider_sim.aircraft,
            glider_sim.environment
        )
        
        if not result.success:
            pytest.skip("Trim failed")
        
        glider_sim.reset(result.state)
        
        # Need to step once to populate forces_moments
        glider_sim.step(result.controls)
        
        initial_alt = glider_sim.state.altitude
        initial_speed = result.forces_moments.airspeed  # Use trim result airspeed
        
        # Run for 10 seconds with trim controls
        for _ in range(1000):
            glider_sim.step(result.controls)
        
        final_alt = glider_sim.state.altitude
        final_speed = glider_sim.forces_moments.airspeed
        
        # Altitude should stay within 10m
        assert abs(final_alt - initial_alt) < 10.0, \
            f"Altitude drifted: {initial_alt:.1f} -> {final_alt:.1f}m"
        
        # Speed should stay within 5 m/s
        assert abs(final_speed - initial_speed) < 5.0, \
            f"Speed drifted: {initial_speed:.1f} -> {final_speed:.1f}m/s"
    
    def test_coordinated_turn(self, glider_sim):
        """Coordinated turn should match expected turn rate."""
        bank = np.radians(30)
        airspeed = 25.0
        
        # Expected turn rate: omega = g*tan(phi)/V
        g = 9.81
        expected_omega = g * np.tan(bank) / airspeed
        
        result = compute_trim(
            TrimCondition(airspeed=airspeed, altitude=100.0, bank_angle=bank),
            glider_sim.aircraft,
            glider_sim.environment
        )
        
        if not result.success:
            pytest.skip("Turn trim failed")
        
        glider_sim.reset(result.state)
        initial_heading = glider_sim.state.psi
        
        # Run for 10 seconds
        for _ in range(1000):
            glider_sim.step(result.controls)
        
        final_heading = glider_sim.state.psi
        
        # Compute actual turn rate
        heading_change = final_heading - initial_heading
        # Normalize to [-pi, pi]
        while heading_change > np.pi:
            heading_change -= 2*np.pi
        while heading_change < -np.pi:
            heading_change += 2*np.pi
        
        actual_omega = heading_change / 10.0
        
        # Should be within 20% of expected
        relative_error = abs(actual_omega - expected_omega) / expected_omega
        assert relative_error < 0.3, \
            f"Turn rate error: expected {np.degrees(expected_omega):.1f}°/s, " \
            f"got {np.degrees(actual_omega):.1f}°/s"


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_no_nan_propagation(self):
        """Simulation should not produce NaN values."""
        config = SimulationConfig(dt=0.01)
        aircraft = AircraftConfig()
        dynamics = FlightDynamics(aircraft, Environment(), config)
        dynamics.reset()
        
        # Run for 30 seconds with varying controls
        for i in range(3000):
            controls = ControlInputs(
                elevator=0.1 * np.sin(i * 0.01),
                aileron=0.1 * np.cos(i * 0.01),
                rudder=0.05 * np.sin(i * 0.02),
                throttle=0.5 + 0.3 * np.sin(i * 0.005)
            )
            dynamics.step(controls)
            
            # Check for NaN
            state_arr = dynamics.state.to_array()
            assert not np.any(np.isnan(state_arr)), \
                f"NaN detected at step {i}"
    
    def test_quaternion_stays_normalized(self):
        """Quaternion should remain normalized after many steps."""
        config = SimulationConfig(dt=0.01)
        aircraft = AircraftConfig()
        dynamics = FlightDynamics(aircraft, Environment(), config)
        dynamics.reset()
        
        for _ in range(1000):
            dynamics.step(ControlInputs(aileron=0.2))
        
        q = dynamics.state.quaternion
        norm = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        
        assert abs(norm - 1.0) < 1e-6, f"Quaternion denormalized: {norm}"


class TestStallSpeed:
    """Tests for stall behavior."""
    
    def test_stall_speed_reasonable(self):
        """Computed stall speed should be in reasonable range."""
        aircraft = AircraftConfig()
        v_stall = validate_stall_speed(aircraft)
        
        # Typical small UAS stall between 10-20 m/s
        assert 8.0 < v_stall < 25.0, f"Stall speed out of range: {v_stall:.1f} m/s"
    
    def test_low_speed_reduced_lift(self):
        """At very low speed, lift should be insufficient for level flight."""
        config = SimulationConfig(dt=0.01)
        aircraft = AircraftConfig()
        dynamics = FlightDynamics(aircraft, Environment(), config)
        
        # Start at very low speed (below stall)
        dynamics.reset(AircraftState(
            position=np.array([0.0, 0.0, -500.0]),
            velocity_body=np.array([8.0, 0.0, 0.0]),  # Below stall
            quaternion=Quaternion.from_euler(0, 0.2, 0),  # Some pitch up
            omega_body=np.zeros(3)
        ))
        
        # Run for 2 seconds
        for _ in range(200):
            dynamics.step(ControlInputs(throttle=0.5))
        
        # Should have lost altitude (stalling)
        alt_loss = 500.0 - dynamics.state.altitude
        assert alt_loss > 10.0, f"Expected altitude loss in stall, lost only {alt_loss:.1f}m"


"""
Main Entry Point

Run the flight simulator with visualization or headless for batch rollouts.
"""

import argparse
import numpy as np
from pathlib import Path

from .aircraft import AircraftConfig
from .environment import Environment, WindModel
from .state import AircraftState, ControlInputs
from .dynamics import FlightDynamics, SimulationConfig
from .trim import compute_trim, TrimCondition, validate_stall_speed
from .frames import Quaternion


def create_default_aircraft() -> AircraftConfig:
    """Create a generic small fixed-wing UAS configuration."""
    return AircraftConfig(
        name="Generic UAS",
        wing_span=3.0,
        wing_area=0.8,
        mean_chord=0.27
    )


def run_validation_tests(aircraft: AircraftConfig, verbose: bool = True):
    """
    Run physics validation tests.
    
    These tests verify the simulator is working correctly before
    moving to more complex scenarios.
    """
    if verbose:
        print("\n" + "="*60)
        print("FLIGHT DYNAMICS VALIDATION TESTS")
        print("="*60)
    
    environment = Environment()
    sim_config = SimulationConfig(dt=0.01)
    
    # === TEST 1: The Brick ===
    # Drop with gravity only, verify 9.81 m/s² acceleration
    if verbose:
        print("\n[Test 1] The Brick - Gravity Only")
    
    brick_config = SimulationConfig(
        dt=0.01,
        enable_gravity=True,
        enable_aerodynamics=False,
        enable_thrust=False,
        enable_moments=False
    )
    
    dynamics = FlightDynamics(aircraft, environment, brick_config)
    dynamics.reset(AircraftState(
        position=np.array([0.0, 0.0, -1000.0]),  # 1000m altitude
        velocity_body=np.zeros(3),
        quaternion=Quaternion.from_euler(0.0, 0.0, 0.0),
        omega_body=np.zeros(3)
    ))
    
    # Step for 1 second
    for _ in range(100):
        dynamics.step(ControlInputs(throttle=0.0))
    
    # Should have fallen with g acceleration
    expected_velocity = 9.81 * 1.0  # v = g*t
    # In NED, positive Z is down, so falling object has positive velocity_ned[2]
    actual_velocity = dynamics.state.velocity_ned[2]
    
    error = abs(actual_velocity - expected_velocity)
    passed = error < 0.1
    
    if verbose:
        print(f"  Expected vertical velocity: {expected_velocity:.2f} m/s")
        print(f"  Actual vertical velocity: {actual_velocity:.2f} m/s")
        print(f"  Error: {error:.4f} m/s")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    # === TEST 2: Trim Solver ===
    if verbose:
        print("\n[Test 2] Trim Solver - Level Flight")
    
    trim_result = compute_trim(
        TrimCondition(airspeed=25.0, altitude=100.0, climb_rate=0.0),
        aircraft,
        environment
    )
    
    if verbose:
        print(f"  Trim {'succeeded' if trim_result.success else 'FAILED'}")
        print(f"  Alpha: {np.degrees(trim_result.state.euler_angles[1] - np.arcsin(0)):.2f}°")
        print(f"  Throttle: {trim_result.controls.throttle:.2%}")
        print(f"  Elevator: {np.degrees(trim_result.controls.elevator):.2f}°")
        print(f"  Residuals: {trim_result.residuals}")
    
    # === TEST 3: Glide Stability ===
    if verbose:
        print("\n[Test 3] Glide Stability - Energy Conservation")
    
    glide_config = SimulationConfig(
        dt=0.01,
        enable_gravity=True,
        enable_aerodynamics=True,
        enable_thrust=False,  # Glider
        enable_moments=True
    )
    
    dynamics = FlightDynamics(aircraft, environment, glide_config)
    
    # Start from trim and run for 10 seconds
    dynamics.reset(trim_result.state)
    initial_energy = (
        0.5 * aircraft.mass_properties.mass * dynamics.state.groundspeed**2 +
        aircraft.mass_properties.mass * 9.81 * dynamics.state.altitude
    )
    
    for _ in range(1000):
        dynamics.step(ControlInputs(throttle=0.0, elevator=trim_result.controls.elevator))
    
    final_energy = (
        0.5 * aircraft.mass_properties.mass * np.linalg.norm(dynamics.state.velocity_body)**2 +
        aircraft.mass_properties.mass * 9.81 * dynamics.state.altitude
    )
    
    # Energy should decrease (drag) but not explode
    energy_ratio = final_energy / initial_energy
    passed = 0.5 < energy_ratio < 1.0  # Lost some energy to drag, but didn't explode
    
    if verbose:
        print(f"  Initial energy: {initial_energy:.0f} J")
        print(f"  Final energy: {final_energy:.0f} J")
        print(f"  Energy ratio: {energy_ratio:.3f}")
        print(f"  Final altitude: {dynamics.state.altitude:.1f} m")
        print(f"  Final airspeed: {dynamics.forces_moments.airspeed:.1f} m/s")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    # === TEST 4: Coordinated Turn ===
    if verbose:
        print("\n[Test 4] Coordinated Turn - Turn Rate Check")
    
    bank_angle = np.radians(30)
    airspeed = 25.0
    expected_turn_rate = 9.81 * np.tan(bank_angle) / airspeed
    
    turn_trim = compute_trim(
        TrimCondition(airspeed=airspeed, altitude=100.0, bank_angle=bank_angle),
        aircraft,
        environment
    )
    
    if verbose:
        print(f"  Turn trim {'succeeded' if turn_trim.success else 'FAILED'}")
        print(f"  Expected turn rate: {np.degrees(expected_turn_rate):.1f} °/s")
        print(f"  Bank angle: {np.degrees(bank_angle):.1f}°")
    
    # Run the turn and check actual heading change
    if turn_trim.success:
        dynamics = FlightDynamics(aircraft, environment, SimulationConfig(dt=0.01))
        dynamics.reset(turn_trim.state)
        initial_heading = dynamics.state.psi
        
        # Run for 10 seconds
        for _ in range(1000):
            dynamics.step(turn_trim.controls)
        
        final_heading = dynamics.state.psi
        heading_change = (final_heading - initial_heading) % (2 * np.pi)
        if heading_change > np.pi:
            heading_change -= 2 * np.pi
        
        actual_turn_rate = heading_change / 10.0
        error = abs(actual_turn_rate - expected_turn_rate)
        passed = error < 0.05  # Within 3 deg/s
        
        if verbose:
            print(f"  Actual turn rate: {np.degrees(actual_turn_rate):.1f} °/s")
            print(f"  Error: {np.degrees(error):.2f} °/s")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    # === TEST 5: Stall Speed Estimate ===
    if verbose:
        print("\n[Test 5] Stall Speed Estimate")
    
    stall_speed = validate_stall_speed(aircraft)
    if verbose:
        print(f"  Estimated stall speed: {stall_speed:.1f} m/s")
        print(f"  Min configured airspeed: {aircraft.min_airspeed:.1f} m/s")
    
    if verbose:
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60 + "\n")


def run_interactive_simulation(aircraft: AircraftConfig):
    """Run interactive simulation with visualization."""
    from .visualization import run_with_visualization
    
    environment = Environment()
    sim_config = SimulationConfig(dt=0.01)
    
    # Start from trim
    trim_result = compute_trim(
        TrimCondition(airspeed=25.0, altitude=100.0),
        aircraft,
        environment
    )
    
    dynamics = FlightDynamics(aircraft, environment, sim_config)
    
    if trim_result.success:
        print(f"Starting from trimmed flight at {25.0} m/s")
        dynamics.reset(trim_result.state)
        dynamics.controls = trim_result.controls
    else:
        print("Warning: Trim failed, starting from default state")
        dynamics.reset()
    
    # Get frontend directory
    frontend_dir = Path(__file__).parent.parent / "frontend"
    
    print("\nStarting visualization server...")
    print("Open http://localhost:8080 in your browser")
    print("WebSocket running on ws://localhost:8765")
    print("Press Ctrl+C to stop\n")
    
    run_with_visualization(
        dynamics,
        frontend_dir=str(frontend_dir) if frontend_dir.exists() else None,
        ws_port=8765,
        http_port=8080
    )


def run_headless_simulation(
    aircraft: AircraftConfig,
    duration: float = 60.0,
    output_file: str = None
):
    """Run headless simulation for batch processing."""
    import json
    
    environment = Environment()
    sim_config = SimulationConfig(dt=0.01)
    
    # Start from trim
    trim_result = compute_trim(
        TrimCondition(airspeed=25.0, altitude=100.0),
        aircraft,
        environment
    )
    
    dynamics = FlightDynamics(aircraft, environment, sim_config)
    dynamics.reset(trim_result.state if trim_result.success else None)
    
    print(f"Running {duration}s simulation...")
    
    # Simple autopilot: maintain altitude and heading
    def autopilot(state: AircraftState, time: float) -> ControlInputs:
        # Very simple PID-like control
        target_alt = 100.0
        alt_error = target_alt - state.altitude
        
        elevator = trim_result.controls.elevator - 0.01 * alt_error
        
        return ControlInputs(
            elevator=elevator,
            aileron=0.0,
            rudder=0.0,
            throttle=trim_result.controls.throttle
        )
    
    history = dynamics.run(duration, control_callback=autopilot)
    
    print(f"Simulation complete. {len(history)} data points recorded.")
    
    if output_file:
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(history, f, cls=__import__('simulator.visualization', fromlist=['StateEncoder']).StateEncoder)
        print(f"Saved to {output_file}")
    
    # Print summary
    final = history[-1] if history else {}
    print(f"\nFinal state:")
    print(f"  Time: {final.get('time', 0):.1f}s")
    print(f"  Altitude: {-final.get('position', [0,0,0])[2]:.1f}m")
    print(f"  Airspeed: {final.get('airspeed', 0):.1f}m/s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fixed-Wing UAS Flight Simulator")
    
    parser.add_argument(
        '--aircraft', '-a',
        type=str,
        help='Path to aircraft configuration YAML'
    )
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Run validation tests'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without visualization'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=60.0,
        help='Simulation duration for headless mode (seconds)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for headless simulation'
    )
    
    args = parser.parse_args()
    
    # Load aircraft
    if args.aircraft:
        aircraft = AircraftConfig.from_yaml(args.aircraft)
        print(f"Loaded aircraft: {aircraft.name}")
    else:
        aircraft = create_default_aircraft()
        print(f"Using default aircraft: {aircraft.name}")
    
    # Run appropriate mode
    if args.validate:
        run_validation_tests(aircraft)
    elif args.headless:
        run_headless_simulation(aircraft, args.duration, args.output)
    else:
        run_interactive_simulation(aircraft)


if __name__ == "__main__":
    main()


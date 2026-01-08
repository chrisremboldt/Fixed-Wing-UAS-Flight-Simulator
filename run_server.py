#!/usr/bin/env python3
"""
Start the flight simulator with visualization server.

This launches:
1. WebSocket server for streaming aircraft state (port 8765)
2. HTTP server for the Three.js frontend (port 8080)

Open http://localhost:8080 in your browser to view the simulation.
Use keyboard controls:
    W/S - Pitch down/up
    A/D - Roll left/right
    Q/E - Yaw left/right
    ↑/↓ - Throttle up/down
    R   - Reset simulation
"""

import sys
import os

# Add simulator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.aircraft import AircraftConfig
from simulator.environment import Environment
from simulator.dynamics import FlightDynamics, SimulationConfig
from simulator.trim import compute_trim, TrimCondition
from simulator.visualization import run_with_visualization
from pathlib import Path


def main():
    print("="*60)
    print("Fixed-Wing UAS Flight Simulator")
    print("="*60)
    print()
    
    # Create aircraft
    aircraft = AircraftConfig(name="Generic UAS")
    print(f"Aircraft: {aircraft.name}")
    print(f"  Mass: {aircraft.mass_properties.mass} kg")
    print(f"  Wing span: {aircraft.wing_span} m")
    print(f"  Wing area: {aircraft.wing_area} m²")
    print()
    
    # Create environment
    environment = Environment()
    
    # Create simulation
    sim_config = SimulationConfig(dt=0.01)
    dynamics = FlightDynamics(aircraft, environment, sim_config)
    
    # Trim for level flight
    print("Computing trim for level flight at 25 m/s, 100m altitude...")
    trim_result = compute_trim(
        TrimCondition(airspeed=25.0, altitude=100.0),
        aircraft,
        environment
    )
    
    if trim_result.success:
        print(f"  ✓ Trim successful")
        print(f"    Alpha: {trim_result.forces_moments.alpha*57.3:.1f}°")
        print(f"    Throttle: {trim_result.controls.throttle:.1%}")
        print(f"    Elevator: {trim_result.controls.elevator*57.3:.1f}°")
        dynamics.reset(trim_result.state)
        dynamics.controls = trim_result.controls
    else:
        print(f"  ✗ Trim failed, using default state")
        dynamics.reset()
    
    print()
    
    # Get frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    
    print("Starting servers...")
    print(f"  HTTP Server: http://localhost:8080")
    print(f"  WebSocket:   ws://localhost:8765")
    print()
    print("Open http://localhost:8080 in your browser")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        run_with_visualization(
            dynamics,
            frontend_dir=str(frontend_dir) if frontend_dir.exists() else None,
            ws_port=8765,
            http_port=8080
        )
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except ImportError as e:
        print(f"\nError: {e}")
        print("Install required packages: pip install websockets")


if __name__ == "__main__":
    main()


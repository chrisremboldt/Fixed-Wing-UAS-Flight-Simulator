#!/usr/bin/env python3
"""
Quick validation script for the flight simulator.

Run this to verify the physics implementation is working correctly.
"""

import sys
import os

# Add simulator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.aircraft import AircraftConfig
from simulator.main import run_validation_tests


if __name__ == "__main__":
    print("Fixed-Wing UAS Flight Simulator")
    print("Physics Validation Suite")
    print()
    
    aircraft = AircraftConfig(name="Validation Test Aircraft")
    run_validation_tests(aircraft, verbose=True)


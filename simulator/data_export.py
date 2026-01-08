"""
Data Export Module

Export simulator data in standardized formats for analysis and validation:
- NASA-style CSV with standard column headers
- MATLAB-compatible formats
- JSON for web visualization
- Custom formats for specific tools
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


def export_nasa_csv(
    history: List[Dict],
    filename: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Export flight history to NASA-style CSV format.
    
    Standard columns following NASA flight test conventions:
    - Time (s)
    - Position: N, E, Alt (m)
    - Euler angles: phi, theta, psi (deg)
    - Angular rates: p, q, r (deg/s)
    - Velocities: u, v, w (m/s)
    - Airdata: V_air, alpha, beta, q_bar (m/s, deg, deg, Pa)
    - Forces: Fx, Fy, Fz (N)
    - Moments: L, M, N (N·m)
    - Coefficients: CL, CD, CY, Cl, Cm, Cn
    - Controls: elevator, aileron, rudder, throttle (deg, deg, deg, %)
    - Load factors: nx, ny, nz (g)
    
    Args:
        history: List of state dictionaries from simulation
        filename: Output filename
        metadata: Optional metadata dictionary for header
    """
    if not history:
        raise ValueError("History is empty")
    
    # Convert to DataFrame for easier manipulation
    rows = []
    
    for record in history:
        row = {
            # Time
            'time_s': record['time'],
            
            # Position (NED frame)
            'north_m': record['position'][0],
            'east_m': record['position'][1],
            'down_m': record['position'][2],
            'altitude_m': -record['position'][2],
            
            # Euler angles (degrees)
            'phi_deg': np.degrees(record['euler'][0]),
            'theta_deg': np.degrees(record['euler'][1]),
            'psi_deg': np.degrees(record['euler'][2]),
            
            # Angular rates (deg/s)
            'p_deg_s': np.degrees(record['omega'][0]),
            'q_deg_s': np.degrees(record['omega'][1]),
            'r_deg_s': np.degrees(record['omega'][2]),
            
            # Velocities (body frame)
            'u_m_s': record['velocity'][0],
            'v_m_s': record['velocity'][1],
            'w_m_s': record['velocity'][2],
            
            # Airdata
            'airspeed_m_s': record['airspeed'],
            'alpha_deg': np.degrees(record['alpha']),
            'beta_deg': np.degrees(record['beta']),
            
            # Forces (body frame)
            'Fx_N': record['forces'][0],
            'Fy_N': record['forces'][1],
            'Fz_N': record['forces'][2],
            
            # Moments (body frame)
            'L_Nm': record['moments'][0],
            'M_Nm': record['moments'][1],
            'N_Nm': record['moments'][2],
            
            # Controls
            'elevator_deg': np.degrees(record['controls'][0]),
            'aileron_deg': np.degrees(record['controls'][1]),
            'rudder_deg': np.degrees(record['controls'][2]),
            'throttle_pct': record['controls'][3] * 100,
        }
        
        # Add dynamic pressure if available
        if 'dynamic_pressure' in record:
            row['q_bar_Pa'] = record['dynamic_pressure']
        
        # Add coefficients if available
        if 'CL' in record:
            row['CL'] = record['CL']
        if 'CD' in record:
            row['CD'] = record['CD']
        if 'Cm' in record:
            row['Cm'] = record['Cm']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Write with header comments
    with open(filename, 'w') as f:
        # Write metadata header
        f.write("# NASA-Style Flight Test Data\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        
        f.write("#\n")
        f.write("# Units:\n")
        f.write("#   Length: meters (m)\n")
        f.write("#   Time: seconds (s)\n")
        f.write("#   Angles: degrees (deg)\n")
        f.write("#   Angular rates: degrees per second (deg/s)\n")
        f.write("#   Forces: Newtons (N)\n")
        f.write("#   Moments: Newton-meters (N·m)\n")
        f.write("#   Pressure: Pascals (Pa)\n")
        f.write("#\n")
        f.write("# Coordinate Systems:\n")
        f.write("#   Position: NED (North-East-Down)\n")
        f.write("#   Velocities: Body frame (X forward, Y right, Z down)\n")
        f.write("#   Euler angles: 3-2-1 sequence (roll-pitch-yaw)\n")
        f.write("#\n")
        
        # Write CSV data
        df.to_csv(f, index=False)
    
    print(f"Exported {len(df)} records to {filename}")


def export_polar_csv(
    alpha_deg: np.ndarray,
    CL: np.ndarray,
    CD: np.ndarray,
    Cm: Optional[np.ndarray],
    filename: str,
    airfoil_name: str = "Unknown",
    reynolds_number: float = 1e6,
    metadata: Optional[Dict] = None
) -> None:
    """
    Export coefficient polar data in standard format.
    
    Args:
        alpha_deg: Angle of attack (degrees)
        CL: Lift coefficient
        CD: Drag coefficient
        Cm: Pitching moment coefficient (optional)
        filename: Output filename
        airfoil_name: Airfoil name for header
        reynolds_number: Reynolds number
        metadata: Additional metadata
    """
    df = pd.DataFrame({
        'alpha_deg': alpha_deg,
        'CL': CL,
        'CD': CD
    })
    
    if Cm is not None:
        df['Cm'] = Cm
    
    with open(filename, 'w') as f:
        f.write("# Airfoil Coefficient Polar Data\n")
        f.write(f"# Airfoil: {airfoil_name}\n")
        f.write(f"# Reynolds Number: {reynolds_number:.2e}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        
        f.write("#\n")
        f.write("# Columns:\n")
        f.write("#   alpha_deg: Angle of attack (degrees)\n")
        f.write("#   CL: Lift coefficient (dimensionless)\n")
        f.write("#   CD: Drag coefficient (dimensionless)\n")
        if Cm is not None:
            f.write("#   Cm: Pitching moment coefficient (dimensionless)\n")
        f.write("#\n")
        
        df.to_csv(f, index=False)
    
    print(f"Exported polar data ({len(df)} points) to {filename}")


def export_maneuver_csv(
    history: List[Dict],
    filename: str,
    maneuver_type: str = "Unknown",
    reference_params: Optional[Dict] = None
) -> None:
    """
    Export maneuver time history for dynamic validation.
    
    Focuses on key parameters for specific maneuvers
    (doublets, pull-ups, turns, etc.)
    
    Args:
        history: Simulation history
        filename: Output filename
        maneuver_type: Type of maneuver for header
        reference_params: Reference conditions (altitude, airspeed, etc.)
    """
    rows = []
    
    for record in history:
        row = {
            'time_s': record['time'],
            'altitude_m': -record['position'][2],
            'airspeed_m_s': record['airspeed'],
            'alpha_deg': np.degrees(record['alpha']),
            'beta_deg': np.degrees(record['beta']),
            'phi_deg': np.degrees(record['euler'][0]),
            'theta_deg': np.degrees(record['euler'][1]),
            'psi_deg': np.degrees(record['euler'][2]),
            'p_deg_s': np.degrees(record['omega'][0]),
            'q_deg_s': np.degrees(record['omega'][1]),
            'r_deg_s': np.degrees(record['omega'][2]),
            'elevator_deg': np.degrees(record['controls'][0]),
            'aileron_deg': np.degrees(record['controls'][1]),
            'rudder_deg': np.degrees(record['controls'][2]),
            'throttle': record['controls'][3]
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    with open(filename, 'w') as f:
        f.write(f"# Maneuver Time History: {maneuver_type}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        
        if reference_params:
            f.write("# Reference Conditions:\n")
            for key, value in reference_params.items():
                f.write(f"#   {key}: {value}\n")
        
        f.write("#\n")
        
        df.to_csv(f, index=False)
    
    print(f"Exported maneuver data ({len(df)} points) to {filename}")


def export_trim_data(
    trim_results: List[Dict],
    filename: str,
    vary_parameter: str = "airspeed"
) -> None:
    """
    Export trim curve data (e.g., elevator vs airspeed).
    
    Args:
        trim_results: List of trim result dictionaries
        filename: Output filename
        vary_parameter: What parameter was varied
    """
    df = pd.DataFrame(trim_results)
    
    with open(filename, 'w') as f:
        f.write(f"# Trim Curve Data (varied: {vary_parameter})\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write("#\n")
        
        df.to_csv(f, index=False)
    
    print(f"Exported trim data ({len(df)} points) to {filename}")


def export_json(
    history: List[Dict],
    filename: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Export to JSON format for web visualization or further processing.
    
    Args:
        history: Simulation history
        filename: Output filename
        metadata: Optional metadata
    """
    output = {
        'metadata': metadata or {},
        'generated': datetime.now().isoformat(),
        'n_points': len(history),
        'data': history
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    print(f"Exported {len(history)} records to {filename}")


def create_validation_package(
    sim_history: List[Dict],
    airfoil_data: Optional[pd.DataFrame],
    trim_data: Optional[pd.DataFrame],
    output_dir: str,
    aircraft_name: str = "aircraft"
) -> None:
    """
    Create a complete validation data package with multiple formats.
    
    Args:
        sim_history: Flight history from simulation
        airfoil_data: Polar coefficient data
        trim_data: Trim curve data
        output_dir: Output directory
        aircraft_name: Aircraft name for filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export flight history
    if sim_history:
        nasa_file = output_path / f"{aircraft_name}_flight_{timestamp}.csv"
        export_nasa_csv(sim_history, str(nasa_file))
        
        json_file = output_path / f"{aircraft_name}_flight_{timestamp}.json"
        export_json(sim_history, str(json_file))
    
    # Export polar data
    if airfoil_data is not None:
        polar_file = output_path / f"{aircraft_name}_polar_{timestamp}.csv"
        export_polar_csv(
            airfoil_data['alpha_deg'].values,
            airfoil_data['CL'].values,
            airfoil_data['CD'].values,
            airfoil_data['Cm'].values if 'Cm' in airfoil_data.columns else None,
            str(polar_file),
            airfoil_name=aircraft_name
        )
    
    # Export trim data
    if trim_data is not None:
        trim_file = output_path / f"{aircraft_name}_trim_{timestamp}.csv"
        trim_data.to_csv(trim_file, index=False)
    
    print(f"\nValidation package created in: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Data Export Module - Example Usage\n")
    
    # Create dummy history data
    dummy_history = []
    for i in range(10):
        t = i * 0.1
        dummy_history.append({
            'time': t,
            'position': np.array([t*20, 0, -100]),
            'velocity': np.array([20, 0, 0]),
            'euler': np.array([0, 0.1, 0]),
            'omega': np.array([0, 0, 0]),
            'airspeed': 20.0,
            'alpha': 0.1,
            'beta': 0.0,
            'forces': np.array([0, 0, -200]),
            'moments': np.array([0, 0, 0]),
            'controls': (0.0, 0.0, 0.0, 0.5)
        })
    
    # Export to NASA CSV
    export_nasa_csv(
        dummy_history,
        'test_flight.csv',
        metadata={'aircraft': 'Test UAS', 'test': 'Example'}
    )
    
    # Export polar
    alpha = np.linspace(-10, 20, 31)
    CL = 0.1 * alpha / 57.3 * 5.0
    CD = 0.01 + 0.05 * CL**2
    
    export_polar_csv(
        alpha, CL, CD, None,
        'test_polar.csv',
        airfoil_name='Test Airfoil',
        reynolds_number=1e6
    )
    
    print("\nExample files created successfully")


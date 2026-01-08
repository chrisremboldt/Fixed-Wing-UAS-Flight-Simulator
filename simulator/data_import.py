"""
Data Import Module

Handles importing and parsing real-world airfoil data from various sources:
- UIUC Low-Speed Airfoil Tests
- NASA Airfoil-Learning Dataset
- Kanakaero/SplineCloud CSVs
- Generic CSV formats

Normalizes data to a common format for validation/tuning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class AirfoilData:
    """
    Standardized container for airfoil performance data.
    
    All angles in degrees, coefficients dimensionless.
    """
    name: str
    reynolds_number: float
    
    # Tabulated data
    alpha: np.ndarray  # deg
    CL: np.ndarray
    CD: np.ndarray
    Cm: Optional[np.ndarray] = None
    
    # Metadata
    source: str = "unknown"
    mach: Optional[float] = None
    notes: str = ""
    
    def __post_init__(self):
        """Validate and convert to numpy arrays."""
        self.alpha = np.asarray(self.alpha, dtype=np.float64)
        self.CL = np.asarray(self.CL, dtype=np.float64)
        self.CD = np.asarray(self.CD, dtype=np.float64)
        if self.Cm is not None:
            self.Cm = np.asarray(self.Cm, dtype=np.float64)
        
        # Validate shapes
        if not (len(self.alpha) == len(self.CL) == len(self.CD)):
            raise ValueError("Alpha, CL, and CD must have same length")
        
        if self.Cm is not None and len(self.Cm) != len(self.alpha):
            raise ValueError("Cm must have same length as alpha")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for easy manipulation."""
        data = {
            'alpha_deg': self.alpha,
            'CL': self.CL,
            'CD': self.CD
        }
        if self.Cm is not None:
            data['Cm'] = self.Cm
        
        df = pd.DataFrame(data)
        df.attrs['name'] = self.name
        df.attrs['Re'] = self.reynolds_number
        df.attrs['source'] = self.source
        return df
    
    def interpolate_at_alpha(self, alpha_deg: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Interpolate coefficients at specified angles of attack.
        
        Args:
            alpha_deg: Angles of attack in degrees
            
        Returns:
            Dict with 'CL', 'CD', and optionally 'Cm' interpolated values
        """
        from scipy.interpolate import interp1d
        
        # Sort by alpha for interpolation
        sort_idx = np.argsort(self.alpha)
        alpha_sorted = self.alpha[sort_idx]
        CL_sorted = self.CL[sort_idx]
        CD_sorted = self.CD[sort_idx]
        
        result = {
            'CL': interp1d(alpha_sorted, CL_sorted, 
                          bounds_error=False, fill_value='extrapolate')(alpha_deg),
            'CD': interp1d(alpha_sorted, CD_sorted,
                          bounds_error=False, fill_value='extrapolate')(alpha_deg)
        }
        
        if self.Cm is not None:
            Cm_sorted = self.Cm[sort_idx]
            result['Cm'] = interp1d(alpha_sorted, Cm_sorted,
                                   bounds_error=False, fill_value='extrapolate')(alpha_deg)
        
        return result


def load_uiuc_dat(filepath: str, airfoil_name: Optional[str] = None) -> AirfoilData:
    """
    Load UIUC airfoil database .dat file.
    
    Expected format:
    - Comment lines start with '#'
    - Whitespace-delimited columns: alpha CL CD Cm (or subset)
    - Alpha in degrees
    
    Reynolds number extracted from filename if present (e.g., 'naca0012-il-100000.dat')
    
    Args:
        filepath: Path to .dat file
        airfoil_name: Override airfoil name (extracted from filename if None)
        
    Returns:
        AirfoilData object
    """
    path = Path(filepath)
    
    # Extract Reynolds number from filename if possible
    # Common format: airfoilname-suffix-Re.dat
    re_number = None
    filename_parts = path.stem.split('-')
    for part in filename_parts:
        if part.isdigit():
            re_number = float(part)
            break
    
    if re_number is None:
        warnings.warn(f"Could not extract Reynolds number from filename: {path.name}")
        re_number = 1e6  # Default assumption
    
    # Determine airfoil name
    if airfoil_name is None:
        # Use first part of filename
        airfoil_name = filename_parts[0] if filename_parts else path.stem
    
    # Read data
    try:
        # Try reading with pandas (handles comments automatically)
        df = pd.read_csv(filepath, sep=r'\s+', comment='#', 
                        names=['alpha', 'CL', 'CD', 'Cm'])
    except ValueError:
        # Might have fewer columns (no Cm)
        try:
            df = pd.read_csv(filepath, sep=r'\s+', comment='#',
                           names=['alpha', 'CL', 'CD'])
        except Exception as e:
            raise ValueError(f"Failed to parse UIUC file {filepath}: {e}")
    
    return AirfoilData(
        name=airfoil_name,
        reynolds_number=re_number,
        alpha=df['alpha'].values,
        CL=df['CL'].values,
        CD=df['CD'].values,
        Cm=df['Cm'].values if 'Cm' in df.columns else None,
        source='UIUC',
        notes=f"Loaded from {path.name}"
    )


def load_generic_csv(
    filepath: str,
    airfoil_name: str,
    reynolds_number: float,
    alpha_col: Optional[str] = None,
    cl_col: str = 'CL',
    cd_col: str = 'CD',
    cm_col: Optional[str] = 'Cm',
    skip_rows: int = 0
) -> AirfoilData:
    """
    Load generic CSV format with flexible column naming.
    
    Args:
        filepath: Path to CSV file
        airfoil_name: Name of airfoil
        reynolds_number: Reynolds number for this data
        alpha_col: Name of alpha column (auto-detects 'alpha' or 'alpha_deg' if None)
        cl_col: Name of CL column
        cd_col: Name of CD column
        cm_col: Name of Cm column (None if not present)
        skip_rows: Number of header rows to skip
        
    Returns:
        AirfoilData object
    """
    df = pd.read_csv(filepath, skiprows=skip_rows, comment='#')
    
    # Auto-detect alpha column if not specified
    if alpha_col is None:
        if 'alpha_deg' in df.columns:
            alpha_col = 'alpha_deg'
        elif 'alpha' in df.columns:
            alpha_col = 'alpha'
        elif 'AoA' in df.columns:
            alpha_col = 'AoA'
        else:
            raise ValueError(f"Could not find alpha column. Available: {df.columns.tolist()}")
    
    # Check required columns exist
    required = [alpha_col, cl_col, cd_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {df.columns.tolist()}")
    
    Cm = df[cm_col].values if cm_col and cm_col in df.columns else None
    
    return AirfoilData(
        name=airfoil_name,
        reynolds_number=reynolds_number,
        alpha=df[alpha_col].values,
        CL=df[cl_col].values,
        CD=df[cd_col].values,
        Cm=Cm,
        source='CSV',
        notes=f"Loaded from {Path(filepath).name}"
    )


def load_airfoiltools_csv(filepath: str) -> AirfoilData:
    """
    Load Airfoil Tools format CSV.
    
    Expected format (from airfoiltools.com):
        Header lines with metadata
        Column headers: Alpha,Cl,Cd,Cdp,Cm,Top_Xtr,Bot_Xtr
        Data rows
    
    Args:
        filepath: Path to Airfoil Tools CSV
        
    Returns:
        AirfoilData object
    """
    path = Path(filepath)
    
    # Read header to extract metadata
    airfoil_name = "Unknown"
    re_number = 1e6
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i > 20:  # Don't search too far
                break
            if 'Airfoil,' in line:
                airfoil_name = line.split(',')[1].strip()
            elif 'Reynolds number,' in line:
                try:
                    re_number = float(line.split(',')[1].strip())
                except:
                    pass
            elif line.startswith('Alpha,'):
                # This is the data header, data starts next line
                skip_rows = i
                break
    
    # Load data
    df = pd.read_csv(filepath, skiprows=skip_rows)
    
    # Column mapping (Airfoil Tools uses 'Alpha' and 'Cl' instead of 'alpha' and 'CL')
    df = df.rename(columns={
        'Alpha': 'alpha',
        'Cl': 'CL',
        'Cd': 'CD',
        'Cm': 'Cm'
    })
    
    return AirfoilData(
        name=airfoil_name,
        reynolds_number=re_number,
        alpha=df['alpha'].values,
        CL=df['CL'].values,
        CD=df['CD'].values,
        Cm=df['Cm'].values if 'Cm' in df.columns else None,
        source='AirfoilTools',
        notes=f"Loaded from {path.name}"
    )


def load_kanakaero_csv(filepath: str) -> AirfoilData:
    """
    Load Kanakaero-format CSV (typically has metadata in header).
    
    Expected format:
    # Airfoil: NACA 0012
    # Re: 100000
    alpha,CL,CD,Cm
    -10,0.1,0.02,0.01
    ...
    
    Args:
        filepath: Path to Kanakaero CSV
        
    Returns:
        AirfoilData object
    """
    path = Path(filepath)
    
    # Read header to extract metadata
    airfoil_name = "Unknown"
    re_number = 1e6
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if 'Airfoil:' in line or 'airfoil:' in line:
                    airfoil_name = line.split(':')[1].strip()
                elif 'Re:' in line or 're:' in line or 'Reynolds:' in line:
                    try:
                        re_str = line.split(':')[1].strip()
                        re_number = float(re_str)
                    except:
                        pass
            else:
                break
    
    # Load data
    df = pd.read_csv(filepath, comment='#')
    
    # Flexible column naming
    alpha_col = next((col for col in df.columns if 'alpha' in col.lower() or 'aoa' in col.lower()), 'alpha')
    cl_col = next((col for col in df.columns if col.upper() == 'CL'), 'CL')
    cd_col = next((col for col in df.columns if col.upper() == 'CD'), 'CD')
    cm_col = next((col for col in df.columns if col.upper() == 'CM'), None)
    
    return AirfoilData(
        name=airfoil_name,
        reynolds_number=re_number,
        alpha=df[alpha_col].values,
        CL=df[cl_col].values,
        CD=df[cd_col].values,
        Cm=df[cm_col].values if cm_col else None,
        source='Kanakaero',
        notes=f"Loaded from {path.name}"
    )


def compute_reynolds_number(
    airspeed: float,
    chord: float,
    altitude: float = 0.0,
    temperature_offset: float = 0.0
) -> float:
    """
    Compute Reynolds number for given flight conditions.
    
    Uses ISA atmosphere model for density and viscosity.
    
    Args:
        airspeed: True airspeed (m/s)
        chord: Reference chord length (m)
        altitude: Altitude above sea level (m)
        temperature_offset: Deviation from ISA temperature (K)
        
    Returns:
        Reynolds number (dimensionless)
    """
    from .environment import isa_atmosphere
    
    atm = isa_atmosphere(altitude)
    T = atm.temperature + temperature_offset
    rho = atm.density
    
    # Sutherland's formula for dynamic viscosity of air
    T0 = 273.15  # Reference temperature (K)
    mu0 = 1.716e-5  # Reference viscosity (Pa·s)
    S = 110.4  # Sutherland constant (K)
    
    mu = mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)
    
    # Re = ρ V L / μ
    Re = rho * airspeed * chord / mu
    
    return Re


def match_reynolds_conditions(
    target_re: float,
    chord: float,
    altitude: float = 0.0
) -> float:
    """
    Compute airspeed needed to match target Reynolds number.
    
    Args:
        target_re: Target Reynolds number
        chord: Reference chord (m)
        altitude: Altitude (m)
        
    Returns:
        Required airspeed (m/s)
    """
    from .environment import isa_atmosphere
    
    atm = isa_atmosphere(altitude)
    T = atm.temperature
    rho = atm.density
    
    # Viscosity
    T0 = 273.15
    mu0 = 1.716e-5
    S = 110.4
    mu = mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)
    
    # Solve for V: Re = ρ V L / μ  =>  V = Re μ / (ρ L)
    V = target_re * mu / (rho * chord)
    
    return V


def create_test_data(name: str = "test_airfoil") -> AirfoilData:
    """
    Create synthetic test data for development/testing.
    
    Generates a typical symmetric airfoil polar.
    
    Args:
        name: Name for test airfoil
        
    Returns:
        AirfoilData with synthetic data
    """
    alpha = np.linspace(-10, 20, 31)
    
    # Typical NACA 0012-like characteristics
    CLa = 0.11  # per degree
    CL0 = 0.0   # Symmetric
    alpha_stall = 15.0
    CL_max = 1.4
    
    CL = np.zeros_like(alpha)
    for i, a in enumerate(alpha):
        if abs(a) < alpha_stall:
            CL[i] = CL0 + CLa * a
        else:
            # Simple stall model
            excess = abs(a) - alpha_stall
            CL[i] = CL_max * np.exp(-0.3 * excess) * np.sign(a)
    
    # Parabolic polar: CD = CD0 + k*CL^2
    CD0 = 0.008
    k = 0.05
    CD = CD0 + k * CL**2
    
    # Pitching moment (slightly nose-down)
    Cm = -0.02 - 0.001 * alpha
    
    return AirfoilData(
        name=name,
        reynolds_number=1e6,
        alpha=alpha,
        CL=CL,
        CD=CD,
        Cm=Cm,
        source='synthetic',
        notes='Generated test data'
    )


if __name__ == "__main__":
    # Example usage
    print("Data Import Module - Example Usage\n")
    
    # Create test data
    test_data = create_test_data("NACA 0012 Synthetic")
    print(f"Created test data: {test_data.name}")
    print(f"  Re: {test_data.reynolds_number:.0e}")
    print(f"  Alpha range: {test_data.alpha.min():.1f}° to {test_data.alpha.max():.1f}°")
    print(f"  CL range: {test_data.CL.min():.3f} to {test_data.CL.max():.3f}")
    print(f"  CD range: {test_data.CD.min():.4f} to {test_data.CD.max():.4f}")
    
    # Reynolds number calculations
    print("\nReynolds Number Calculations:")
    chord = 0.27  # m
    V = 25.0  # m/s
    Re = compute_reynolds_number(V, chord, altitude=100.0)
    print(f"  At V={V} m/s, chord={chord} m, altitude=100m: Re = {Re:.0e}")
    
    V_required = match_reynolds_conditions(1e5, chord, altitude=100.0)
    print(f"  To match Re=1e5 at altitude=100m: V = {V_required:.1f} m/s")


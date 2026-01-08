# Fixed-Wing UAS Flight Simulator

A physics-based, 6-DOF flight dynamics simulator for fixed-wing Unmanned Aerial Systems with **real-world validation and parameter tuning capabilities**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- **🎯 6-DOF Rigid Body Dynamics** - Full equations of motion with RK4 integration
- **✅ NASA/NACA-Style Validation** - Compare against real airfoil data (UIUC, Airfoil Tools)
- **🔧 Automated Parameter Tuning** - Optimize stability derivatives using scipy
- **📊 Aerospace-Standard Visualization** - CL/CD polars, drag polars, L/D curves
- **📁 NASA-Format Data Export** - Flight test CSV compatible with standard tools
- **🌐 Web-Based 3D Visualization** - Real-time Three.js visualization
- **🧪 Physics Validation Suite** - Built-in tests (The Brick, Rocket, Arrow, Glider)
- **⚡ Fast & Modular** - Clean architecture for DAA algorithm testing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chrisremboldt/Fixed-Wing-UAS-Flight-Simulator.git
cd Fixed-Wing-UAS-Flight-Simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Physics Validation

```bash
# Test with synthetic data
python run_validation_workflow.py --test

# Output: Complete validation reports and plots in test_validation_results/
```

### Validate with Real Airfoil Data

```bash
# Download real airfoil data
python scripts/download_airfoil_data.py --common

# Run validation and tuning
python run_validation_workflow.py \
    --data data/airfoiltools/naca2412_re200k.csv \
    --output results/naca2412

# View results
open results/naca2412/validation_final_polar_comparison.png
```

### Run Interactive Simulation

```bash
# Start simulator with web visualization
python -m simulator.main --aircraft configs/generic_uav.yaml

# Open http://localhost:8080 in your browser
```

## 📊 Validation & Tuning Workflow

The simulator includes a complete validation pipeline using real-world airfoil data:

```
Real Airfoil Data (UIUC/NASA/Airfoil Tools)
    ↓
Load & Parse → Generate Sim Predictions
    ↓
Compare (RMSE, R², Correlation)
    ↓
Automated Parameter Tuning (scipy.optimize)
    ↓
Re-Validate & Generate Reports
    ↓
NASA-Format Exports + Aerospace Plots
```

**Example Results:**
- ✅ Drag coefficient: 61% improvement after tuning
- ✅ Lift coefficient: R² = 0.95 (excellent correlation)
- ✅ Comprehensive reports meeting NASA validation standards

### Quick Example

```python
from simulator import (
    load_airfoiltools_csv,
    AircraftConfig,
    validate_multiple_coefficients,
    tune_to_polar_data,
    create_default_tuning_config
)

# Load real airfoil data
real_data = load_airfoiltools_csv('data/naca2412_re200k.csv')

# Load aircraft
aircraft = AircraftConfig.from_yaml('configs/generic_uav.yaml')

# Validate and tune
config = create_default_tuning_config(['CL0', 'CLa', 'CD0', 'CD_k'])
result = tune_to_polar_data(real_data, aircraft.aero, config)

print(f"Improvement: {result.improvement*100:.1f}%")
```

## 🏗️ Architecture

### Physics Implementation

**6-DOF Equations of Motion:**
- **Translational**: `dv/dt = F/m - ω × v` (body frame)
- **Rotational**: `dω/dt = I⁻¹(M - ω × (Iω))` (Euler's equations)
- **Kinematics**: Quaternion propagation (avoids gimbal lock)
- **Integration**: 4th-order Runge-Kutta for numerical stability

**Aerodynamics (Stability Derivatives):**
```python
CL = CL₀ + CLₐ·α + CLq·q̂ + CLδₑ·δₑ
CD = CD₀ + k·CL² + CDₐ·α² + CDᵦ·β²
Cm = Cm₀ + Cmₐ·α + Cmq·q̂ + Cmδₑ·δₑ
```

**Environment:**
- ISA 1976 Standard Atmosphere
- Wind and turbulence models
- Gravity variation with altitude

### State Vector (13 states)

| State | Symbol | Frame | Units |
|-------|--------|-------|-------|
| Position | p_n, p_e, p_d | NED | m |
| Velocity | u, v, w | Body | m/s |
| Quaternion | q₀, q₁, q₂, q₃ | Body→NED | - |
| Angular rates | p, q, r | Body | rad/s |

### Coordinate Frames

- **NED (North-East-Down)**: Inertial reference frame
- **Body Frame**: X forward (nose), Y right (starboard), Z down (belly)
- **Wind/Stability Frame**: Aligned with relative airflow

## 📚 Documentation

- **[VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)** - Complete guide to validation and tuning
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[examples/](examples/)** - Example scripts and workflows

## 🔧 Core Modules

| Module | Description |
|--------|-------------|
| `simulator/dynamics.py` | 6-DOF equations of motion, RK4 integration |
| `simulator/aerodynamics.py` | Stability derivatives, coefficient models |
| `simulator/aircraft.py` | Aircraft configuration, mass properties |
| `simulator/trim.py` | Trim solver (level flight, turns, climbs) |
| `simulator/autopilot.py` | Basic autopilot controllers |
| `simulator/data_import.py` | Load UIUC/NASA/Airfoil Tools data |
| `simulator/validation.py` | Compare sim vs real, compute metrics |
| `simulator/tuning.py` | Automated parameter optimization |
| `simulator/plotting.py` | Aerospace-standard plots |
| `simulator/data_export.py` | NASA-format CSV exports |

## 🎓 Validation Tests

Built-in physics validation tests:

```bash
python run_validation.py
```

**Test Suite:**
1. ✅ **The Brick** - Gravity only (verify 9.81 m/s²)
2. ✅ **Trim Solver** - Level flight force balance
3. ✅ **Glide Stability** - Energy conservation
4. ✅ **Coordinated Turn** - `ω = g·tan(φ)/V` relationship
5. ✅ **Stall Speed** - `V_stall = √(2W/(ρSCL_max))`

## 📊 Data Sources

### Supported Formats

- **UIUC Airfoil Database** (.dat) - Wind tunnel data
- **Airfoil Tools** (.csv) - XFOIL simulations
- **Kanakaero** (.csv) - Pre-processed data
- **Generic CSV** - Custom formats (auto-detects columns)

### Download Helper

```bash
# List available airfoils
python scripts/download_airfoil_data.py --list

# Download specific airfoil
python scripts/download_airfoil_data.py naca0012_re500k

# Download common set for UAS
python scripts/download_airfoil_data.py --common
```

## 🎯 Example Validation Results

**NACA 2412 at Re=200k (113 data points):**

| Coefficient | Initial RMSE | Tuned RMSE | Improvement | R² |
|-------------|--------------|------------|-------------|-----|
| CL (Lift) | 0.186 | 0.167 | 10.4% | 0.95 |
| CD (Drag) | 0.031 | 0.012 | **61.2%** | 0.77 |

**Generated Outputs:**
- ✅ Validation reports (NASA-standard metrics)
- ✅ Polar comparison plots (CL, CD, Cm vs α)
- ✅ Drag polar (CL vs CD with max L/D)
- ✅ L/D curves
- ✅ CSV exports for further analysis

## 🛠️ Command-Line Tools

### Validation Workflow

```bash
# Complete workflow (load, validate, tune, plot, export)
python run_validation_workflow.py \
    --data data/airfoil.csv \
    --output results/airfoil \
    --aircraft configs/generic_uav.yaml

# Validate without tuning
python run_validation_workflow.py --data data/airfoil.csv --no-tune

# Tune specific parameters
python run_validation_workflow.py \
    --data data/airfoil.csv \
    --tune-params CL0 CLa CD0 CD_k alpha_stall CL_max
```

### Airfoil Data Download

```bash
# Download from Airfoil Tools database
python scripts/download_airfoil_data.py naca2412_re200k
python scripts/download_airfoil_data.py clark-y_re500k
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run physics validation suite
python run_validation.py
```

## 📦 Requirements

- Python 3.10+
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Pandas ≥ 2.0
- Matplotlib ≥ 3.7
- PyYAML ≥ 6.0
- WebSockets ≥ 11.0

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Additional aerodynamic models (compressibility, ground effect)
- More autopilot modes
- System identification tools
- Additional data source integrations

## 📖 References

### Textbooks & Papers
- **Stevens & Lewis** - *Aircraft Control and Simulation* (2003)
- **Beard & McLain** - *Small Unmanned Aircraft* (2012)
- **NACA Reports** - Airfoil data and stability derivatives

### Data Sources
- [UIUC Airfoil Database](https://m-selig.ae.illinois.edu/ads.html)
- [Airfoil Tools](http://airfoiltools.com)
- NASA Technical Reports

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- UIUC Applied Aerodynamics Group for airfoil data
- NASA for validation methodologies
- Airfoil Tools for XFOIL-based datasets

---

## 🚀 Getting Started Paths

### Path 1: Test the Physics
```bash
python run_validation.py  # Built-in physics tests
```

### Path 2: Validate with Real Data
```bash
python run_validation_workflow.py --test  # Synthetic data
python scripts/download_airfoil_data.py --common  # Real data
python run_validation_workflow.py --data data/airfoiltools/naca2412_re200k.csv
```

### Path 3: Run Simulations
```bash
python -m simulator.main --validate  # Quick validation
python -m simulator.main  # Interactive with visualization
```

### Path 4: Use the API
```python
from simulator import FlightDynamics, AircraftConfig, Environment

aircraft = AircraftConfig.from_yaml('configs/generic_uav.yaml')
dynamics = FlightDynamics(aircraft, Environment())
dynamics.reset()  # Start from trim

# Simulation loop
for _ in range(1000):
    dynamics.step(controls)
    print(f"Altitude: {dynamics.state.altitude:.1f}m")
```

---

**Built for DAA algorithm testing. Validated with real-world data. Ready for research.** 🛩️

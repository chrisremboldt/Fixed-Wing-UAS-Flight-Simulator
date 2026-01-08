# Validation and Tuning Guide

This guide explains how to use real-world airfoil data to validate and tune the simulator's physics model.

## Overview

The simulator uses **stability derivatives** and **coefficient-based aerodynamics**, which aligns directly with NACA/NASA airfoil data methodologies. This makes it straightforward to:

1. **Validate**: Compare simulator outputs against real experimental data
2. **Tune**: Adjust parameters to minimize errors and improve realism

## Quick Start

### Test with Synthetic Data

```bash
# Run complete workflow with built-in test data
python run_validation_workflow.py --test
```

This will:
- Generate synthetic "real" data
- Create a simulator with intentionally wrong parameters
- Tune the parameters to match the data
- Generate validation reports and plots

### Use Real Data

```bash
# With UIUC airfoil data
python run_validation_workflow.py \
    --data path/to/naca0012-il-100000.dat \
    --output results/naca0012/ \
    --aircraft configs/generic_uav.yaml

# With custom CSV data
python run_validation_workflow.py \
    --data my_airfoil_data.csv \
    --output results/custom/
```

## Data Sources

### UIUC Low-Speed Airfoil Tests

Best for UAS validation (low Reynolds numbers).

**Download**: https://m-selig.ae.illinois.edu/ads/coord_database.html

**Format**: Text files with columns `alpha CL CD Cm`
```
# NACA 0012
-10.0  -0.850  0.0250  0.0120
 -5.0  -0.425  0.0110  0.0060
  0.0   0.000  0.0080  0.0000
  5.0   0.525  0.0095 -0.0080
 ...
```

### NASA Airfoil-Learning Dataset

Computational data across broad Re ranges.

**Source**: NASA repositories / Kaggle

**Format**: Typically CSV or NumPy arrays

### Kanakaero / SplineCloud

Pre-processed CSV datasets.

**Format**: CSV with metadata header
```csv
# Airfoil: NACA 2412
# Re: 500000
alpha,CL,CD,Cm
-10,0.1,0.02,0.01
...
```

## Workflow Steps

### 1. Data Import

```python
from simulator.data_import import load_uiuc_dat, compute_reynolds_number

# Load real data
real_data = load_uiuc_dat('naca0012-il-100000.dat')

# Check conditions match your UAS
Re_sim = compute_reynolds_number(
    airspeed=25.0,    # m/s
    chord=0.27,       # m
    altitude=100.0    # m
)
print(f"Simulator Re: {Re_sim:.2e}")
print(f"Real data Re: {real_data.reynolds_number:.2e}")
```

### 2. Generate Simulator Data

```python
from simulator.aircraft import AircraftConfig

# Load your aircraft
aircraft = AircraftConfig.from_yaml('configs/generic_uav.yaml')

# Generate polar at matching conditions
from run_validation_workflow import generate_sim_polar

sim_data = generate_sim_polar(
    aircraft.aero,
    alpha_deg_range=real_data.alpha
)
```

### 3. Validate

```python
from simulator.validation import validate_multiple_coefficients, generate_validation_report

# Compare coefficients
results = validate_multiple_coefficients(
    real_data,
    sim_data,
    coefficients=['CL', 'CD', 'Cm']
)

# Generate report
report = generate_validation_report(results)
print(report)
```

**Acceptance Criteria** (NASA standard for low-fidelity sims):
- RMSE < 0.05 for coefficients
- R² > 0.95

### 4. Tune Parameters

```python
from simulator.tuning import tune_to_polar_data, create_default_tuning_config

# Configure tuning
config = create_default_tuning_config(['CL0', 'CLa', 'CD0', 'CD_k'])

# Optimize
tuning_result = tune_to_polar_data(
    real_data,
    aircraft.aero,  # Modified in-place
    config,
    verbose=True
)

print(tuning_result)
```

### 5. Visualize Results

```python
from simulator.plotting import create_validation_report_plots

create_validation_report_plots(
    real_data,
    sim_data,
    results,
    output_dir='plots/',
    prefix='my_validation'
)
```

Generates:
- `my_validation_polar_comparison.png` - CL, CD, Cm vs α
- `my_validation_validation_metrics.png` - With RMSE/R² annotations
- `my_validation_drag_polar.png` - CL vs CD
- `my_validation_LD_curve.png` - L/D vs α

### 6. Export Data

```python
from simulator.data_export import export_polar_csv, export_nasa_csv

# Export sim polar
export_polar_csv(
    sim_data['alpha_deg'].values,
    sim_data['CL'].values,
    sim_data['CD'].values,
    sim_data['Cm'].values,
    'output/sim_polar.csv',
    airfoil_name='My Airfoil (Sim)',
    reynolds_number=1e5
)
```

## Parameter Tuning Details

### Which Parameters to Tune?

Run sensitivity analysis first:

```python
from simulator.tuning import tune_with_sensitivity_analysis

sensitivity = tune_with_sensitivity_analysis(
    real_data,
    aircraft.aero,
    test_parameters=['CL0', 'CLa', 'CD0', 'CD_k', 'Cma', 'Cmq']
)

print(sensitivity)
```

This identifies the most influential parameters for your data.

### Common Tuning Sets

**Basic CL/CD matching:**
```python
parameters = ['CL0', 'CLa', 'CD0', 'CD_k']
```

**With stall characteristics:**
```python
parameters = ['CL0', 'CLa', 'alpha_stall', 'CL_max', 'CD0', 'CD_k']
```

**Full longitudinal:**
```python
parameters = ['CL0', 'CLa', 'CLq', 'CLde',
              'CD0', 'CD_k', 'CDa',
              'Cm0', 'Cma', 'Cmq', 'Cmde']
```

### Tuning Algorithms

**Least Squares** (default, fast for local optimization):
```python
config = TuningConfig(
    parameters=['CL0', 'CLa', 'CD0', 'CD_k'],
    method='trf',  # Trust Region Reflective
    ftol=1e-8
)
```

**Differential Evolution** (global optimization, slower but more robust):
```python
config = TuningConfig(
    parameters=['CL0', 'CLa', 'CD0', 'CD_k'],
    method='de',  # Differential Evolution
    max_iterations=1000
)
```

### Weighting

Weight coefficients by importance:

```python
config = TuningConfig(
    parameters=['CL0', 'CLa', 'CD0', 'CD_k'],
    weights={'CL': 2.0, 'CD': 1.0, 'Cm': 0.5}  # Emphasize CL
)
```

### Regularization

Penalize large deviations from initial guess (prevents overfitting):

```python
config = TuningConfig(
    parameters=['CL0', 'CLa', 'CD0', 'CD_k'],
    regularization=0.1  # Mild penalty
)
```

## Advanced Topics

### Matching Reynolds Number

If real data is at different Re than your sim conditions:

```python
from simulator.data_import import match_reynolds_conditions

# Find airspeed to match target Re
V_required = match_reynolds_conditions(
    target_re=1e5,
    chord=0.27,
    altitude=100.0
)

print(f"Need V = {V_required:.1f} m/s to match Re=1e5")
```

### 2D Airfoil → 3D Wing Corrections

Real data is 2D; your sim is 3D. Account for:

1. **Induced drag**: Already in sim via `CD_k` (aspect ratio effects)
2. **Finite wing effects**: May need to adjust `CLa` slightly downward
3. **Tip losses**: Effective aspect ratio < geometric

### Dynamic Validation

Beyond static polars, validate maneuver responses:

```python
from simulator.dynamics import FlightDynamics
from simulator.data_export import export_maneuver_csv

# Run elevator doublet
dynamics = FlightDynamics(aircraft, environment)
# ... apply doublet input ...

export_maneuver_csv(
    dynamics.history,
    'doublet_response.csv',
    maneuver_type='Elevator Doublet',
    reference_params={'V': 25.0, 'altitude': 100.0}
)
```

Compare to flight test data (if available).

### System Identification

Estimate derivatives from time-series data (reverse problem):

```python
# TODO: Implement in future version
# from simulator.system_id import estimate_derivatives
# 
# derivatives = estimate_derivatives(flight_test_data)
```

## Output Files

After running workflow, you'll have:

```
validation_results/
├── validation_initial.txt          # Pre-tuning metrics
├── validation_tuned.txt            # Post-tuning metrics
├── validation_initial_*.png        # Initial comparison plots
├── validation_final_*.png          # Final comparison plots
├── sim_polar_final.csv             # Tuned sim polar
├── real_polar.csv                  # Real data (for reference)
└── metrics_comparison.csv          # Derived metrics (L/D, CL_max, etc.)
```

## Troubleshooting

### "Insufficient overlap in alpha range"

Real and sim data must cover overlapping α range. Extend sim range:

```python
sim_data = generate_sim_polar(
    aircraft.aero,
    alpha_deg_range=np.arange(-15, 25, 1)  # Wider range
)
```

### Poor convergence in tuning

1. Try different optimization method (`method='de'`)
2. Widen parameter bounds
3. Tune fewer parameters (use sensitivity analysis)
4. Add regularization to prevent overfitting

### Sim diverges after tuning

Tuned parameters may be physically unrealistic. Check:
- Parameter bounds are reasonable
- Regularization is enabled
- Not overfitting to noisy data

## References

- **UIUC Airfoil Database**: https://m-selig.ae.illinois.edu/ads.html
- **NASA TM-110346**: *Aircraft Parameter Estimation*
- **NASA TP-2015-218749**: *Flight Test Maneuvers for Efficient Estimation*
- **NACA Report 824**: *Summary of Airfoil Data*

## Example: Complete Workflow

```python
#!/usr/bin/env python3
from simulator.data_import import load_uiuc_dat
from simulator.validation import validate_multiple_coefficients
from simulator.tuning import tune_to_polar_data, create_default_tuning_config
from simulator.plotting import create_validation_report_plots
from simulator.aircraft import AircraftConfig
from run_validation_workflow import generate_sim_polar

# 1. Load data
real_data = load_uiuc_dat('data/naca0012-il-100000.dat')

# 2. Load aircraft
aircraft = AircraftConfig.from_yaml('configs/generic_uav.yaml')

# 3. Generate initial sim data
sim_initial = generate_sim_polar(aircraft.aero, real_data.alpha)

# 4. Validate initial
results_initial = validate_multiple_coefficients(real_data, sim_initial)
print(f"Initial CL RMSE: {results_initial['CL'].metrics.rmse:.6f}")

# 5. Tune
config = create_default_tuning_config(['CL0', 'CLa', 'CD0', 'CD_k'])
tuning_result = tune_to_polar_data(real_data, aircraft.aero, config)
print(f"Improvement: {tuning_result.improvement*100:.1f}%")

# 6. Validate tuned
sim_tuned = generate_sim_polar(aircraft.aero, real_data.alpha)
results_tuned = validate_multiple_coefficients(real_data, sim_tuned)
print(f"Tuned CL RMSE: {results_tuned['CL'].metrics.rmse:.6f}")

# 7. Generate plots
create_validation_report_plots(real_data, sim_tuned, results_tuned, 'plots/')

print("Done!")
```

## API Reference

See module docstrings for detailed API documentation:
- `simulator.data_import` - Load airfoil data
- `simulator.validation` - Compare sim vs real
- `simulator.tuning` - Optimize parameters
- `simulator.plotting` - Generate plots
- `simulator.data_export` - Export results


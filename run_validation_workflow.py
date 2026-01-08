#!/usr/bin/env python3
"""
Complete Validation and Tuning Workflow

This script demonstrates the end-to-end process of:
1. Loading real-world airfoil data (UIUC, NASA, or custom)
2. Running simulator at matching conditions
3. Comparing sim vs real data
4. Tuning parameters to improve accuracy
5. Generating validation reports and plots

Usage:
    python run_validation_workflow.py --data path/to/airfoil_data.dat --output results/
    
    Or run with synthetic test data:
    python run_validation_workflow.py --test
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add simulator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.data_import import (
    load_uiuc_dat, 
    load_generic_csv,
    load_airfoiltools_csv,
    create_test_data,
    compute_reynolds_number,
    match_reynolds_conditions
)
from simulator.validation import (
    validate_multiple_coefficients,
    generate_validation_report,
    compare_derived_metrics
)
from simulator.tuning import (
    tune_to_polar_data,
    tune_with_sensitivity_analysis,
    create_default_tuning_config
)
from simulator.plotting import (
    create_validation_report_plots,
    plot_polar_comparison
)
from simulator.data_export import (
    export_polar_csv,
    export_nasa_csv,
    create_validation_package
)
from simulator.aircraft import AircraftConfig, AeroCoefficients
from simulator.trim import compute_trim, TrimCondition
from simulator.dynamics import FlightDynamics, SimulationConfig
from simulator.environment import Environment
from simulator.state import ControlInputs


def generate_sim_polar(
    aero_coeffs: AeroCoefficients,
    alpha_deg_range: np.ndarray,
    reference_chord: float = 0.27,
    altitude: float = 100.0
) -> pd.DataFrame:
    """
    Generate coefficient polar from simulator at specified conditions.
    
    Args:
        aero_coeffs: Aerodynamic coefficients to use
        alpha_deg_range: Range of angles of attack (degrees)
        reference_chord: Reference chord (m)
        altitude: Altitude (m)
        
    Returns:
        DataFrame with sim polar data
    """
    results = []
    
    for alpha_deg in alpha_deg_range:
        alpha_rad = np.deg2rad(alpha_deg)
        
        # Compute coefficients at this alpha
        CL = aero_coeffs.get_CL(alpha_rad, q_hat=0.0, delta_e=0.0)
        CD = aero_coeffs.get_CD(CL, alpha_rad, beta=0.0)
        Cm = aero_coeffs.get_Cm(alpha_rad, q_hat=0.0, delta_e=0.0)
        
        results.append({
            'alpha_deg': alpha_deg,
            'CL': CL,
            'CD': CD,
            'Cm': Cm
        })
    
    return pd.DataFrame(results)


def run_validation_workflow(
    data_file: str,
    output_dir: str,
    aircraft_config: AircraftConfig,
    tune: bool = True,
    tune_params: list = None,
    verbose: bool = True
):
    """
    Execute complete validation and tuning workflow.
    
    Args:
        data_file: Path to real-world data file
        output_dir: Output directory for results
        aircraft_config: Aircraft configuration
        tune: Whether to perform parameter tuning
        tune_params: List of parameters to tune (auto-select if None)
        verbose: Print detailed output
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("VALIDATION AND TUNING WORKFLOW")
    print("="*70)
    
    # ===== STEP 1: Load Real-World Data =====
    print("\n[1/6] Loading real-world data...")
    
    # Detect file type and load
    if data_file.endswith('.dat'):
        real_data = load_uiuc_dat(data_file)
    elif data_file.endswith('.csv'):
        # Check if it's Airfoil Tools format
        with open(data_file, 'r') as f:
            first_line = f.readline()
            if 'Xfoil polar' in first_line or 'Polar key,' in f.readline():
                real_data = load_airfoiltools_csv(data_file)
            else:
                # Generic CSV format (will auto-detect column names)
                real_data = load_generic_csv(
                    data_file,
                    airfoil_name=Path(data_file).stem,
                    reynolds_number=1e6  # Default, should be in file
                )
    else:
        raise ValueError(f"Unsupported file format: {data_file}")
    
    if verbose:
        print(f"  Loaded: {real_data.name}")
        print(f"  Source: {real_data.source}")
        print(f"  Reynolds number: {real_data.reynolds_number:.2e}")
        print(f"  Alpha range: {real_data.alpha.min():.1f}° to {real_data.alpha.max():.1f}°")
        print(f"  Data points: {len(real_data.alpha)}")
    
    # ===== STEP 2: Generate Simulator Polar (Before Tuning) =====
    print("\n[2/6] Generating simulator polar (initial parameters)...")
    
    sim_data_initial = generate_sim_polar(
        aircraft_config.aero,
        real_data.alpha,
        aircraft_config.mean_chord
    )
    
    if verbose:
        print(f"  Generated {len(sim_data_initial)} points")
    
    # ===== STEP 3: Initial Validation =====
    print("\n[3/6] Validating initial parameters...")
    
    initial_results = validate_multiple_coefficients(
        real_data,
        sim_data_initial,
        coefficients=['CL', 'CD', 'Cm'] if real_data.Cm is not None else ['CL', 'CD']
    )
    
    if verbose:
        for coef, result in initial_results.items():
            print(f"\n  {coef}:")
            print(f"    RMSE: {result.metrics.rmse:.6f}")
            print(f"    R²: {result.metrics.r_squared:.4f}")
            print(f"    Status: {'PASS' if result.passes_threshold() else 'FAIL'}")
    
    # Save initial validation report
    initial_report = generate_validation_report(
        initial_results,
        output_file=str(output_path / "validation_initial.txt")
    )
    
    # ===== STEP 4: Parameter Tuning (Optional) =====
    tuned_results = None
    sim_data_final = sim_data_initial
    
    if tune:
        print("\n[4/6] Performing parameter tuning...")
        
        # Sensitivity analysis to identify key parameters
        if tune_params is None:
            print("  Running sensitivity analysis...")
            candidate_params = ['CL0', 'CLa', 'CLq', 'CLde', 
                              'CD0', 'CD_k', 'CDa',
                              'Cm0', 'Cma', 'Cmq', 'Cmde']
            sensitivity = tune_with_sensitivity_analysis(
                real_data,
                aircraft_config.aero,
                candidate_params,
                perturbation=0.01
            )
            
            if verbose:
                print("\n  Top 5 most sensitive parameters:")
                print(sensitivity.head(5).to_string(index=False))
            
            # Select top parameters to tune
            tune_params = sensitivity.head(6)['Parameter'].tolist()
        
        print(f"\n  Tuning parameters: {tune_params}")
        
        # Create tuning configuration
        tuning_config = create_default_tuning_config(tune_params)
        
        # Perform tuning
        tuning_result = tune_to_polar_data(
            real_data,
            aircraft_config.aero,  # Will be modified in place
            tuning_config,
            verbose=verbose
        )
        
        print("\n" + str(tuning_result))
        
        # Generate sim polar with tuned parameters
        print("\n  Generating simulator polar with tuned parameters...")
        sim_data_final = generate_sim_polar(
            aircraft_config.aero,
            real_data.alpha,
            aircraft_config.mean_chord
        )
        
        # Validate tuned results
        print("\n  Validating tuned parameters...")
        tuned_results = validate_multiple_coefficients(
            real_data,
            sim_data_final,
            coefficients=['CL', 'CD', 'Cm'] if real_data.Cm is not None else ['CL', 'CD']
        )
        
        if verbose:
            for coef, result in tuned_results.items():
                initial_rmse = initial_results[coef].metrics.rmse
                tuned_rmse = result.metrics.rmse
                improvement = (initial_rmse - tuned_rmse) / initial_rmse * 100
                
                print(f"\n  {coef}:")
                print(f"    Initial RMSE: {initial_rmse:.6f}")
                print(f"    Tuned RMSE:   {tuned_rmse:.6f}")
                print(f"    Improvement:  {improvement:.1f}%")
                print(f"    Status: {'PASS' if result.passes_threshold() else 'FAIL'}")
        
        # Save tuned validation report
        tuned_report = generate_validation_report(
            tuned_results,
            output_file=str(output_path / "validation_tuned.txt")
        )
    else:
        print("\n[4/6] Skipping parameter tuning (--no-tune flag)")
    
    # ===== STEP 5: Generate Plots =====
    print("\n[5/6] Generating validation plots...")
    
    create_validation_report_plots(
        real_data,
        sim_data_final,
        tuned_results if tuned_results else initial_results,
        str(output_path),
        prefix="validation_final"
    )
    
    # Also save initial comparison if we tuned
    if tune:
        create_validation_report_plots(
            real_data,
            sim_data_initial,
            initial_results,
            str(output_path),
            prefix="validation_initial"
        )
    
    print("  Plots saved")
    
    # ===== STEP 6: Export Data =====
    print("\n[6/6] Exporting data...")
    
    # Export final sim polar
    export_polar_csv(
        sim_data_final['alpha_deg'].values,
        sim_data_final['CL'].values,
        sim_data_final['CD'].values,
        sim_data_final['Cm'].values if 'Cm' in sim_data_final.columns else None,
        str(output_path / "sim_polar_final.csv"),
        airfoil_name=real_data.name + " (Simulator)",
        reynolds_number=real_data.reynolds_number
    )
    
    # Export real data for comparison
    export_polar_csv(
        real_data.alpha,
        real_data.CL,
        real_data.CD,
        real_data.Cm,
        str(output_path / "real_polar.csv"),
        airfoil_name=real_data.name,
        reynolds_number=real_data.reynolds_number
    )
    
    # Export derived metrics comparison
    metrics_comparison = compare_derived_metrics(real_data, sim_data_final)
    metrics_comparison.to_csv(output_path / "metrics_comparison.csv", index=False)
    
    print("  Data exported")
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_path.absolute()}")
    print("\nGenerated files:")
    for file in sorted(output_path.iterdir()):
        print(f"  - {file.name}")
    
    if tune and tuned_results:
        passed = sum(1 for r in tuned_results.values() if r.passes_threshold())
        total = len(tuned_results)
        print(f"\nFinal validation: {passed}/{total} coefficients passed")
    
    print()


def run_test_workflow(output_dir: str):
    """Run workflow with synthetic test data."""
    print("Running with synthetic test data...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create synthetic "real" data
    real_data = create_test_data("NACA 0012 Synthetic")
    
    # Save to temp file
    temp_file = Path(output_dir) / "test_data.csv"
    real_df = real_data.to_dataframe()
    real_df.to_csv(temp_file, index=False)
    
    # Create aircraft with intentionally wrong parameters
    aircraft = AircraftConfig(
        name="Test UAS",
        wing_span=3.0,
        wing_area=0.8,
        mean_chord=0.27
    )
    
    # Make parameters wrong for demonstration
    aircraft.aero.CL0 = 0.15      # Should be ~0.0
    aircraft.aero.CLa = 4.5       # Should be ~6.3
    aircraft.aero.CD0 = 0.025     # Should be ~0.008
    aircraft.aero.CD_k = 0.08     # Should be ~0.05
    
    # Run workflow
    run_validation_workflow(
        str(temp_file),
        output_dir,
        aircraft,
        tune=True,
        verbose=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run complete validation and tuning workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to real-world airfoil data file (.dat or .csv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='validation_results',
        help='Output directory for results (default: validation_results)'
    )
    parser.add_argument(
        '--aircraft', '-a',
        type=str,
        help='Path to aircraft config YAML (uses default if not specified)'
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip parameter tuning (only validate)'
    )
    parser.add_argument(
        '--tune-params',
        type=str,
        nargs='+',
        help='Specific parameters to tune (e.g., CL0 CLa CD0)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run with synthetic test data'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        run_test_workflow(args.output)
        return
    
    # Normal mode - require data file
    if not args.data:
        parser.error("--data is required (or use --test for synthetic data)")
    
    if not Path(args.data).exists():
        parser.error(f"Data file not found: {args.data}")
    
    # Load aircraft config
    if args.aircraft:
        aircraft = AircraftConfig.from_yaml(args.aircraft)
        print(f"Loaded aircraft: {aircraft.name}")
    else:
        aircraft = AircraftConfig(name="Generic UAS")
        print("Using default aircraft configuration")
    
    # Run workflow
    run_validation_workflow(
        args.data,
        args.output,
        aircraft,
        tune=not args.no_tune,
        tune_params=args.tune_params,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()


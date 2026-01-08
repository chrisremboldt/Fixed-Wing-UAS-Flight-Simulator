#!/usr/bin/env python3
"""
Quick Validation Example

Demonstrates the validation and tuning workflow in minimal code.
Uses synthetic data for immediate testing without external files.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from simulator import (
    create_test_data,
    AircraftConfig,
    validate_multiple_coefficients,
    tune_to_polar_data,
    create_default_tuning_config,
    plot_polar_comparison,
    export_polar_csv
)


def main():
    print("="*70)
    print("QUICK VALIDATION EXAMPLE")
    print("="*70)
    
    # Step 1: Create synthetic "real" data
    print("\n[1/5] Creating synthetic real data...")
    real_data = create_test_data("NACA 0012 Synthetic")
    print(f"  Alpha range: {real_data.alpha.min():.1f}° to {real_data.alpha.max():.1f}°")
    print(f"  Points: {len(real_data.alpha)}")
    
    # Step 2: Create aircraft with intentionally wrong parameters
    print("\n[2/5] Setting up simulator with initial (wrong) parameters...")
    aircraft = AircraftConfig(name="Test UAS")
    
    # Make parameters intentionally wrong
    aircraft.aero.CL0 = 0.15      # Should be ~0.0 for symmetric airfoil
    aircraft.aero.CLa = 4.5       # Should be ~6.3 (0.11 per deg)
    aircraft.aero.CD0 = 0.025     # Should be ~0.008
    aircraft.aero.CD_k = 0.08     # Should be ~0.05
    
    print(f"  Initial CL0:  {aircraft.aero.CL0:.4f}")
    print(f"  Initial CLa:  {aircraft.aero.CLa:.4f}")
    print(f"  Initial CD0:  {aircraft.aero.CD0:.4f}")
    print(f"  Initial CD_k: {aircraft.aero.CD_k:.4f}")
    
    # Step 3: Generate initial simulator data
    print("\n[3/5] Generating initial simulator data...")
    import pandas as pd
    
    sim_data_initial = []
    for alpha_deg in real_data.alpha:
        alpha_rad = np.deg2rad(alpha_deg)
        CL = aircraft.aero.get_CL(alpha_rad, 0, 0)
        CD = aircraft.aero.get_CD(CL, alpha_rad, 0)
        Cm = aircraft.aero.get_Cm(alpha_rad, 0, 0)
        
        sim_data_initial.append({
            'alpha_deg': alpha_deg,
            'CL': CL,
            'CD': CD,
            'Cm': Cm
        })
    
    sim_df_initial = pd.DataFrame(sim_data_initial)
    
    # Validate initial
    print("\n  Initial validation:")
    results_initial = validate_multiple_coefficients(real_data, sim_df_initial)
    for coef, result in results_initial.items():
        print(f"    {coef}: RMSE = {result.metrics.rmse:.6f}, R² = {result.metrics.r_squared:.4f}")
    
    # Step 4: Tune parameters
    print("\n[4/5] Tuning parameters...")
    config = create_default_tuning_config(['CL0', 'CLa', 'CD0', 'CD_k'])
    
    tuning_result = tune_to_polar_data(
        real_data,
        aircraft.aero,
        config,
        verbose=False
    )
    
    print(f"\n  Optimization: {'SUCCESS' if tuning_result.success else 'FAILED'}")
    print(f"  Improvement: {tuning_result.improvement*100:.1f}%")
    print(f"\n  Optimized parameters:")
    for name, value in tuning_result.optimized_params.items():
        initial = tuning_result.initial_params[name]
        change = ((value - initial) / initial * 100) if abs(initial) > 1e-10 else 0
        print(f"    {name:8s}: {value:8.4f}  (was {initial:8.4f}, {change:+6.1f}% change)")
    
    # Step 5: Validate tuned results
    print("\n[5/5] Validating tuned parameters...")
    
    sim_data_tuned = []
    for alpha_deg in real_data.alpha:
        alpha_rad = np.deg2rad(alpha_deg)
        CL = aircraft.aero.get_CL(alpha_rad, 0, 0)
        CD = aircraft.aero.get_CD(CL, alpha_rad, 0)
        Cm = aircraft.aero.get_Cm(alpha_rad, 0, 0)
        
        sim_data_tuned.append({
            'alpha_deg': alpha_deg,
            'CL': CL,
            'CD': CD,
            'Cm': Cm
        })
    
    sim_df_tuned = pd.DataFrame(sim_data_tuned)
    
    results_tuned = validate_multiple_coefficients(real_data, sim_df_tuned)
    
    print("\n  Tuned validation:")
    for coef, result in results_tuned.items():
        initial_rmse = results_initial[coef].metrics.rmse
        tuned_rmse = result.metrics.rmse
        improvement = (initial_rmse - tuned_rmse) / initial_rmse * 100
        
        print(f"    {coef}:")
        print(f"      Initial RMSE: {initial_rmse:.6f}")
        print(f"      Tuned RMSE:   {tuned_rmse:.6f}")
        print(f"      Improvement:  {improvement:.1f}%")
        print(f"      Status:       {'PASS ✓' if result.passes_threshold() else 'FAIL ✗'}")
    
    # Generate comparison plot
    print("\n[Bonus] Generating comparison plot...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig = plot_polar_comparison(real_data, sim_df_tuned)
        output_file = Path(__file__).parent / "validation_example.png"
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to: {output_file}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        print("  (matplotlib not available - skipping plot)")
    
    # Export data
    print("\n[Bonus] Exporting data...")
    output_dir = Path(__file__).parent / "validation_output"
    output_dir.mkdir(exist_ok=True)
    
    export_polar_csv(
        sim_df_tuned['alpha_deg'].values,
        sim_df_tuned['CL'].values,
        sim_df_tuned['CD'].values,
        sim_df_tuned['Cm'].values,
        str(output_dir / "tuned_polar.csv"),
        airfoil_name="Test Airfoil (Tuned Sim)",
        reynolds_number=real_data.reynolds_number
    )
    
    print(f"  Saved CSV to: {output_dir / 'tuned_polar.csv'}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results_tuned.values() if r.passes_threshold())
    total = len(results_tuned)
    
    print(f"\nValidation: {passed}/{total} coefficients passed NASA thresholds")
    print(f"Overall: {'✓ SUCCESS' if passed == total else '✗ NEEDS WORK'}")
    
    print("\nThis example demonstrated:")
    print("  ✓ Creating/loading airfoil data")
    print("  ✓ Generating simulator predictions")
    print("  ✓ Computing validation metrics")
    print("  ✓ Tuning parameters to match data")
    print("  ✓ Visualizing results")
    print("  ✓ Exporting validated data")
    
    print("\nNext steps:")
    print("  - Use real UIUC/NASA data instead of synthetic")
    print("  - Run full workflow: ./run_validation_workflow.py --test")
    print("  - See VALIDATION_GUIDE.md for detailed instructions")
    
    print()


if __name__ == "__main__":
    main()


"""
Validation Module

Compares simulator outputs against real-world data to assess accuracy.
Computes error metrics and generates comparison reports.

Supports validation of:
- Static coefficient polars (CL, CD, Cm vs alpha)
- Dynamic maneuver responses
- Trim conditions
- Performance metrics (stall speed, L/D, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from .data_import import AirfoilData


@dataclass
class ValidationMetrics:
    """Statistical metrics for validation comparison."""
    
    # Error metrics
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
    max_error: float  # Maximum absolute error
    
    # Statistical measures
    correlation: float  # Pearson correlation coefficient
    r_squared: float   # R² coefficient of determination
    
    # Bias
    mean_bias: float  # Mean (sim - real)
    
    # Sample info
    n_points: int
    
    def __str__(self) -> str:
        return (
            f"Validation Metrics (n={self.n_points}):\n"
            f"  RMSE:        {self.rmse:.6f}\n"
            f"  MAE:         {self.mae:.6f}\n"
            f"  Max Error:   {self.max_error:.6f}\n"
            f"  Correlation: {self.correlation:.4f}\n"
            f"  R²:          {self.r_squared:.4f}\n"
            f"  Mean Bias:   {self.mean_bias:.6f}"
        )


@dataclass
class ValidationResult:
    """Complete validation result for a coefficient."""
    
    coefficient_name: str  # 'CL', 'CD', 'Cm', etc.
    metrics: ValidationMetrics
    
    # Data arrays
    independent_var: np.ndarray  # e.g., alpha
    real_values: np.ndarray
    sim_values: np.ndarray
    
    # Metadata
    airfoil_name: str = ""
    reynolds_number: Optional[float] = None
    test_condition: str = ""
    
    def passes_threshold(
        self, 
        rmse_threshold: float = 0.05,
        correlation_threshold: float = 0.95
    ) -> bool:
        """
        Check if validation passes acceptance criteria.
        
        NASA typical tolerances:
        - RMSE < 0.05 for coefficients (low-fidelity sim)
        - Correlation > 0.95
        
        Args:
            rmse_threshold: Maximum acceptable RMSE
            correlation_threshold: Minimum acceptable correlation
            
        Returns:
            True if validation passes
        """
        return (self.metrics.rmse < rmse_threshold and 
                self.metrics.correlation > correlation_threshold)


def compute_metrics(real: np.ndarray, sim: np.ndarray) -> ValidationMetrics:
    """
    Compute validation metrics between real and simulated data.
    
    Args:
        real: Real-world measured values
        sim: Simulator predicted values
        
    Returns:
        ValidationMetrics object
    """
    # Ensure arrays are same length
    if len(real) != len(sim):
        raise ValueError(f"Array length mismatch: real={len(real)}, sim={len(sim)}")
    
    # Remove any NaN/inf values
    valid_mask = np.isfinite(real) & np.isfinite(sim)
    real = real[valid_mask]
    sim = sim[valid_mask]
    
    if len(real) == 0:
        raise ValueError("No valid data points after filtering NaN/inf")
    
    # Errors
    errors = sim - real
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    mean_bias = np.mean(errors)
    
    # Correlation
    if np.std(real) > 1e-10 and np.std(sim) > 1e-10:
        correlation = np.corrcoef(real, sim)[0, 1]
        
        # R² (coefficient of determination)
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((real - np.mean(real))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    else:
        # Constant data - perfect if equal, else terrible
        correlation = 1.0 if rmse < 1e-10 else 0.0
        r_squared = correlation
    
    return ValidationMetrics(
        rmse=rmse,
        mae=mae,
        max_error=max_error,
        correlation=correlation,
        r_squared=r_squared,
        mean_bias=mean_bias,
        n_points=len(real)
    )


def validate_coefficient_polar(
    real_data: AirfoilData,
    sim_data: pd.DataFrame,
    coefficient: str = 'CL',
    interpolate: bool = True
) -> ValidationResult:
    """
    Validate a single coefficient polar (e.g., CL vs alpha).
    
    Args:
        real_data: Real-world airfoil data
        sim_data: Simulator output DataFrame with 'alpha_deg' and coefficient columns
        coefficient: Which coefficient to validate ('CL', 'CD', 'Cm')
        interpolate: If True, interpolate sim data to match real alpha points
        
    Returns:
        ValidationResult with comparison metrics
    """
    # Get real data
    real_alpha = real_data.alpha
    
    if coefficient == 'CL':
        real_values = real_data.CL
    elif coefficient == 'CD':
        real_values = real_data.CD
    elif coefficient == 'Cm':
        if real_data.Cm is None:
            raise ValueError("Real data does not contain Cm")
        real_values = real_data.Cm
    else:
        raise ValueError(f"Unknown coefficient: {coefficient}")
    
    # Get sim data
    if 'alpha_deg' not in sim_data.columns:
        raise ValueError("Sim data must have 'alpha_deg' column")
    if coefficient not in sim_data.columns:
        raise ValueError(f"Sim data must have '{coefficient}' column")
    
    sim_alpha = sim_data['alpha_deg'].values
    sim_values_raw = sim_data[coefficient].values
    
    if interpolate:
        # Interpolate sim data to match real alpha points
        from scipy.interpolate import interp1d
        
        # Check overlap
        alpha_min = max(real_alpha.min(), sim_alpha.min())
        alpha_max = min(real_alpha.max(), sim_alpha.max())
        
        # Filter real data to overlap region
        overlap_mask = (real_alpha >= alpha_min) & (real_alpha <= alpha_max)
        if np.sum(overlap_mask) < 3:
            raise ValueError(f"Insufficient overlap in alpha range. Real: [{real_alpha.min():.1f}, {real_alpha.max():.1f}], Sim: [{sim_alpha.min():.1f}, {sim_alpha.max():.1f}]")
        
        real_alpha_overlap = real_alpha[overlap_mask]
        real_values_overlap = real_values[overlap_mask]
        
        # Interpolate sim to real alpha points
        interp_func = interp1d(sim_alpha, sim_values_raw, 
                               bounds_error=False, fill_value='extrapolate')
        sim_values = interp_func(real_alpha_overlap)
        
        alpha_final = real_alpha_overlap
        real_final = real_values_overlap
        sim_final = sim_values
    else:
        # Use data as-is (must have matching alpha points)
        if not np.allclose(real_alpha, sim_alpha, atol=0.1):
            raise ValueError("Alpha points do not match. Use interpolate=True or ensure matching grids.")
        alpha_final = real_alpha
        real_final = real_values
        sim_final = sim_values_raw
    
    # Compute metrics
    metrics = compute_metrics(real_final, sim_final)
    
    return ValidationResult(
        coefficient_name=coefficient,
        metrics=metrics,
        independent_var=alpha_final,
        real_values=real_final,
        sim_values=sim_final,
        airfoil_name=real_data.name,
        reynolds_number=real_data.reynolds_number,
        test_condition=f"Re={real_data.reynolds_number:.2e}"
    )


def validate_multiple_coefficients(
    real_data: AirfoilData,
    sim_data: pd.DataFrame,
    coefficients: List[str] = ['CL', 'CD']
) -> Dict[str, ValidationResult]:
    """
    Validate multiple coefficients simultaneously.
    
    Args:
        real_data: Real-world data
        sim_data: Simulator output
        coefficients: List of coefficients to validate
        
    Returns:
        Dictionary mapping coefficient name to ValidationResult
    """
    results = {}
    
    for coef in coefficients:
        try:
            result = validate_coefficient_polar(real_data, sim_data, coef)
            results[coef] = result
        except Exception as e:
            print(f"Warning: Could not validate {coef}: {e}")
    
    return results


def generate_validation_report(
    results: Dict[str, ValidationResult],
    output_file: Optional[str] = None
) -> str:
    """
    Generate a text validation report.
    
    Args:
        results: Dictionary of ValidationResults
        output_file: Optional path to save report
        
    Returns:
        Report string
    """
    report_lines = [
        "="*70,
        "VALIDATION REPORT",
        "="*70,
        ""
    ]
    
    # Header info from first result
    if results:
        first_result = next(iter(results.values()))
        report_lines.extend([
            f"Airfoil: {first_result.airfoil_name}",
            f"Reynolds Number: {first_result.reynolds_number:.2e}",
            f"Test Condition: {first_result.test_condition}",
            ""
        ])
    
    # Individual coefficient results
    for coef_name, result in results.items():
        report_lines.extend([
            f"--- {coef_name} Validation ---",
            str(result.metrics),
            f"Pass Status: {'PASS' if result.passes_threshold() else 'FAIL'}",
            ""
        ])
    
    # Summary
    report_lines.extend([
        "="*70,
        "SUMMARY",
        "="*70
    ])
    
    passed = sum(1 for r in results.values() if r.passes_threshold())
    total = len(results)
    
    report_lines.append(f"Passed: {passed}/{total} coefficients")
    
    if passed == total:
        report_lines.append("Overall Status: ✓ PASS")
    else:
        report_lines.append("Overall Status: ✗ FAIL")
    
    report_lines.append("")
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report


def compute_derived_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Compute derived performance metrics from coefficient data.
    
    Args:
        data: DataFrame with 'alpha_deg', 'CL', 'CD' columns
        
    Returns:
        Dictionary of derived metrics
    """
    metrics = {}
    
    # L/D ratio
    CL = data['CL'].values
    CD = data['CD'].values
    
    # Avoid division by zero
    valid_mask = CD > 1e-6
    if np.any(valid_mask):
        LD = CL[valid_mask] / CD[valid_mask]
        max_LD_idx = np.argmax(LD)
        
        metrics['max_LD'] = LD[max_LD_idx]
        metrics['max_LD_alpha'] = data['alpha_deg'].values[valid_mask][max_LD_idx]
        metrics['max_LD_CL'] = CL[valid_mask][max_LD_idx]
    
    # CL max and stall angle
    max_CL_idx = np.argmax(CL)
    metrics['CL_max'] = CL[max_CL_idx]
    metrics['alpha_CLmax'] = data['alpha_deg'].values[max_CL_idx]
    
    # Zero-lift angle
    # Find alpha where CL crosses zero
    if np.any(CL > 0) and np.any(CL < 0):
        from scipy.interpolate import interp1d
        alpha = data['alpha_deg'].values
        try:
            # Interpolate to find zero crossing
            f = interp1d(CL, alpha, bounds_error=False)
            alpha_L0 = f(0.0)
            if np.isfinite(alpha_L0):
                metrics['alpha_L0'] = float(alpha_L0)
        except:
            pass
    
    # CD min
    metrics['CD_min'] = np.min(CD)
    metrics['CD_min_alpha'] = data['alpha_deg'].values[np.argmin(CD)]
    
    return metrics


def compare_derived_metrics(
    real_data: AirfoilData,
    sim_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare derived performance metrics between real and sim.
    
    Args:
        real_data: Real airfoil data
        sim_data: Simulator output
        
    Returns:
        DataFrame with comparison of key metrics
    """
    real_df = real_data.to_dataframe()
    
    real_metrics = compute_derived_metrics(real_df)
    sim_metrics = compute_derived_metrics(sim_data)
    
    # Build comparison table
    comparison = []
    for key in real_metrics.keys():
        if key in sim_metrics:
            real_val = real_metrics[key]
            sim_val = sim_metrics[key]
            error = sim_val - real_val
            rel_error = (error / real_val * 100) if abs(real_val) > 1e-10 else 0
            
            comparison.append({
                'Metric': key,
                'Real': real_val,
                'Sim': sim_val,
                'Error': error,
                'Error_%': rel_error
            })
    
    return pd.DataFrame(comparison)


if __name__ == "__main__":
    # Example usage
    print("Validation Module - Example Usage\n")
    
    from .data_import import create_test_data
    
    # Create test data
    real_data = create_test_data("Test Airfoil")
    
    # Simulate some noisy sim data
    sim_data = real_data.to_dataframe()
    # Add some noise
    np.random.seed(42)
    sim_data['CL'] = sim_data['CL'] + np.random.normal(0, 0.02, len(sim_data))
    sim_data['CD'] = sim_data['CD'] + np.random.normal(0, 0.001, len(sim_data))
    
    # Validate
    print("Validating CL...")
    cl_result = validate_coefficient_polar(real_data, sim_data, 'CL')
    print(cl_result.metrics)
    print(f"\nPasses threshold: {cl_result.passes_threshold()}\n")
    
    # Multiple coefficients
    print("Validating multiple coefficients...")
    results = validate_multiple_coefficients(real_data, sim_data, ['CL', 'CD'])
    
    # Generate report
    report = generate_validation_report(results)
    print(report)
    
    # Derived metrics
    print("\nDerived Performance Metrics:")
    comparison = compare_derived_metrics(real_data, sim_data)
    print(comparison.to_string(index=False))


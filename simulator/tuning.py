"""
Parameter Tuning Module

Automated optimization of simulator parameters to match real-world data.

Uses least-squares and other optimization methods to adjust:
- Stability derivatives (CLa, Cma, etc.)
- Base coefficients (CL0, CD0, Cm0)
- Nonlinear terms (stall characteristics, drag factors)

Includes system identification capability for estimating parameters from
time-series flight test data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import warnings

from .data_import import AirfoilData
from .aircraft import AeroCoefficients
from .validation import compute_metrics, ValidationMetrics


@dataclass
class TuningConfig:
    """Configuration for parameter tuning."""
    
    # Which parameters to tune
    parameters: List[str] = field(default_factory=list)
    
    # Bounds for each parameter (name: (min, max))
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Optimization method
    method: str = 'trf'  # 'trf', 'dogbox', 'lm' for least_squares; 'de' for differential_evolution
    
    # Convergence criteria
    ftol: float = 1e-8
    xtol: float = 1e-8
    max_iterations: int = 1000
    
    # Weighting for multi-objective (e.g., weight CL more than CD)
    weights: Dict[str, float] = field(default_factory=lambda: {'CL': 1.0, 'CD': 1.0, 'Cm': 1.0})
    
    # Regularization (penalize large deviations from initial guess)
    regularization: float = 0.0


@dataclass
class TuningResult:
    """Result of parameter tuning optimization."""
    
    success: bool
    message: str
    
    # Optimized parameters
    optimized_params: Dict[str, float]
    initial_params: Dict[str, float]
    
    # Quality metrics
    initial_error: float
    final_error: float
    improvement: float  # (initial - final) / initial
    
    # Optimization info
    iterations: int
    function_evaluations: int
    
    # Validation metrics after tuning
    validation_metrics: Optional[Dict[str, ValidationMetrics]] = None
    
    def __str__(self) -> str:
        lines = [
            "="*60,
            "TUNING RESULT",
            "="*60,
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Message: {self.message}",
            "",
            f"Initial Error: {self.initial_error:.6f}",
            f"Final Error:   {self.final_error:.6f}",
            f"Improvement:   {self.improvement*100:.2f}%",
            "",
            f"Iterations: {self.iterations}",
            f"Function Evals: {self.function_evaluations}",
            "",
            "Optimized Parameters:"
        ]
        
        for name, value in self.optimized_params.items():
            initial = self.initial_params.get(name, np.nan)
            change = ((value - initial) / initial * 100) if abs(initial) > 1e-10 else 0
            lines.append(f"  {name:15s}: {value:10.6f}  (initial: {initial:10.6f}, change: {change:+6.2f}%)")
        
        if self.validation_metrics:
            lines.extend(["", "Validation Metrics After Tuning:"])
            for coef, metrics in self.validation_metrics.items():
                lines.append(f"  {coef}: RMSE={metrics.rmse:.6f}, R²={metrics.r_squared:.4f}")
        
        return "\n".join(lines)


def create_default_tuning_config(parameters: List[str]) -> TuningConfig:
    """
    Create default tuning configuration with reasonable bounds.
    
    Args:
        parameters: List of parameter names to tune
        
    Returns:
        TuningConfig with default bounds
    """
    # Default bounds for common parameters
    default_bounds = {
        # Lift
        'CL0': (-0.5, 0.5),
        'CLa': (3.0, 7.0),
        'CLq': (0.0, 10.0),
        'CLde': (0.0, 1.5),
        
        # Drag
        'CD0': (0.005, 0.1),
        'CD_k': (0.01, 0.15),
        'CDa': (0.0, 0.5),
        'CDb': (0.0, 0.5),
        
        # Pitching moment
        'Cm0': (-0.2, 0.2),
        'Cma': (-2.0, 0.0),
        'Cmq': (-30.0, 0.0),
        'Cmde': (-2.0, 0.0),
        
        # Side force
        'CYb': (-1.0, 0.0),
        'CYdr': (0.0, 0.5),
        'CYp': (-0.5, 0.5),
        'CYr': (0.0, 1.0),
        
        # Rolling moment
        'Clb': (-0.3, 0.0),
        'Clp': (-1.0, 0.0),
        'Clr': (0.0, 0.5),
        'Clda': (0.0, 0.5),
        'Cldr': (0.0, 0.1),
        
        # Yawing moment
        'Cnb': (0.0, 0.5),
        'Cnp': (-0.2, 0.2),
        'Cnr': (-0.5, 0.0),
        'Cnda': (-0.1, 0.1),
        'Cndr': (-0.3, 0.0),
        
        # Stall characteristics
        'alpha_stall': (0.15, 0.35),  # radians (~8-20 deg)
        'CL_max': (1.0, 2.0),
        'CL_stall_drop': (0.0, 0.5)
    }
    
    bounds = {p: default_bounds.get(p, (-10.0, 10.0)) for p in parameters}
    
    return TuningConfig(
        parameters=parameters,
        bounds=bounds,
        method='trf',
        weights={'CL': 1.0, 'CD': 1.0, 'Cm': 0.5}  # Weight CL and CD more
    )


def tune_to_polar_data(
    real_data: AirfoilData,
    aero_coeffs: AeroCoefficients,
    config: Optional[TuningConfig] = None,
    verbose: bool = True
) -> TuningResult:
    """
    Tune aerodynamic coefficients to match real polar data.
    
    This is a static (steady-state) tuning problem where we adjust
    parameters to minimize error in CL, CD, Cm vs alpha curves.
    
    Args:
        real_data: Real-world airfoil polar data
        aero_coeffs: Initial AeroCoefficients object (will be modified)
        config: Tuning configuration
        verbose: Print progress
        
    Returns:
        TuningResult with optimized parameters
    """
    if config is None:
        # Auto-configure for basic CL/CD tuning
        config = create_default_tuning_config(['CL0', 'CLa', 'CD0', 'CD_k'])
    
    # Extract initial parameter values
    initial_params = {name: getattr(aero_coeffs, name) for name in config.parameters}
    
    # Extract real data
    alpha_rad = np.deg2rad(real_data.alpha)
    real_CL = real_data.CL
    real_CD = real_data.CD
    real_Cm = real_data.Cm if real_data.Cm is not None else None
    
    # Setup bounds
    lower_bounds = [config.bounds[p][0] for p in config.parameters]
    upper_bounds = [config.bounds[p][1] for p in config.parameters]
    
    # Define residual function
    def compute_residuals(param_values):
        """Compute weighted residuals between sim and real."""
        # Update aero coefficients with new parameters
        for name, value in zip(config.parameters, param_values):
            setattr(aero_coeffs, name, value)
        
        # Compute simulated coefficients
        sim_CL = []
        sim_CD = []
        sim_Cm = []
        
        for alpha in alpha_rad:
            # Compute CL (with stall model)
            CL = aero_coeffs.get_CL(alpha, q_hat=0.0, delta_e=0.0)
            sim_CL.append(CL)
            
            # Compute CD
            CD = aero_coeffs.get_CD(CL, alpha, beta=0.0)
            sim_CD.append(CD)
            
            # Compute Cm
            if real_Cm is not None:
                Cm = aero_coeffs.get_Cm(alpha, q_hat=0.0, delta_e=0.0)
                sim_Cm.append(Cm)
        
        sim_CL = np.array(sim_CL)
        sim_CD = np.array(sim_CD)
        
        # Compute weighted residuals
        residuals = []
        
        # CL residuals
        cl_weight = config.weights.get('CL', 1.0)
        residuals.extend((sim_CL - real_CL) * cl_weight)
        
        # CD residuals
        cd_weight = config.weights.get('CD', 1.0)
        residuals.extend((sim_CD - real_CD) * cd_weight)
        
        # Cm residuals (if available)
        if real_Cm is not None:
            sim_Cm = np.array(sim_Cm)
            cm_weight = config.weights.get('Cm', 1.0)
            residuals.extend((sim_Cm - real_Cm) * cm_weight)
        
        # Regularization (penalize deviation from initial guess)
        if config.regularization > 0:
            for i, name in enumerate(config.parameters):
                deviation = (param_values[i] - initial_params[name]) / initial_params[name]
                residuals.append(deviation * config.regularization)
        
        return np.array(residuals)
    
    # Compute initial error
    initial_residuals = compute_residuals([initial_params[p] for p in config.parameters])
    initial_error = np.sqrt(np.mean(initial_residuals**2))
    
    if verbose:
        print(f"Initial RMSE: {initial_error:.6f}")
        print(f"Tuning {len(config.parameters)} parameters...")
    
    # Optimize
    if config.method == 'de':
        # Differential evolution (global optimization)
        def objective(x):
            res = compute_residuals(x)
            return np.sum(res**2)
        
        result = differential_evolution(
            objective,
            bounds=list(zip(lower_bounds, upper_bounds)),
            maxiter=config.max_iterations,
            tol=config.ftol,
            seed=42,
            disp=verbose
        )
        
        optimized_values = result.x
        success = result.success
        message = result.message
        iterations = result.nit
        function_evals = result.nfev
    else:
        # Least squares (local optimization)
        result = least_squares(
            compute_residuals,
            x0=[initial_params[p] for p in config.parameters],
            bounds=(lower_bounds, upper_bounds),
            method=config.method,
            ftol=config.ftol,
            xtol=config.xtol,
            max_nfev=config.max_iterations,
            verbose=2 if verbose else 0
        )
        
        optimized_values = result.x
        success = result.success
        message = result.message if hasattr(result, 'message') else str(result.status)
        iterations = result.nfev
        function_evals = result.nfev
    
    # Compute final error
    final_residuals = compute_residuals(optimized_values)
    final_error = np.sqrt(np.mean(final_residuals**2))
    
    # Update aero_coeffs with optimized values
    optimized_params = {}
    for name, value in zip(config.parameters, optimized_values):
        setattr(aero_coeffs, name, value)
        optimized_params[name] = value
    
    improvement = (initial_error - final_error) / initial_error if initial_error > 0 else 0.0
    
    if verbose:
        print(f"\nFinal RMSE: {final_error:.6f}")
        print(f"Improvement: {improvement*100:.2f}%")
    
    # Compute validation metrics for each coefficient
    from .validation import compute_metrics
    
    # Re-compute with final parameters
    sim_CL = np.array([aero_coeffs.get_CL(a, 0, 0) for a in alpha_rad])
    sim_CD = np.array([aero_coeffs.get_CD(sim_CL[i], alpha_rad[i], 0) 
                       for i in range(len(alpha_rad))])
    
    validation_metrics = {
        'CL': compute_metrics(real_CL, sim_CL),
        'CD': compute_metrics(real_CD, sim_CD)
    }
    
    if real_Cm is not None:
        sim_Cm = np.array([aero_coeffs.get_Cm(a, 0, 0) for a in alpha_rad])
        validation_metrics['Cm'] = compute_metrics(real_Cm, sim_Cm)
    
    return TuningResult(
        success=success,
        message=message,
        optimized_params=optimized_params,
        initial_params=initial_params,
        initial_error=initial_error,
        final_error=final_error,
        improvement=improvement,
        iterations=iterations,
        function_evaluations=function_evals,
        validation_metrics=validation_metrics
    )


def tune_with_sensitivity_analysis(
    real_data: AirfoilData,
    aero_coeffs: AeroCoefficients,
    test_parameters: List[str],
    perturbation: float = 0.01
) -> pd.DataFrame:
    """
    Perform sensitivity analysis to identify most influential parameters.
    
    Useful for deciding which parameters to tune.
    
    Args:
        real_data: Real-world data
        aero_coeffs: Current aero coefficients
        test_parameters: Parameters to test
        perturbation: Fractional perturbation (e.g., 0.01 = 1%)
        
    Returns:
        DataFrame with sensitivity results
    """
    from .validation import compute_metrics
    
    alpha_rad = np.deg2rad(real_data.alpha)
    real_CL = real_data.CL
    real_CD = real_data.CD
    
    # Baseline error
    sim_CL_base = np.array([aero_coeffs.get_CL(a, 0, 0) for a in alpha_rad])
    sim_CD_base = np.array([aero_coeffs.get_CD(sim_CL_base[i], alpha_rad[i], 0) 
                           for i in range(len(alpha_rad))])
    
    baseline_CL_error = compute_metrics(real_CL, sim_CL_base).rmse
    baseline_CD_error = compute_metrics(real_CD, sim_CD_base).rmse
    
    results = []
    
    for param_name in test_parameters:
        # Save original value
        original_value = getattr(aero_coeffs, param_name)
        
        # Perturb upward
        perturbed_value = original_value * (1 + perturbation)
        setattr(aero_coeffs, param_name, perturbed_value)
        
        # Re-compute
        sim_CL_pert = np.array([aero_coeffs.get_CL(a, 0, 0) for a in alpha_rad])
        sim_CD_pert = np.array([aero_coeffs.get_CD(sim_CL_pert[i], alpha_rad[i], 0) 
                               for i in range(len(alpha_rad))])
        
        pert_CL_error = compute_metrics(real_CL, sim_CL_pert).rmse
        pert_CD_error = compute_metrics(real_CD, sim_CD_pert).rmse
        
        # Sensitivity: change in error per unit change in parameter
        CL_sensitivity = (pert_CL_error - baseline_CL_error) / (perturbed_value - original_value)
        CD_sensitivity = (pert_CD_error - baseline_CD_error) / (perturbed_value - original_value)
        
        results.append({
            'Parameter': param_name,
            'Original_Value': original_value,
            'CL_Sensitivity': abs(CL_sensitivity),
            'CD_Sensitivity': abs(CD_sensitivity),
            'Total_Sensitivity': abs(CL_sensitivity) + abs(CD_sensitivity)
        })
        
        # Restore original
        setattr(aero_coeffs, param_name, original_value)
    
    df = pd.DataFrame(results)
    df = df.sort_values('Total_Sensitivity', ascending=False)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Parameter Tuning Module - Example Usage\n")
    
    from .data_import import create_test_data
    from .aircraft import AeroCoefficients
    
    # Create test data
    real_data = create_test_data("Test Airfoil")
    
    # Create initial aero coefficients (intentionally wrong)
    aero = AeroCoefficients(
        CL0=0.2,      # Wrong (should be ~0.0)
        CLa=4.0,      # Wrong (should be ~6.3 for 0.11 per deg)
        CD0=0.02,     # Wrong (should be ~0.008)
        CD_k=0.08     # Wrong (should be ~0.05)
    )
    
    print("Initial parameters:")
    print(f"  CL0 = {aero.CL0:.4f}")
    print(f"  CLa = {aero.CLa:.4f}")
    print(f"  CD0 = {aero.CD0:.4f}")
    print(f"  CD_k = {aero.CD_k:.4f}\n")
    
    # Tune
    config = create_default_tuning_config(['CL0', 'CLa', 'CD0', 'CD_k'])
    result = tune_to_polar_data(real_data, aero, config, verbose=True)
    
    print("\n" + str(result))
    
    # Sensitivity analysis
    print("\n\nSensitivity Analysis:")
    sensitivity = tune_with_sensitivity_analysis(
        real_data, aero, 
        ['CL0', 'CLa', 'CLq', 'CLde', 'CD0', 'CD_k']
    )
    print(sensitivity.to_string(index=False))


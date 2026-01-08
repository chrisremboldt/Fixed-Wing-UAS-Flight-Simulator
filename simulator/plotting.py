"""
Plotting Module

Generate standard aerospace validation plots:
- Coefficient polars (CL, CD, Cm vs alpha)
- Drag polar (CL vs CD)
- L/D curve
- Trim curves
- Maneuver time histories
- Comparison plots (sim vs real)

Uses matplotlib with aerospace-standard formatting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from .data_import import AirfoilData
from .validation import ValidationResult


# Aerospace standard plot styling
PLOT_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'grid.alpha': 0.3
}


def setup_plot_style():
    """Apply aerospace-standard plot styling."""
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.rcParams.update(PLOT_STYLE)


def plot_coefficient_comparison(
    real_data: AirfoilData,
    sim_data: pd.DataFrame,
    coefficient: str = 'CL',
    ax: Optional[plt.Axes] = None,
    show_error: bool = True
) -> plt.Axes:
    """
    Plot comparison of a single coefficient vs alpha.
    
    Args:
        real_data: Real-world data
        sim_data: Simulator data (must have 'alpha_deg' and coefficient columns)
        coefficient: Which coefficient to plot
        ax: Existing axes (creates new if None)
        show_error: Show error band
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # Get data
    real_alpha = real_data.alpha
    
    if coefficient == 'CL':
        real_coef = real_data.CL
        ylabel = r'$C_L$'
    elif coefficient == 'CD':
        real_coef = real_data.CD
        ylabel = r'$C_D$'
    elif coefficient == 'Cm':
        if real_data.Cm is None:
            raise ValueError("Real data does not have Cm")
        real_coef = real_data.Cm
        ylabel = r'$C_m$'
    else:
        raise ValueError(f"Unknown coefficient: {coefficient}")
    
    sim_alpha = sim_data['alpha_deg'].values
    sim_coef = sim_data[coefficient].values
    
    # Plot
    ax.plot(real_alpha, real_coef, 'o-', label='Real Data', color='black', markersize=4)
    ax.plot(sim_alpha, sim_coef, 's--', label='Simulator', color='red', markersize=4)
    
    # Error band
    if show_error:
        from scipy.interpolate import interp1d
        # Interpolate sim to real alpha
        f = interp1d(sim_alpha, sim_coef, bounds_error=False, fill_value='extrapolate')
        sim_interp = f(real_alpha)
        error = sim_interp - real_coef
        
        ax.fill_between(real_alpha, real_coef, sim_interp, 
                        alpha=0.2, color='red', label='Error')
    
    ax.set_xlabel(r'$\alpha$ (deg)')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add zero line for Cm
    if coefficient == 'Cm':
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    return ax


def plot_polar_comparison(
    real_data: AirfoilData,
    sim_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive polar comparison plot (CL, CD, Cm vs alpha).
    
    Args:
        real_data: Real-world data
        sim_data: Simulator data
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    # Determine layout based on available data
    has_Cm = real_data.Cm is not None and 'Cm' in sim_data.columns
    
    if has_Cm:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # CL vs alpha
    plot_coefficient_comparison(real_data, sim_data, 'CL', ax=axes[0])
    axes[0].set_title('Lift Coefficient')
    
    # CD vs alpha
    plot_coefficient_comparison(real_data, sim_data, 'CD', ax=axes[1])
    axes[1].set_title('Drag Coefficient')
    
    if has_Cm:
        # Cm vs alpha
        plot_coefficient_comparison(real_data, sim_data, 'Cm', ax=axes[2])
        axes[2].set_title('Pitching Moment Coefficient')
        
        # Drag polar (CL vs CD)
        axes[3].plot(real_data.CD, real_data.CL, 'o-', label='Real', color='black', markersize=4)
        axes[3].plot(sim_data['CD'].values, sim_data['CL'].values, 
                    's--', label='Sim', color='red', markersize=4)
        axes[3].set_xlabel(r'$C_D$')
        axes[3].set_ylabel(r'$C_L$')
        axes[3].set_title('Drag Polar')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    fig.suptitle(f'Airfoil: {real_data.name}, Re = {real_data.reynolds_number:.2e}')
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_validation_results(
    results: Dict[str, ValidationResult],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot validation results with metrics.
    
    Args:
        results: Dictionary of ValidationResults (from validation module)
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for ax, (coef_name, result) in zip(axes, results.items()):
        # Plot data
        ax.plot(result.independent_var, result.real_values, 
               'o-', label='Real', color='black', markersize=4)
        ax.plot(result.independent_var, result.sim_values,
               's--', label='Sim', color='red', markersize=4)
        
        # Add metrics text box
        metrics_text = (
            f"RMSE: {result.metrics.rmse:.4f}\n"
            f"R²: {result.metrics.r_squared:.4f}\n"
            f"Max Error: {result.metrics.max_error:.4f}"
        )
        ax.text(0.02, 0.98, metrics_text,
               transform=ax.transAxes,
               fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Pass/fail indicator
        passes = result.passes_threshold()
        status_text = "✓ PASS" if passes else "✗ FAIL"
        status_color = 'green' if passes else 'red'
        ax.text(0.98, 0.98, status_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               horizontalalignment='right',
               color=status_color,
               weight='bold')
        
        ax.set_xlabel(r'$\alpha$ (deg)')
        ax.set_ylabel(f'${coef_name}$')
        ax.set_title(f'{coef_name} Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved validation plot to {save_path}")
    
    return fig


def plot_drag_polar(
    real_data: AirfoilData,
    sim_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot drag polar (CL vs CD) with L/D annotations.
    
    Args:
        real_data: Real data
        sim_data: Simulator data
        save_path: Optional save path
        
    Returns:
        Figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot polars
    ax.plot(real_data.CD, real_data.CL, 'o-', 
           label='Real Data', color='black', markersize=5)
    ax.plot(sim_data['CD'].values, sim_data['CL'].values, 
           's--', label='Simulator', color='red', markersize=5)
    
    # Find and annotate max L/D points
    for data, color, label in [(real_data, 'black', 'Real'), 
                                 (sim_data, 'red', 'Sim')]:
        if isinstance(data, AirfoilData):
            CL = data.CL
            CD = data.CD
        else:
            CL = data['CL'].values
            CD = data['CD'].values
        
        valid = CD > 1e-6
        LD = CL[valid] / CD[valid]
        max_idx = np.argmax(LD)
        
        max_LD = LD[max_idx]
        CL_max_LD = CL[valid][max_idx]
        CD_max_LD = CD[valid][max_idx]
        
        ax.plot(CD_max_LD, CL_max_LD, '*', color=color, markersize=12)
        ax.annotate(f'{label}: L/D={max_LD:.1f}',
                   xy=(CD_max_LD, CL_max_LD),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9,
                   color=color,
                   arrowprops=dict(arrowstyle='->', color=color, lw=1))
    
    ax.set_xlabel(r'$C_D$')
    ax.set_ylabel(r'$C_L$')
    ax.set_title(f'Drag Polar: {real_data.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_LD_curve(
    real_data: AirfoilData,
    sim_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot L/D ratio vs alpha.
    
    Args:
        real_data: Real data
        sim_data: Simulator data
        save_path: Optional save path
        
    Returns:
        Figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute L/D
    real_LD = real_data.CL / np.maximum(real_data.CD, 1e-6)
    sim_LD = sim_data['CL'].values / np.maximum(sim_data['CD'].values, 1e-6)
    
    # Plot
    ax.plot(real_data.alpha, real_LD, 'o-', label='Real', color='black', markersize=4)
    ax.plot(sim_data['alpha_deg'].values, sim_LD, 's--', label='Sim', color='red', markersize=4)
    
    # Annotate max values
    max_real = np.max(real_LD)
    alpha_max_real = real_data.alpha[np.argmax(real_LD)]
    ax.axhline(max_real, color='black', linestyle=':', alpha=0.5)
    ax.text(ax.get_xlim()[1]*0.7, max_real, f'Max L/D = {max_real:.1f}',
           fontsize=9, color='black')
    
    ax.set_xlabel(r'$\alpha$ (deg)')
    ax.set_ylabel('L/D')
    ax.set_title(f'Lift-to-Drag Ratio: {real_data.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_maneuver_time_history(
    history: List[Dict],
    variables: List[str] = ['alpha_deg', 'theta_deg', 'q_deg_s', 'elevator_deg'],
    title: str = "Maneuver Time History",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time history of maneuver variables.
    
    Args:
        history: Simulation history
        variables: List of variables to plot
        title: Plot title
        save_path: Optional save path
        
    Returns:
        Figure
    """
    setup_plot_style()
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 2.5*n_vars), sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    # Extract data
    df = pd.DataFrame(history)
    time = df['time'].values
    
    var_labels = {
        'alpha_deg': r'$\alpha$ (deg)',
        'beta_deg': r'$\beta$ (deg)',
        'theta_deg': r'$\theta$ (deg)',
        'phi_deg': r'$\phi$ (deg)',
        'psi_deg': r'$\psi$ (deg)',
        'p_deg_s': r'$p$ (deg/s)',
        'q_deg_s': r'$q$ (deg/s)',
        'r_deg_s': r'$r$ (deg/s)',
        'elevator_deg': r'$\delta_e$ (deg)',
        'aileron_deg': r'$\delta_a$ (deg)',
        'rudder_deg': r'$\delta_r$ (deg)',
        'airspeed': 'Airspeed (m/s)',
        'altitude': 'Altitude (m)'
    }
    
    for ax, var in zip(axes, variables):
        if var in df.columns:
            data = df[var].values
        else:
            # Compute from other data if needed
            if var == 'alpha_deg' and 'alpha' in df.columns:
                data = np.degrees(df['alpha'].values)
            elif var == 'altitude' and 'position' in df.columns:
                data = -np.array([p[2] for p in df['position'].values])
            else:
                print(f"Warning: Variable {var} not found in history")
                continue
        
        ax.plot(time, data, 'b-', linewidth=1.5)
        ax.set_ylabel(var_labels.get(var, var))
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_validation_report_plots(
    real_data: AirfoilData,
    sim_data: pd.DataFrame,
    results: Dict[str, ValidationResult],
    output_dir: str,
    prefix: str = "validation"
) -> None:
    """
    Generate complete set of validation plots and save to directory.
    
    Args:
        real_data: Real airfoil data
        sim_data: Simulator data
        results: Validation results
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Polar comparison
    fig = plot_polar_comparison(real_data, sim_data)
    fig.savefig(output_path / f"{prefix}_polar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Validation results with metrics
    fig = plot_validation_results(results)
    fig.savefig(output_path / f"{prefix}_validation_metrics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Drag polar
    fig = plot_drag_polar(real_data, sim_data)
    fig.savefig(output_path / f"{prefix}_drag_polar.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # L/D curve
    fig = plot_LD_curve(real_data, sim_data)
    fig.savefig(output_path / f"{prefix}_LD_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nValidation plots saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Plotting Module - Example Usage\n")
    
    from .data_import import create_test_data
    
    # Create test data
    real_data = create_test_data("Test Airfoil")
    sim_data = real_data.to_dataframe()
    
    # Add some noise to sim
    np.random.seed(42)
    sim_data['CL'] = sim_data['CL'] + np.random.normal(0, 0.02, len(sim_data))
    sim_data['CD'] = sim_data['CD'] + np.random.normal(0, 0.001, len(sim_data))
    
    setup_plot_style()
    
    # Create plots
    print("Creating polar comparison...")
    fig = plot_polar_comparison(real_data, sim_data)
    plt.show()
    
    print("\nDone! Close plot windows to exit.")


"""
Fixed-Wing UAS Flight Simulator

A modular, physics-based flight dynamics engine for testing DAA algorithms.
Includes validation and tuning capabilities using real-world airfoil data.
"""

__version__ = "0.2.0"

# Core simulation modules
from .aircraft import AircraftConfig
from .state import AircraftState, ControlInputs
from .dynamics import FlightDynamics, SimulationConfig
from .environment import Environment

# Validation and tuning modules
from .data_import import (
    AirfoilData,
    load_uiuc_dat,
    load_generic_csv,
    load_airfoiltools_csv,
    load_kanakaero_csv,
    create_test_data,
    compute_reynolds_number,
    match_reynolds_conditions
)

from .validation import (
    ValidationMetrics,
    ValidationResult,
    compute_metrics,
    validate_coefficient_polar,
    validate_multiple_coefficients,
    generate_validation_report,
    compare_derived_metrics
)

from .tuning import (
    TuningConfig,
    TuningResult,
    create_default_tuning_config,
    tune_to_polar_data,
    tune_with_sensitivity_analysis
)

from .data_export import (
    export_nasa_csv,
    export_polar_csv,
    export_maneuver_csv,
    export_trim_data,
    export_json,
    create_validation_package
)

from .plotting import (
    plot_coefficient_comparison,
    plot_polar_comparison,
    plot_validation_results,
    plot_drag_polar,
    plot_LD_curve,
    plot_maneuver_time_history,
    create_validation_report_plots
)


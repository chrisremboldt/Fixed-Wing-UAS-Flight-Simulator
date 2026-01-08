#!/usr/bin/env python3
"""
Helper script to download airfoil performance data from various sources.
"""

import sys
import argparse
from pathlib import Path
import urllib.request
import urllib.error


AIRFOIL_TOOLS_POLARS = {
    # Format: 'name': 'polar_id'
    'naca0012_re100k': 'xf-naca0012-il-100000',
    'naca0012_re200k': 'xf-naca0012-il-200000',
    'naca0012_re500k': 'xf-naca0012-il-500000',
    'naca0012_re1000k': 'xf-naca0012-il-1000000',
    
    'naca2412_re100k': 'xf-naca2412-il-100000',
    'naca2412_re200k': 'xf-naca2412-il-200000',
    'naca2412_re500k': 'xf-naca2412-il-500000',
    
    'naca4412_re100k': 'xf-naca4412-il-100000',
    'naca4412_re200k': 'xf-naca4412-il-200000',
    'naca4412_re500k': 'xf-naca4412-il-500000',
    
    'clark-y_re500k': 'xf-clarky-il-500000',
    'eppler-193_re500k': 'xf-e193-il-500000',
    's1223_re200k': 'xf-s1223-il-200000',
}


def download_from_airfoiltools(polar_name: str, output_file: Path) -> bool:
    """
    Download polar data from Airfoil Tools.
    
    Args:
        polar_name: Name from AIRFOIL_TOOLS_POLARS
        output_file: Where to save
        
    Returns:
        True if successful
    """
    if polar_name not in AIRFOIL_TOOLS_POLARS:
        print(f"Error: '{polar_name}' not in database.")
        print(f"Available: {', '.join(AIRFOIL_TOOLS_POLARS.keys())}")
        return False
    
    polar_id = AIRFOIL_TOOLS_POLARS[polar_name]
    url = f"http://airfoiltools.com/polar/csv?polar={polar_id}"
    
    print(f"Downloading {polar_name}...")
    print(f"  URL: {url}")
    print(f"  Saving to: {output_file}")
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"  ✓ Success! ({output_file.stat().st_size} bytes)")
        return True
    except urllib.error.URLError as e:
        print(f"  ✗ Failed: {e}")
        return False


def list_available():
    """List all available polars."""
    print("Available Airfoil Polars from Airfoil Tools:\n")
    
    current_airfoil = None
    for name in sorted(AIRFOIL_TOOLS_POLARS.keys()):
        airfoil = name.rsplit('_', 1)[0]
        if airfoil != current_airfoil:
            if current_airfoil is not None:
                print()
            print(f"{airfoil}:")
            current_airfoil = airfoil
        re_num = name.split('_')[-1]
        print(f"  - {name:30s} ({re_num})")


def download_common_set(output_dir: Path):
    """Download a common set of airfoils useful for UAS."""
    print("Downloading common UAS airfoils...\n")
    
    common = [
        'naca0012_re200k',  # Symmetric, low Re
        'naca2412_re200k',  # Cambered, low Re
        'naca4412_re200k',  # High camber
        's1223_re200k',     # High-lift, low Re
    ]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for name in common:
        output_file = output_dir / f"{name}.csv"
        if download_from_airfoiltools(name, output_file):
            success_count += 1
        print()
    
    print(f"\nDownloaded {success_count}/{len(common)} files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download airfoil performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available data
  python download_airfoil_data.py --list
  
  # Download specific airfoil
  python download_airfoil_data.py naca0012_re500k
  
  # Download common set for UAS testing
  python download_airfoil_data.py --common
  
  # Download with custom output location
  python download_airfoil_data.py naca2412_re200k --output my_data/naca2412.csv
        """
    )
    
    parser.add_argument(
        'polar_name',
        nargs='?',
        help='Name of polar to download (see --list)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available polars'
    )
    parser.add_argument(
        '--common', '-c',
        action='store_true',
        help='Download common set of UAS airfoils'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (default: data/airfoiltools/<name>.csv)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available()
        return
    
    if args.common:
        output_dir = Path('data/airfoiltools')
        download_common_set(output_dir)
        return
    
    if not args.polar_name:
        parser.error("Polar name required (or use --list or --common)")
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f'data/airfoiltools/{args.polar_name}.csv')
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Download
    if download_from_airfoiltools(args.polar_name, output_file):
        print(f"\n✓ Ready to use with:")
        print(f"  python run_validation_workflow.py --data {output_file}")


if __name__ == "__main__":
    main()


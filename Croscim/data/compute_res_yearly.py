import sys
from pathlib import Path
from tqdm import tqdm
import subprocess

def process_one_file(nc_file):
    basename = nc_file.stem.replace('_x1', '')
    x3_file = nc_file.parent / f"{basename}_x3.zarr"
    x10_file = nc_file.parent / f"{basename}_x10.zarr"

    if x3_file.exists() and x10_file.exists():
        print(f"{nc_file.stem}: already processed")
        return []
    script_path = Path(__file__).parent / "compute_res_daily.py"
    cmd = [sys.executable, str(script_path), str(nc_file), "--quiet"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        created = []
        if x3_file.exists():
            created.append('x3')
        if x10_file.exists():
            created.append('x10')
        return created
    else:
        raise RuntimeError(f"Error: {result.stderr[:200]}")


def process_year(year):
    """
    Traite tous les fichiers NetCDF d'une année.
    """
    data_dir = Path(f'/dmidata/users/malegu/data/netcdf_{year}')
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    nc_files = sorted(data_dir.glob('*_x1.nc'))
    if not nc_files:
        print(f"No NetCDF files found in {data_dir}")
        return
    for nc_file in tqdm(nc_files, desc=f"Processing {year}", unit="file"):
        try:
            created = process_one_file(nc_file)
            if created:
                tqdm.write(f"{nc_file.stem}: {', '.join(created)} created")
        except Exception as e:
            tqdm.write(f"Error processing {nc_file.stem}: {e}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python compute_res_yearly.py YEAR")
        sys.exit(1)
    year = int(sys.argv[1])
    process_year(year)
    print(f"Processing for year {year} completed.")

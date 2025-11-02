import sys
from pathlib import Path
from tqdm import tqdm
import subprocess

def process_one_file(nc_file, zarr_output_dir=None):
    basename = nc_file.stem.replace('_x1', '')

    # Déterminer le dossier de sortie pour les zarr
    if zarr_output_dir is None:
        year = basename[:4]
        zarr_output_dir = Path(f'/dmidata/projects/4dvarnet/data_{year}')
    else:
        zarr_output_dir = Path(zarr_output_dir)

    x3_file = zarr_output_dir / f"{basename}_x3.zarr"
    x10_file = zarr_output_dir / f"{basename}_x10.zarr"

    if x3_file.exists() and x10_file.exists():
        print(f"{nc_file.stem}: already processed")
        return []
    script_path = Path(__file__).parent / "compute_res_daily.py"
    cmd = [sys.executable, str(script_path), str(nc_file), "-o", str(zarr_output_dir), "--quiet"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    created = []
    if x3_file.exists():
        created.append('x3')
    if x10_file.exists():
        created.append('x10')
    if created:
        return created
    else:
        raise RuntimeError(f"Error: {result.stderr[:200]}")


def process_year(year, nb_workers=1):
    """
    Traite tous les fichiers NetCDF d'une année.
    nb_workers : nombre de processus parallèles (1 = séquentiel)
    """
    # NetCDF dans espace personnel, zarr dans espace projet
    nc_dir = Path(f'/dmidata/users/malegu/netcdf_{year}')
    zarr_dir = Path(f'/dmidata/projects/4dvarnet/data_{year}')

    if not nc_dir.exists():
        raise FileNotFoundError(f"Directory not found: {nc_dir}")
    zarr_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(nc_dir.glob('*_x1.nc'))
    if not nc_files:
        print(f"No NetCDF files found in {nc_dir}")
        return

    if nb_workers > 1:
        from multiprocessing import Pool
        from functools import partial
        print(f"Traitement parallèle: {len(nc_files)} fichiers avec {nb_workers} workers")
        process_with_zarr_dir = partial(process_one_file, zarr_output_dir=zarr_dir)
        with Pool(processes=nb_workers) as pool:
            results = list(tqdm(pool.imap(process_with_zarr_dir, nc_files), total=len(nc_files), desc=f"Year {year}", unit="fichier"))

        success = sum(1 for r in results if r)
        print(f"\nRésumé: {success}/{len(nc_files)} fichiers traités")
    else:
        # Mode séquentiel (original)
        for nc_file in tqdm(nc_files, desc=f"Processing {year}", unit="file"):
            try:
                created = process_one_file(nc_file, zarr_output_dir=zarr_dir)
                if created:
                    tqdm.write(f"{nc_file.stem}: {', '.join(created)} created")
            except Exception as e:
                tqdm.write(f"Error processing {nc_file.stem}: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compute_res_yearly.py YEAR [--parallel NB_WORKERS]")
        sys.exit(1)

    year = int(sys.argv[1])
    nb_workers = 1

    if '--parallel' in sys.argv:
        idx = sys.argv.index('--parallel')
        nb_workers = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 4

    process_year(year, nb_workers)
    print(f"Processing for year {year} completed.")

import sys
from pathlib import Path
from tqdm import tqdm
import subprocess


def is_complete_zarr(path):
    path = Path(path)
    return path.is_dir() and (path / ".zmetadata").is_file()


def process_one_file(nc_file, output_dir=None, save_format='zarr'):
    """
    Génère x3 et x10 pour un fichier x1.

    Args:
        nc_file: Fichier x1.nc
        output_dir: Dossier de sortie (défaut: /nwp/sst_malegu/)
        save_format: 'netcdf', 'zarr' ou 'both' (défaut: 'netcdf')
    """
    basename = nc_file.stem.replace('_x1', '')

    # Dossier de sortie sur SSD
    if output_dir is None:
        year = basename[:4]
        output_dir = Path(f'/nwp/sst_malegu/data_{year}')
    else:
        output_dir = Path(output_dir)

    # Vérifier existence selon le format
    x3_nc = output_dir / f"{basename}_x3.nc"
    x10_nc = output_dir / f"{basename}_x10.nc"
    x3_zarr = output_dir / f"{basename}_x3.zarr"
    x10_zarr = output_dir / f"{basename}_x10.zarr"

    # Check si déjà généré
    if save_format == 'netcdf' and x3_nc.exists() and x10_nc.exists():
        return []
    elif save_format == 'zarr' and is_complete_zarr(x3_zarr) and is_complete_zarr(x10_zarr):
        return []
    elif save_format == 'both' and all([x3_nc.exists(), x10_nc.exists(), is_complete_zarr(x3_zarr), is_complete_zarr(x10_zarr)]):
        return []

    script_path = Path(__file__).parent / "compute_res_daily.py"
    cmd = [sys.executable, str(script_path), str(nc_file), "-o", str(output_dir), "--quiet", "--format", save_format]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error: {result.stderr[:500]}")

    created = []
    if save_format in ('netcdf', 'both'):
        if x3_nc.exists():
            created.append('x3.nc')
        if x10_nc.exists():
            created.append('x10.nc')
    if save_format in ('zarr', 'both'):
        if is_complete_zarr(x3_zarr):
            created.append('x3.zarr')
        if is_complete_zarr(x10_zarr):
            created.append('x10.zarr')

    expected_count = 2 if save_format in ('netcdf', 'zarr') else 4
    if len(created) != expected_count:
        raise RuntimeError(f"Incomplete outputs for {basename}: {created}")
    return created


def process_year(year, nb_workers=1, save_format='netcdf', output_dir=None):
    """
    Traite tous les fichiers NetCDF d'une année.
    nb_workers : nombre de processus parallèles (1 = séquentiel)
    save_format: 'netcdf', 'zarr' ou 'both'
    output_dir: dossier de sortie (défaut: /nwp/sst_malegu/data_{year})
    """
    # Dossier où chercher les x1.zarr (même dossier que la sortie)
    if output_dir is None:
        output_dir = Path(f'/nwp/sst_malegu/data_{year}')
    else:
        output_dir = Path(output_dir)

    zarr_dir = output_dir  # x1, x3, x10 tous dans le même dossier

    if not zarr_dir.exists():
        raise FileNotFoundError(f"Directory not found: {zarr_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(zarr_dir.glob('*_x1.zarr'))
    if not nc_files:
        print(f"No NetCDF files found in {zarr_dir}")
        return

    if nb_workers > 1:
        from multiprocessing import Pool
        from functools import partial
        print(f"Traitement parallèle: {len(nc_files)} fichiers avec {nb_workers} workers")
        process_with_params = partial(process_one_file, output_dir=output_dir, save_format=save_format)
        with Pool(processes=nb_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_with_params, nc_files), total=len(nc_files), desc=f"Year {year}", unit="fichier"))

        success = sum(1 for r in results if r)
        print(f"\nRésumé: {success}/{len(nc_files)} fichiers traités")
    else:
        # Mode séquentiel (original)
        for nc_file in tqdm(nc_files, desc=f"Processing {year}", unit="file"):
            try:
                created = process_one_file(nc_file, output_dir=output_dir, save_format=save_format)
                if created:
                    tqdm.write(f"{nc_file.stem}: {', '.join(created)} created")
            except Exception as e:
                tqdm.write(f"Error processing {nc_file.stem}: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute x3 and x10 resolutions from x1')
    parser.add_argument('year', type=int, help='Year to process')
    parser.add_argument('--parallel', type=int, metavar='N', help='Number of parallel workers (default: 1)')
    parser.add_argument('--save-format', type=str, choices=['netcdf', 'zarr', 'both'], default='netcdf', help='Output format (default: netcdf)')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: /nwp/sst_malegu/data_{YEAR})')

    args = parser.parse_args()

    nb_workers = args.parallel if args.parallel else 1
    output_dir = Path(args.output_dir) if args.output_dir else None

    process_year(args.year, nb_workers=nb_workers, save_format=args.save_format, output_dir=output_dir)
    print(f"Processing for year {args.year} completed.")

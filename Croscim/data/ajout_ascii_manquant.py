import numpy as np
from pathlib import Path

try:
    from .ascii_files import satellite_ascii_candidates
except ImportError:
    from ascii_files import satellite_ascii_candidates


CANONICAL_NAMES = {
    "aasti": "aasti_ist_l2p",
    "avhrr": "avhrr_c3s_l3u",
    "pmw": "pmw_cci_l2p",
    "slstr": "slstr_c3s_l3u",
}

def find_reference_ascii(directory):
    """
    Trouve un fichier ASCII existant dans l'extraction pour copier son header et
    sa géométrie. Utile sur Gefion quand reference_ascii.asc n'existe pas.
    """
    for path in sorted(Path(directory).glob("*/*.asc")):
        return path
    return None

def ajout_ascii_manquant(ascii_path, ascii_similaire):
    """
    Ajoute un fichier ASCII rempli de 99.0 en copiant la structure d'un fichier similaire.
    """
    with open(ascii_similaire, 'r') as f:
        lines = f.readlines()
    header = lines[:3]
    nrows = len(lines) - 3
    ncols = len(lines[3].strip().split())
    data = np.full((nrows, ncols), 99.0)
    with open(ascii_path, 'w') as f:
        f.writelines(header)
        for row in data:
            f.write(' '.join(f"{val:.1f}" for val in row) + '\n')
    print(f"Fichier manquant créé: {ascii_path}")
    
def ajout_multiples_ascii(year, days, source_dir=None, reference_file=None):
    """
    A partir de la liste des jours manquants, on ajoute les fichiers ASCII manquants.
    year : int, année à traiter
    days : liste de la forme [YYYYMMDD12, ...]
    """
    if source_dir is None:
        directory = Path(f'/dmidata/projects/4dvarnet/squash_{year}_extract')
    else:
        directory = Path(source_dir)

    if reference_file is None:
        legacy_reference = Path('/dmidata/projects/4dvarnet/reference_ascii.asc')
        reference_file = legacy_reference if legacy_reference.exists() else find_reference_ascii(directory)
    else:
        reference_file = Path(reference_file)

    if reference_file is None or not reference_file.exists():
        raise FileNotFoundError(f"Aucun fichier ASCII de référence trouvé pour {directory}")

    print(f"Référence utilisée: {reference_file}")

    nb_created = 0
    for day in days:
        day_dir = directory / day
        if not day_dir.exists() or not day_dir.is_dir():
            print(f"Le dossier {day_dir} n'existe pas ou n'est pas un dossier.")
            continue

        for satellite, canonical_name in CANONICAL_NAMES.items():
            for statistic in ("av", "std"):
                candidates = satellite_ascii_candidates(
                    day_dir, day, satellite, statistic
                )
                if len(candidates) > 1:
                    names = ", ".join(path.name for path in candidates)
                    raise RuntimeError(
                        f"Fichiers ambigus pour {satellite}_{statistic} "
                        f"dans {day_dir}: {names}"
                    )
                if candidates:
                    continue

                suffix = "std_av" if statistic == "std" else "av"
                target = day_dir / f"{day}_{canonical_name}_{suffix}.asc"
                ajout_ascii_manquant(target, reference_file)
                nb_created += 1

    print(f"Tous les fichiers manquants ont été ajoutés ({nb_created} fichiers).")
    return

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Create placeholder ASCII files for missing SST satellite files")
    parser.add_argument("year", type=int, help="Year to patch")
    parser.add_argument("--source-dir", type=str, help="Extracted source directory")
    parser.add_argument("--missing-days-file", type=str, help="Missing-days file produced by verif_fichiers.py")
    parser.add_argument("--reference-file", type=str, help="ASCII file used as header/shape reference")
    args = parser.parse_args()

    year = args.year
    file_path = args.missing_days_file or f"/tmp/missing_days_{year}.txt"
    with open(file_path, 'r') as f:
        days = [line.strip() for line in f if line.strip()]
    if not days:
        print("Aucun jour à corriger")
        sys.exit(0)
    print(f"Correction de {len(days)} jours")
    ajout_multiples_ascii(year, days, source_dir=args.source_dir, reference_file=args.reference_file)

from pathlib import Path
from tqdm import tqdm

def verify_extraction(year, return_days=False, source_dir=None):
    """
    Petite fonction pour verifier que dans l'extraction du squash, on a bien tous les fichiers necessaires à la création de nos netcdf
    year : int, année à vérifier
    return_days : bool, si True retourne la liste des jours avec problèmes
    """
    if source_dir is None:
        directory = Path(f'/dmidata/projects/4dvarnet/squash_{year}_extract')
    else:
        directory = Path(source_dir)
    if not directory.exists() or not directory.is_dir():
        print(f"Le dossier {directory} n'existe pas ou n'est pas un dossier.")
        return [] if return_days else None
    day_dirs = [d for d in directory.iterdir() if d.is_dir()]
    nb_jour_pb = 0
    days_with_problems = []
    if not day_dirs:
        print(f"Aucun dossier de jour trouvé dans {directory}.")
        return [] if return_days else None
    for day_dir in tqdm(sorted(day_dirs), desc=f"Vérification extraction {year}", unit="jour"):
        # le soucis technique est que chaque fichier change de nom car il contient le jour, mais a priori ce jour est le meme que le nom du fichier
        # Utilisation de patterns pour supporter les changements de nomenclature (c3s/cci)
        expected_patterns = [
            f"{day_dir.name}_aasti_*av.asc",
            f"{day_dir.name}_aasti_*std_av.asc",
            (f"{day_dir.name}_avhrr_*av.asc", "avhrr_av"),
            (f"{day_dir.name}_avhrr_*std_av.asc", "avhrr_std"),
            f"{day_dir.name}_pmw_*av.asc",
            f"{day_dir.name}_pmw_*std_av.asc",
            (f"{day_dir.name}_slstr_*av.asc", "slstr_av"),
            (f"{day_dir.name}_slstr_*std_av.asc", "slstr_std"),
            f"surfmask_{day_dir.name}.asc",
            f"oi_{day_dir.name}.asc"
        ]
        missing_files = []
        for pattern in expected_patterns:
            if isinstance(pattern, tuple):
                glob_pattern, label = pattern
                pattern_to_use = glob_pattern
            else:
                pattern_to_use = pattern
            if '*' in pattern_to_use:
                if not list(day_dir.glob(pattern_to_use.split('/')[-1])):
                    missing_files.append(pattern_to_use)
            else:
                if not (day_dir / pattern_to_use).exists():
                    missing_files.append(pattern_to_use)
        nc_files = list(day_dir.glob(f"{day_dir.name}*.nc"))
        if not nc_files:
            missing_files.append(f"{day_dir.name}0000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-*.nc")
        if missing_files:
            print(f"Dans {day_dir.name}, fichiers manquants: {missing_files}")
            nb_jour_pb += 1
            days_with_problems.append(day_dir.name)
    if nb_jour_pb == 0:
        print(f"Tous les fichiers sont présents pour l'année {year}.")
    else:
        print(f"Il y a {nb_jour_pb} jours avec des fichiers manquants pour l'année {year}.")

    if return_days:
        return days_with_problems
    return nb_jour_pb

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Verify extracted SST files")
    parser.add_argument("year", type=int, help="Year to verify")
    parser.add_argument("--source-dir", type=str, help="Extracted source directory")
    parser.add_argument("--missing-days-file", type=str, help="Where to write missing days (default: /tmp/missing_days_{YEAR}.txt)")
    args = parser.parse_args()

    year = args.year
    days = verify_extraction(year, return_days=True, source_dir=args.source_dir)

    # Toujours sauvegarder la liste des jours manquants dans /tmp
    output_file = args.missing_days_file or f"/tmp/missing_days_{year}.txt"
    if days:
        with open(output_file, 'w') as f:
            for day in days:
                f.write(f"{day}\n")
        print(f"Liste des jours manquants sauvée: {output_file}")
    else:
        # Créer fichier vide si pas de problèmes
        open(output_file, 'w').close()

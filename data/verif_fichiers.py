from pathlib import Path
from tqdm import tqdm

def verify_extraction(year):
    """
    Petite fonction pour verifier que dans l'extraction du squash, on a bien tous les fichiers necessaires à la création de nos netcdf
    year : int, année à vérifier
    """
    directory = Path(f'/dmidata/users/malegu/data/squash_{year}_extract')
    if not directory.exists() or not directory.is_dir():
        print(f"Le dossier {directory} n'existe pas ou n'est pas un dossier.")
        return
    day_dirs = [d for d in directory.iterdir() if d.is_dir()]
    nb_jour_pb = 0
    if not day_dirs:
        print(f"Aucun dossier de jour trouvé dans {directory}.")
        return
    for day_dir in tqdm(sorted(day_dirs), desc=f"Vérification extraction {year}", unit="jour"):
        # le soucis technique est que chaque fichier change de nom car il contient le jour, mais a priori ce jour est le meme que le nom du fichier
        expected_files = [
            f"{day_dir.name}_aasti_ist_l2p_av.asc",
            f"{day_dir.name}_aasti_ist_l2p_std_av.asc",
            f"{day_dir.name}_avhrr_c3s_l3u_av.asc",
            f"{day_dir.name}_avhrr_c3s_l3u_std_av.asc",
            f"{day_dir.name}_pmw_cci_l2p_av.asc",
            f"{day_dir.name}_pmw_cci_l2p_std_av.asc",
            f"{day_dir.name}_slstr_c3s_l3u_av.asc",
            f"{day_dir.name}_slstr_c3s_l3u_std_av.asc",
            f"surfmask_{day_dir.name}.asc",
            f"oi_{day_dir.name}.asc"
        ]
        missing_files = [f for f in expected_files if not (day_dir / f).exists()]
        nc_files = list(day_dir.glob(f"{day_dir.name}0000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-*.nc"))
        if not nc_files:
            missing_files.append(f"{day_dir.name}0000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-*.nc")
        if missing_files:
            print(f"Dans {day_dir.name}, fichiers manquants: {missing_files}")
            nb_jour_pb += 1
    if nb_jour_pb == 0:
        print(f"Tous les fichiers sont présents pour l'année {year}.")
    else:
        print(f"Il y a {nb_jour_pb} jours avec des fichiers manquants pour l'année {year}.")
    return nb_jour_pb

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 verif_fichiers.py YEAR")
        sys.exit(1)
    year = int(sys.argv[1])
    verify_extraction(year)
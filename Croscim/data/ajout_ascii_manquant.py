import numpy as np
from pathlib import Path

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
    
def ajout_multiples_ascii(year, days):
    """
    A partir de la liste des jours manquants, on ajoute les fichiers ASCII manquants.
    year : int, année à traiter
    days : liste de la forme [YYYYMMDD12, ...]
    """
    directory = Path(f'/dmidata/projects/4dvarnet/squash_{year}_extract')
    reference_file = Path('/dmidata/projects/4dvarnet/reference_ascii.asc')
    # print(f"Référence utilisée: {reference_file}")

    nb_created = 0
    for day in days:
        day_dir = directory / day
        if not day_dir.exists() or not day_dir.is_dir():
            print(f"Le dossier {day_dir} n'existe pas ou n'est pas un dossier.")
            continue

        expected_files = [f"{day}_aasti_ist_l2p_av.asc", f"{day}_aasti_ist_l2p_std_av.asc",
                          f"{day}_avhrr_c3s_l3u_av.asc", f"{day}_avhrr_c3s_l3u_std_av.asc",
                          f"{day}_pmw_cci_l2p_av.asc", f"{day}_pmw_cci_l2p_std_av.asc",
                          f"{day}_slstr_c3s_l3u_av.asc", f"{day}_slstr_c3s_l3u_std_av.asc"]

        for expected in expected_files:
            target = day_dir / expected
            if not target.exists():
                ajout_ascii_manquant(target, reference_file)
                nb_created += 1

    print(f"Tous les fichiers manquants ont été ajoutés ({nb_created} fichiers).")
    return

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 ajout_ascii_manquant.py YEAR")
        sys.exit(1)
    year = int(sys.argv[1])
    file_path = f"/tmp/missing_days_{year}.txt"
    with open(file_path, 'r') as f:
        days = [line.strip() for line in f if line.strip()]
    if not days:
        print("Aucun jour à corriger")
        sys.exit(0)
    print(f"Correction de {len(days)} jours")
    ajout_multiples_ascii(year, days)


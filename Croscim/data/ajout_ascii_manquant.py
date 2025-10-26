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
    
def ajout_multiples_ascii(days):
    """
    A partir de la liste des jours manquants, on ajoute les fichiers ASCII manquants.
    days : liste de la forme [YYYYMMDD12, ...]
    """
    directory = Path('/dmidata/users/malegu/data/squash_2024_extract')
    for day in days:
        day_dir = directory / day
        if not day_dir.exists() or not day_dir.is_dir():
            print(f"Le dossier {day_dir} n'existe pas ou n'est pas un dossier.")
            continue
        fichiers_ajout = [f"{day}_pmw_cci_l2p_av.asc",
                          f"{day}_pmw_cci_l2p_std_av.asc"]
        ajout_ascii_manquant(f"{day_dir}/{fichiers_ajout[0]}", f"{day_dir}/{day}_aasti_ist_l2p_av.asc")
        ajout_ascii_manquant(f"{day_dir}/{fichiers_ajout[1]}", f"{day_dir}/{day}_aasti_ist_l2p_std_av.asc")
    print("Tous les fichiers manquants ont été ajoutés.")
    return

if __name__ == '__main__':
    days = [ "2024020212", "2024021212", "2024021312", "2024021412"]
    ajout_multiples_ascii(days)
    

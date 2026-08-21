# Présentation de Croscim

Support pour présenter le projet à un superviseur et à un nouveau stagiaire.
L'objectif n'est pas de parcourir chaque fichier, mais de leur donner une carte
mentale suffisante pour comprendre, lancer et modifier le projet.

Pour cette présentation, utiliser ce document et les fichiers sous `docs/`
plutôt que `todo.md` : la TODO conserve certains états intermédiaires,
notamment l'ancienne hypothèse que 2014–2016 seraient exploitables.

## Résumé En 30 Secondes

> Croscim reconstruit la température de surface de la mer globale à partir
> d'observations satellites incomplètes. Le modèle travaille sur trois
> résolutions imbriquées. Il reconstruit d'abord un contexte global grossier en
> x10, puis apprend des résidus en x3 et x1. Chaque résolution utilise un solver
> 4D-VarNet déroulé : un prior neuronal définit les états plausibles et un
> ConvLSTM apprend comment mettre à jour l'état à partir du gradient du coût
> variationnel. Le code est configuré avec Hydra, entraîné avec Lightning et
> exécuté sur Gefion avec SLURM et 8 H100.

## Déroulé Conseillé Pour Une Réunion De 20 Minutes

1. **2 min — Problème scientifique**
   - SST globale, satellites incomplets, nuages et trous d'observation.
   - Objectif : reconstruire un champ cohérent dans l'espace et dans le temps.

2. **3 min — Données et multi-résolution**
   - Montrer les fichiers quotidiens x1/x3/x10.
   - Expliquer les patches géographiquement imbriqués et les fenêtres
     temporelles 15/9/5 jours.

3. **6 min — Architecture**
   - Expliquer le solver 4D-VarNet, le prior et le ConvLSTM.
   - Montrer la cascade x10 -> x3 -> x1 et les résidus.

4. **4 min — Organisation du code**
   - Montrer `config/`, `contrib/SST/`, `src/`, `scripts/` et `data/`.
   - Suivre le chemin `main.py -> Hydra -> Trainer -> DataModule -> modèle`.

5. **3 min — Comment lancer**
   - Baseline, test checkpoint, TensorBoard et preprocessing Gefion.

6. **2 min — État actuel et reprise**
   - Baseline fonctionnelle.
   - ResUNet intégré mais problème mémoire encore ouvert.
   - Prochaines expériences : prior ResUNet, prior Swin, puis grad mod moderne.

## Le Problème Scientifique

Les observations SST proviennent de plusieurs instruments :

- `SLSTR` : précis mais incomplet et absent aux pôles ;
- `AASTI` : utilisé notamment dans les régions polaires ;
- `AVHRR` : couverture globale importante ;
- `PMW` : très couvrant mais plus lisse ;
- `sea_ice_fraction` : covariable décrivant la glace de mer.

La cible `tgt_sst` est une fusion :

```text
si sea_ice_fraction >= 0.70 : AASTI
sinon                         : SLSTR
```

L'apprentissage est auto-supervisé. Une partie des observations disponibles
est artificiellement masquée :

```text
tgt_sst_full : cible complète avant masquage artificiel
tgt_sst      : observations visibles données au solver
inpaint_mask : pixels retirés artificiellement
```

Le modèle doit reconstruire les pixels cachés tout en restant fidèle aux
observations visibles.

## Parcours Des Données

### 1. Archives brutes

Sur Gefion, les archives SQFS durables sont dans :

```text
/dcai/projects/cu_0026/data_sst/sqfs
```

### 2. Preprocessing annuel

```text
SQFS
  -> extraction
  -> vérification des fichiers quotidiens
  -> correction des ASCII manquants
  -> conversion x1 Zarr
  -> calcul des Zarr x3 et x10
```

Commande :

```bash
sbatch data/process_year_gefion.slurm 2022
```

Les sorties suivent ce format :

```text
/dcai/projects/cu_0026/data_sst/data_YYYY/
  YYYYMMDD12_x1.zarr
  YYYYMMDD12_x3.zarr
  YYYYMMDD12_x10.zarr
```

La plage actuellement retenue comme complète et scientifiquement exploitable
est **2017–2024**, soit 2922 jours. Les années 2014–2015 n'ont pas de SLSTR et
2016 a une couverture SLSTR incomplète.

### 3. Dataset et DataModule

Les fichiers importants sont :

- `contrib/SST/data.py` : chargement d'une résolution, normalisation, fusion de
  cible, masquage SSL et filtrage des patches ;
- `contrib/SST/data_multires.py` : construction des trois patches imbriqués et
  DataLoaders Lightning ;
- `contrib/SST/load_data.py` : définition des groupes satellites et fonctions
  de coarsening.

Un échantillon retourné au modèle ressemble conceptuellement à :

```text
batch
├── patch_x10 : contexte global grossier, 256x256
├── patch_x3  : zone intermédiaire contenue dans x10, 256x256
└── patch_x1  : zone fine contenue dans x3, 256x256
```

L'extraction commence par x1. Le code cherche ensuite un patch x3 qui contient
x1, puis un patch x10 qui contient x3. Les trois patches ont donc la même taille
en pixels, mais couvrent des surfaces géographiques différentes.

### 4. Normalisation

Toutes les températures moyennes utilisent une échelle commune `sst_common`.
C'est indispensable car x3 et x1 manipulent des anomalies soustraites à la
prédiction coarse. Les incertitudes `*_std` conservent leurs propres
statistiques.

Pour 2017–2024 :

```yaml
sst_common:
  mean: 7.78095617993726
  std: 20.58217902545633
```

Les statistiques sont générées par `contrib/SST/compute_statistics.py`, jamais
inventées manuellement.

## Architecture Du Modèle

## Vue Globale

```text
observations x10
    |
    v
solver x10 -> SST x10
    |
    | interpolation sur la grille x3
    v
coarse x3 + solver x3 prédisant un résidu -> SST x3
    |
    | interpolation sur la grille x1
    v
coarse x1 + solver x1 prédisant un résidu -> SST x1
```

Les fenêtres temporelles sont :

```text
x10 : 15 jours
x3  :  9 jours centraux
x1  :  5 jours centraux
```

Le passage x10 -> x3 -> x1 est une cascade résiduelle :

```text
target_residual = target_fine - prediction_coarse_interpolée
prediction_fine = prediction_coarse_interpolée + résidu_prédit
```

Ce point est important : x3 et x1 ne doivent pas réapprendre toute la
température absolue.

## Ce Que Fait Un Solver

Chaque résolution possède son propre `GradSolver` :

```text
state_0 = fusion SST masquée

pour chaque étape :
    coût = coût_prior + coût_observation
    gradient = d(coût) / d(state)
    correction = ConvLSTM(gradient)
    state = state - correction
```

Il existe deux niveaux d'optimisation :

1. **Dans le forward** : le solver optimise le champ SST `state`.
2. **Pendant le backward Lightning** : Adam optimise les poids du prior et du
   ConvLSTM à travers toutes les étapes déroulées.

Ce n'est pas un modèle de diffusion. Il ne part pas d'un bruit aléatoire : il
part des observations fusionnées et apprend une procédure d'optimisation.

## Prior, Observation Cost Et Grad Mod

### Observation cost

`BaseObsCost` contraint le state à respecter les pixels réellement observés.

### Prior

Le prior reçoit le state courant et les covariables :

```text
Phi([state, covariables]) -> reconstruction plausible
coût_prior = MSE(state, Phi(...))
```

Il ne fournit donc pas directement la sortie finale. Il définit une énergie qui
guide toutes les mises à jour du solver.

Deux priors existent :

- `BilinReconstructorPriorCost` : baseline convolutionnelle légère et validée ;
- `ResUNetPriorCost` : expérience plus expressive avec encodeur, décodeur,
  blocs résiduels et skip connections.

### Gradient modulator

`ConvLstmGradModel` reçoit la carte de gradient et garde une mémoire entre les
étapes. Il apprend comment transformer ce gradient en correction du state.

Les composants sont séparés ici :

```text
contrib/SST/model_components/
├── priors/
│   ├── bilinear.py
│   └── resunet.py
└── grad_mods/
    └── convlstm.py
```

## Où Se Trouve Quoi

```text
main.py
    Point d'entrée Hydra. Appelle les entrypoints de la config.

config/main.yaml
    Config Hydra racine.

config/xp/SST/
    Expériences complètes : trainer, données, modèle, chemins et losses.

src/train.py
    Entrée d'entraînement générique : trainer.fit().

src/test.py
    Entrée de test générique : trainer.test().

contrib/SST/data.py
    Dataset SST mono-résolution et préparation scientifique.

contrib/SST/data_multires.py
    Dataset multi-résolution et DataModule Lightning.

contrib/SST/models.py
    LightningModule SST, cascade, losses, validation, test et agrégation.

contrib/SST/solver.py
    Boucle d'optimisation variationnelle déroulée.

contrib/SST/model_components/
    Priors et gradient modulators interchangeables.

contrib/SST/visualization.py
    Figures de validation et de test.

data/
    Pipeline de preprocessing.

scripts/local/ et scripts/gefion/
    Commandes reproductibles par environnement.

docs/
    Documentation maintenue.

archive/ et notes/
    Historique et contexte, pas source de vérité.
```

## Chemin D'Exécution

```text
commande shell
  -> python main.py xp=SST/<config>
  -> Hydra lit le YAML
  -> Hydra instancie Trainer, DataModule, Lit4dVarNet_SST et les solvers
  -> src.train.base_training()
  -> trainer.fit()
  -> DataModule.setup()
  -> DataLoader retourne patch_x10/x3/x1
  -> Lit4dVarNet_SST.multistep()
  -> GradSolver de chaque résolution
  -> losses et backward Lightning
```

Pour comprendre une run, il faut donc lire en premier sa config YAML. Elle
indique exactement les classes instanciées, les dimensions, les dates, les
ressources et les chemins.

## Comment Ajouter Un Nouveau Modèle

Pour changer uniquement le prior :

1. créer une classe dans `contrib/SST/model_components/priors/` ;
2. conserver l'interface :

```python
self.dim_out
forward_reconstructor(x_obs)
forward(state, batch)
```

3. créer une config expérimentale copiée de la baseline ;
4. remplacer seulement `_target_` et les paramètres du prior ;
5. ne pas modifier simultanément le ConvLSTM, les losses et les données ;
6. comparer avec la même validation et le même test.

Exemple Hydra :

```yaml
prior_cost:
  _target_: contrib.SST.model_components.priors.resunet.ResUNetPriorCost
  dim_in: 124
  dim_hidden: 48
  dim_out: 15
```

Cette séparation permet d'attribuer une amélioration ou une régression au bon
composant.

## Commandes Essentielles

## Local

```bash
conda activate croscim
export PYTHONPATH=$PWD:$PYTHONPATH
./scripts/local/run_train_lite.sh 0
```

Run local principale :

```bash
./scripts/local/run.sh 0
```

## Gefion

Après connexion :

```bash
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
source scripts/gefion/env.sh
```

Les scripts SLURM chargent eux-mêmes cet environnement. Il n'est pas nécessaire
de l'activer avant `sbatch`.

Baseline DDP validée :

```bash
sbatch scripts/gefion/train_gefion.sh
```

Expérience ResUNet :

```bash
sbatch scripts/gefion/train_gefion_resunet.sh
```

Suivi :

```bash
squeue -u "$USER"
tail -f logs/slurm_<job_id>.out
tail -f logs/train_<job_id>.log
scancel <job_id>
```

Test d'un checkpoint :

```bash
sbatch scripts/gefion/test_checkpoint_gefion.slurm \
  /dcai/projects/cu_0026/guimae/croscim/checkpoints/model.ckpt
```

## TensorBoard Gefion

Sur le même login node que le tunnel MobaXterm :

```bash
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
source scripts/gefion/env.sh
tensorboard --logdir /dcai/projects/cu_0026/guimae/croscim/results \
  --host 127.0.0.1 --port 6123
```

Puis ouvrir :

```text
http://127.0.0.1:6123
```

## Résultats

```text
/dcai/projects/cu_0026/guimae/croscim/results
    TensorBoard

/dcai/projects/cu_0026/guimae/croscim/checkpoints
    checkpoints Lightning

/dcai/projects/cu_0026/guimae/croscim/outputs
    figures de validation, NetCDF et analyses de test

Croscim/logs
    sorties SLURM et logs texte
```

## État Actuel À Présenter Honnêtement

### Fonctionnel

- preprocessing annuel SQFS -> x1/x3/x10 ;
- données 2017–2024 préparées et statistiques recalculées ;
- entraînement DDP baseline sur 8 H100 ;
- cycle d'entraînement x10 -> x3 -> x1 ;
- targets résiduelles x3/x1 et normalisation commune corrigées ;
- sélection de validation déterministe et compatible DDP ;
- test checkpoint, reconstruction globale et visualisations ;
- couverture complète des bords et artefacts côtiers corrigés ;
- TensorBoard accessible via Citrix/MobaXterm.

### Expérience En Cours

Le prior ResUNet est intégré, mais sa run Gefion n'est pas encore validée.
L'OOM actuel ne vient pas de la taille des fichiers Zarr. Le solver exécute le
prior à chaque étape et utilise :

```python
torch.autograd.grad(..., create_graph=True)
```

Le graphe du prior et l'état récurrent du ConvLSTM peuvent donc conserver de
nombreuses activations à travers les 10 ou 20 étapes. L'OOM le plus récent
arrive pendant le sanity check de validation, où ces graphes d'ordre supérieur
ne devraient pas être conservés.

Prochaine correction envisagée :

1. validation/test avec `create_graph=False` ;
2. détachement de l'état ConvLSTM entre étapes d'évaluation ;
3. mesure mémoire par résolution et par étape ;
4. activation checkpointing du ResUNet pendant le train si nécessaire.

Il faut corriger ce mécanisme avant de conclure que le ResUNet est simplement
trop gros ou de réduire systématiquement le batch à 1.

## Points De Vigilance Pour Le Repreneur

- Toujours vérifier le YAML réellement lancé avant une longue run.
- Ne jamais changer silencieusement l'ordre des canaux 124/76/44.
- Utiliser `lat_geo`/`lon_geo` pour l'interpolation, pas les coordonnées
  normalisées `lat`/`lon`.
- Garder la même normalisation `sst_common` pour les températures utilisées
  dans les soustractions résiduelles.
- Ne pas utiliser 2014–2016 dans la config actuelle sans revoir la cible.
- En DDP, ne pas placer `sync_dist=True` dans un bloc exécuté uniquement par le
  rank 0.
- Garder `DASK_SCHEDULER=synchronous` et `persistent_workers: false` sur
  Gefion.
- Ne pas juger x3/x1 avec les checkpoints antérieurs au correctif résiduel.
- `archive/` et `notes/` ne sont pas la documentation opérationnelle.

## Proposition De Reprise Pour Le Nouveau Stagiaire

1. Lire `README.md`, `docs/architecture.md`, `docs/data.md` et
   `docs/workflows.md`.
2. Charger `SST/multires_lite` et suivre l'instanciation Hydra.
3. Dessiner les shapes d'un batch x10/x3/x1 jusqu'au solver.
4. Lancer un smoke test local ou une évaluation de checkpoint existant.
5. Corriger et profiler la mémoire du ResUNet sans changer la science.
6. Comparer ResUNet et prior bilinéaire avec protocole identique.
7. Implémenter ensuite un `SwinPriorCost`.
8. Ne changer le gradient modulator qu'après avoir isolé l'effet du prior.

## Questions Utiles À Poser Pendant La Réunion

- Quelle métrique scientifique doit décider qu'un nouveau prior est meilleur ?
- Quel protocole commun utiliser pour baseline, ResUNet et Swin ?
- Faut-il prioriser la qualité globale, les zones côtières, les pôles ou les
  grands trous nuageux ?
- Quelle baseline OI doit être utilisée et sur quelles dates ?
- Le nouveau stagiaire doit-il travailler d'abord sur le prior, le profiling
  mémoire, les données ou l'évaluation ?
- Quelle partie du projet doit devenir publiable ou réutilisable au-delà de
  cette expérience ?

## Fichiers À Montrer À L'Écran

Ordre recommandé :

1. `README.md`
2. `config/xp/SST/multires_gefion.yaml`
3. `contrib/SST/data_multires.py`
4. `contrib/SST/models.py`, fonction `multistep`
5. `contrib/SST/solver.py`
6. `contrib/SST/model_components/priors/bilinear.py`
7. `contrib/SST/model_components/priors/resunet.py`
8. `scripts/gefion/train_gefion.sh`
9. `docs/current-state.md`

Ne pas commencer par `models.py` en entier : le fichier est volumineux. Montrer
d'abord le schéma, puis uniquement les fonctions correspondant au chemin
d'exécution.

## Documentation À Donner Au Repreneur

- `README.md` : point d'entrée ;
- `AGENTS.md` : conventions, chemins et pièges ;
- `docs/architecture.md` : architecture ;
- `docs/data.md` : contrat des données ;
- `docs/configuration.md` : configs Hydra ;
- `docs/workflows.md` : commandes ;
- `docs/current-state.md` : état et travail ouvert ;
- `docs/tests-and-tools.md` : tests et outils disponibles.

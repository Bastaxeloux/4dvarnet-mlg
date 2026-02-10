# Guide d'utilisation Cluster Gefion (H100)

- **8× H100 GPUs** par node (80GB chacun)
- **DDP (Distributed Data Parallel)** pour training multi-GPU
- **Float32** par défaut (ou `bf16-mixed` pour accélérer)

## Lancement du training

### Multi-GPU (8 GPUs DDP) - RECOMMANDÉ
```bash
conda activate 4denv
./run_gefion.sh
```

**Specs:**
- 8 GPUs × batch_size=4 × accumulate=5 = **160 effective batch size**
- Gradient synchronisé entre GPUs automatiquement
- Throughput: ~8× plus rapide qu'un seul GPU

### Single-GPU (test/debug)
```bash
conda activate 4denv
./run_gefion_single.sh 0  # GPU 0
```

## Monitoring

### Logs
```bash
# Multi-GPU
tail -f run_gefion.log

# Single-GPU
tail -f run_gefion_single.log
```

### GPU usage
```bash
watch -n 1 nvidia-smi
```

### TensorBoard
```bash
./tensorboard.sh
# Ouvrir http://localhost:6006 (ou port forwarding si remote)
```

## Configuration multi-GPU

### Fichier: `config/xp/SST/multires_gefion.yaml`

**Paramètres clés:**
```yaml
trainer:
  devices: 8              # Nombre de GPUs
  strategy: ddp           # Distributed Data Parallel
  accumulate_grad_batches: 5   # 8×4×5 = 160 effective batch
  
dl_kw:
  batch_size: 4           # Par GPU
  num_workers: 8          # Par GPU (ajuster selon CPU)
```

### Calcul du batch effectif
```
Effective batch = devices × batch_size × accumulate_grad_batches
                = 8 × 4 × 5 = 160 samples par gradient update
```

## Optimisation des performances

### 1. Augmenter batch_size si mémoire disponible
H100 = 80GB, vous pouvez essayer:
```yaml
dl_kw:
  batch_size: 8  # Au lieu de 4
  # → 8×8×5 = 320 effective batch
```

### 2. Utiliser bfloat16 pour accélérer
```yaml
trainer:
  precision: "bf16-mixed"  # Au lieu de 32
  # → 2× plus rapide, même range que float32
```

### 3. Ajuster num_workers selon CPU
```yaml
dl_kw:
  num_workers: 16  # Si beaucoup de CPU disponibles
```

## Différences vs A40

| Paramètre | A40 (dev) | Gefion H100 (cluster) |
|-----------|-----------|----------------|
| GPUs | 1 | 8 (DDP) |
| VRAM | 40GB | 80GB |
| Batch size | 2 | 4 (ou 8) |
| Accumulate | 20 | 5 |
| Effective batch | 40 | 160 (ou 320) |
| Precision | float32 | float32 ou bf16 |
| Throughput | ~2 samp/s | ~16+ samp/s |

## Troubleshooting

### OOM malgré 80GB
- Réduire `batch_size` de 4 → 2
- Activer gradient checkpointing (si implémenté)

### Slow dataloading
- Augmenter `num_workers` (8 → 16)
- Vérifier que `persistent_workers=true`

### Logs vides ou bloqués
- Vérifier `run_gefion.log` pour erreurs
- Tester d'abord avec `run_gefion_single.sh` (1 GPU)

### DDP hang au démarrage
- Vérifier que tous les GPUs sont accessibles: `nvidia-smi`
- Problème réseau entre GPUs: vérifier NCCL_DEBUG=INFO

## Commandes utiles

```bash
# Tuer le training
kill $(cat process_gefion.pid)

# Vérifier les processus Python
ps aux | grep "python main.py"

# Nettoyer la mémoire GPU
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9

# Compresser les logs
gzip run_gefion.log
```

## Checkpoints

Sauvegardés dans: `/dmidata/projects/4dvarnet/checkpoints_sst_multires/`

Format: `epoch=XXX-val/loss=Y.YYYYY.ckpt`

Le meilleur checkpoint (top-3) et le dernier sont conservés.

## Test d'un checkpoint

```bash
./run_test_checkpoint_gefion.sh /path/to/checkpoint.ckpt [GPU_ID]
```

Exemple:
```bash
./run_test_checkpoint_gefion.sh /dmidata/projects/4dvarnet/checkpoints_sst_multires/epoch=050-val/loss=0.00123.ckpt 0
```

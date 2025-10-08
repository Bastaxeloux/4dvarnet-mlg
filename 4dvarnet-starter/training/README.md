# 🎯 Guide d'Entraînement 4dVarNet

## 🚀 Démarrage rapide

```bash
cd /home/malegu/4dvarnet-starter
python training_system.py
```

## 📚 Structure du système

### Niveau 1 : Complétion de code basique
- **Exercice 1.1** : Méthode `__init__` - Apprendre l'initialisation
- **Exercice 1.2** : Propriété `norm_stats` - Gestion des statistiques
- **Exercice 1.3** : Méthode `forward` - Forward pass simple

### Niveau 2 : Méthodes de calcul
- **Exercice 2.1** : `weighted_mse` - Loss function complexe
- **Exercice 2.2** : `base_step` - Forward + loss + logging
- **Exercice 2.3** : Training steps - Intégration PyTorch Lightning

### Niveau 3 : Test et reconstruction (à venir)
- `test_step` - Collection des données de test
- `on_test_epoch_end` - Reconstruction des patches
- Gestion des métriques

### Niveau 4 : Optimisations avancées (à venir)
- Optimisations de performance
- Fonctionnalités avancées
- Debugging et monitoring

## 💡 Conseils pour réussir

1. **Lisez attentivement** les consignes de chaque exercice
2. **Utilisez les hints** fournis dans les templates
3. **N'hésitez pas à relancer** les vérifications
4. **Progressez à votre rythme** - la progression est sauvegardée

## 🔧 Fonctionnalités

- ✅ **Vérification automatique** des exercices
- 💾 **Sauvegarde de progression** 
- 🔄 **Redémarrage de niveaux**
- 🎓 **Mode libre** pour accéder à tous les niveaux
- 📊 **Suivi de progression détaillé**

## 📁 Structure des fichiers

```
training/
├── progress.txt          # Votre progression
├── exercise_1_1.py      # Templates d'exercices
├── exercise_1_2.py
├── reference.py         # Solutions de référence
└── README.md           # Ce guide
```

## 🆘 Aide

Si vous rencontrez des problèmes :
1. Vérifiez la syntaxe Python
2. Relisez les consignes
3. Consultez le code original dans `src/models.py`
4. Utilisez le mode libre pour revoir les exercices précédents

Bon entraînement ! 🎓

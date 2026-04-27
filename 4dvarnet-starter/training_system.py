#!/usr/bin/env python3

"""
SYSTÈME D'ENTRAÎNEMENT PROGRESSIF - 4dVarNet
=============================================

Ce système vous permet d'apprendre progressivement à coder la classe Lit4dVarNet.
Il est organisé en niveaux de difficulté croissante avec vérification automatique.

Structure:
- Niveau 1: Complétion de code avec trous
- Niveau 2: Écriture de méthodes simples
- Niveau 3: Logique complexe et intégrations
- Niveau 4: Optimisations et fonctionnalités avancées

Instructions:
1. Lancez: python training_system.py
2. Suivez les instructions pour chaque exercice
3. Le système vérifie automatiquement vos réponses
4. Progressez à votre rythme !
"""

import os
import sys
import inspect
import importlib.util
from pathlib import Path
import torch
import pytorch_lightning as pl

class TrainingSystem:
    def __init__(self):
        self.current_level = 1
        self.current_exercise = 1
        self.progress_file = Path("training/progress.txt")
        self.load_progress()
        
    def load_progress(self):
        """Charger la progression depuis le fichier."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    self.current_level = int(lines[0].strip())
                    self.current_exercise = int(lines[1].strip())
    
    def save_progress(self):
        """Sauvegarder la progression."""
        self.progress_file.parent.mkdir(exist_ok=True)
        with open(self.progress_file, 'w') as f:
            f.write(f"{self.current_level}\n{self.current_exercise}\n")
    
    def show_menu(self):
        """Afficher le menu principal."""
        print("\n" + "="*60)
        print("🎯 SYSTÈME D'ENTRAÎNEMENT 4dVarNet")
        print("="*60)
        print(f"📍 Position actuelle: Niveau {self.current_level}, Exercice {self.current_exercise}")
        print("\nOptions disponibles:")
        print("1. 🚀 Continuer l'entraînement")
        print("2. 📊 Voir la progression complète")
        print("3. 🔄 Recommencer un niveau")
        print("4. 🎓 Mode libre (accès à tous les niveaux)")
        print("5. ❌ Quitter")
        print("-"*60)
        
    def run(self):
        """Lancer le système d'entraînement."""
        while True:
            self.show_menu()
            choice = input("Votre choix (1-5): ").strip()
            
            if choice == '1':
                self.continue_training()
            elif choice == '2':
                self.show_progress()
            elif choice == '3':
                self.restart_level()
            elif choice == '4':
                self.free_mode()
            elif choice == '5':
                print("👋 À bientôt !")
                break
            else:
                print("❌ Choix invalide, réessayez.")
    
    def continue_training(self):
        """Continuer l'entraînement au niveau actuel."""
        level_method = getattr(self, f'level_{self.current_level}', None)
        if level_method:
            level_method()
        else:
            print(f"🎉 Félicitations ! Vous avez terminé tous les niveaux disponibles !")
    
    def show_progress(self):
        """Afficher la progression détaillée."""
        print("\n" + "="*50)
        print("📊 PROGRESSION DÉTAILLÉE")
        print("="*50)
        
        levels = {
            1: "Complétion de code basique",
            2: "Écriture de méthodes simples", 
            3: "Logique complexe et intégrations",
            4: "Optimisations avancées"
        }
        
        for level, description in levels.items():
            status = "✅" if level < self.current_level else "🔄" if level == self.current_level else "⏳"
            print(f"{status} Niveau {level}: {description}")
            
        print(f"\n📍 Position actuelle: Niveau {self.current_level}, Exercice {self.current_exercise}")
    
    def restart_level(self):
        """Recommencer un niveau."""
        print("\n🔄 Recommencer un niveau:")
        level = input("Quel niveau voulez-vous recommencer ? (1-4): ").strip()
        try:
            level = int(level)
            if 1 <= level <= 4:
                self.current_level = level
                self.current_exercise = 1
                self.save_progress()
                print(f"✅ Niveau {level} redémarré !")
            else:
                print("❌ Niveau invalide.")
        except ValueError:
            print("❌ Veuillez entrer un nombre.")
    
    def free_mode(self):
        """Mode libre pour accéder à n'importe quel niveau."""
        print("\n🎓 MODE LIBRE")
        print("Niveaux disponibles:")
        print("1. Complétion de code basique")
        print("2. Écriture de méthodes simples")
        print("3. Logique complexe et intégrations") 
        print("4. Optimisations avancées")
        
        level = input("Choisissez un niveau (1-4): ").strip()
        try:
            level = int(level)
            if 1 <= level <= 4:
                level_method = getattr(self, f'level_{level}')
                level_method()
            else:
                print("❌ Niveau invalide.")
        except (ValueError, AttributeError):
            print("❌ Niveau invalide.")
    
    def level_1(self):
        """Niveau 1: Complétion de code basique."""
        print("\n" + "="*50)
        print("📚 NIVEAU 1: COMPLÉTION DE CODE BASIQUE")
        print("="*50)
        
        exercises = [
            self.exercise_1_1_init,
            self.exercise_1_2_properties,
            self.exercise_1_3_forward,
        ]
        
        if self.current_exercise <= len(exercises):
            exercises[self.current_exercise - 1]()
        else:
            print("🎉 Niveau 1 terminé ! Passage au niveau 2...")
            self.current_level = 2
            self.current_exercise = 1
            self.save_progress()
    
    def exercise_1_1_init(self):
        """Exercice 1.1: Méthode __init__"""
        print("\n🎯 EXERCICE 1.1: Méthode __init__")
        print("-"*40)
        print("Objectif: Compléter la méthode __init__ de Lit4dVarNet")
        print("\nConsignes:")
        print("- Hériter de pl.LightningModule")
        print("- Initialiser les attributs: solver, rec_weight, norm_stats, etc.")
        print("- Gérer le cas où rec_weight est un dict ou un array")
        
        # Créer le template
        template = '''import pytorch_lightning as pl
import torch

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True):
        # TODO: Appeler le constructeur parent
        _______________()
        
        # TODO: Assigner solver
        self.solver = _______
        
        # TODO: Gérer rec_weight (dict ou array)
        if not isinstance(rec_weight, dict):
            # TODO: Enregistrer comme buffer pour un array
            self.________________('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        else:
            self.rec_weight = {}
            for key, weight_array in rec_weight.items():
                buffer_name = f"_rec_weight_{key}"
                weight_tensor = torch.from_numpy(weight_array).to("cuda")
                # TODO: Enregistrer le buffer
                self.________________(buffer_name, weight_tensor, persistent=persist_rw)
                self.rec_weight[key] = getattr(self, buffer_name)
        
        # TODO: Initialiser les autres attributs
        self.test_data = ____
        self._norm_stats = __________
        self.opt_fn = ______
        self.metrics = test_metrics or ___
        self.pre_metric_fn = pre_metric_fn or (lambda x: _)
'''
        
        self.create_exercise_file("exercise_1_1.py", template)
        
        print(f"\n📝 Template créé dans: training/exercise_1_1.py")
        print("💡 Conseils:")
        print("- super().__init__() pour appeler le parent")
        print("- register_buffer() pour enregistrer des tenseurs")
        print("- Utilisez None, {}, etc. pour les valeurs par défaut")
        
        input("\n⏸️  Complétez le code puis appuyez sur Entrée pour vérifier...")
        
        if self.check_exercise_1_1():
            print("✅ Excellent ! Exercice 1.1 réussi !")
            self.current_exercise = 2
            self.save_progress()
        else:
            print("❌ Pas tout à fait. Réessayez !")
    
    def check_exercise_1_1(self):
        """Vérifier l'exercice 1.1."""
        try:
            # Charger le fichier de l'utilisateur
            spec = importlib.util.spec_from_file_location("exercise", "training/exercise_1_1.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Tester la classe
            solver = None  # Mock
            rec_weight = torch.randn(10, 10).numpy()
            opt_fn = lambda x: None
            
            # Test 1: Instantiation basique
            model = module.Lit4dVarNet(solver, rec_weight, opt_fn)
            
            # Vérifications
            checks = [
                hasattr(model, 'solver'),
                hasattr(model, 'rec_weight'),
                hasattr(model, 'test_data'),
                hasattr(model, '_norm_stats'),
                hasattr(model, 'opt_fn'),
                hasattr(model, 'metrics'),
                hasattr(model, 'pre_metric_fn'),
                isinstance(model, pl.LightningModule),
                model.test_data is None,
                isinstance(model.metrics, dict)
            ]
            
            passed = sum(checks)
            total = len(checks)
            
            print(f"\n📊 Résultats: {passed}/{total} vérifications réussies")
            
            if passed == total:
                return True
            else:
                print("❌ Vérifications échouées:")
                check_names = [
                    "Attribut solver",
                    "Attribut rec_weight", 
                    "Attribut test_data",
                    "Attribut _norm_stats",
                    "Attribut opt_fn",
                    "Attribut metrics",
                    "Attribut pre_metric_fn",
                    "Héritage de LightningModule",
                    "test_data initialisé à None",
                    "metrics est un dict"
                ]
                for i, (check, name) in enumerate(zip(checks, check_names)):
                    if not check:
                        print(f"  - {name}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de la vérification: {e}")
            return False
    
    def exercise_1_2_properties(self):
        """Exercice 1.2: Propriétés et méthodes simples"""
        print("\n🎯 EXERCICE 1.2: Propriétés norm_stats")
        print("-"*40)
        print("Objectif: Implémenter la propriété norm_stats")
        
        template = '''# Ajoutez cette méthode à votre classe Lit4dVarNet

@property
def norm_stats(self):
    """Retourner les statistiques de normalisation."""
    # TODO: Si _norm_stats n'est pas None, le retourner
    if self._norm_stats is not None:
        return _______________
    # TODO: Sinon, si trainer.datamodule existe, utiliser sa méthode norm_stats()
    elif self.trainer.datamodule is not None:
        return self.trainer.datamodule.__________()
    # TODO: Sinon, retourner les valeurs par défaut (0., 1.)
    return (__, ___)
'''
        
        self.create_exercise_file("exercise_1_2.py", template)
        print(f"\n📝 Template créé dans: training/exercise_1_2.py")
        print("💡 Cette propriété gère les statistiques de normalisation avec fallback")
        
        input("\n⏸️  Complétez le code puis appuyez sur Entrée pour continuer...")
        self.current_exercise = 3
        self.save_progress()
    
    def exercise_1_3_forward(self):
        """Exercice 1.3: Méthode forward"""
        print("\n🎯 EXERCICE 1.3: Méthode forward")
        print("-"*40)
        print("Objectif: Implémenter la méthode forward")
        
        template = '''# Méthode forward - la plus simple !

def forward(self, batch):
    """Forward pass: déléguer au solver."""
    # TODO: Retourner le résultat du solver avec le batch
    return self._______(_____)
'''
        
        self.create_exercise_file("exercise_1_3.py", template)
        print(f"\n📝 Template créé dans: training/exercise_1_3.py")
        print("💡 La méthode forward délègue simplement au solver")
        
        input("\n⏸️  Complétez le code puis appuyez sur Entrée pour terminer le niveau 1...")
        
        print("🎉 Niveau 1 terminé ! Vous maîtrisez les bases !")
        self.current_level = 2
        self.current_exercise = 1
        self.save_progress()
    
    def level_2(self):
        """Niveau 2: Méthodes plus complexes"""
        print("\n" + "="*50)
        print("🔥 NIVEAU 2: MÉTHODES COMPLEXES")
        print("="*50)
        print("Nous allons maintenant implémenter les méthodes de calcul de loss")
        print("et les steps d'entraînement/validation.")
        
        exercises = [
            self.exercise_2_1_weighted_mse,
            self.exercise_2_2_base_step,
            self.exercise_2_3_training_steps,
        ]
        
        if self.current_exercise <= len(exercises):
            exercises[self.current_exercise - 1]()
        else:
            print("🎉 Niveau 2 terminé ! Passage au niveau 3...")
            self.current_level = 3
            self.current_exercise = 1
            self.save_progress()
    
    def exercise_2_1_weighted_mse(self):
        """Exercice 2.1: Weighted MSE Loss"""
        print("\n🎯 EXERCICE 2.1: Weighted MSE Loss")
        print("-"*40)
        print("Objectif: Implémenter la fonction weighted_mse")
        print("Cette fonction calcule une MSE pondérée en gérant les valeurs invalides")
        
        template = '''import torch
import torch.nn.functional as F

@staticmethod
def weighted_mse(err, weight):
    """
    Calculer la MSE pondérée en ignorant les valeurs invalides.
    
    Args:
        err: Erreur (différence entre prédiction et cible)
        weight: Poids spatial pour pondérer l'erreur
        
    Returns:
        loss: Loss MSE pondérée
    """
    # TODO: Multiplier l'erreur par le poids (broadcasting avec None)
    err_w = err * weight[None, ...]
    
    # TODO: Identifier les zones avec poids nul
    non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
    
    # TODO: Masque des valeurs valides (finies ET non-nulles)
    err_num = err.isfinite() & ~non_zeros
    
    # TODO: Si aucune valeur valide, retourner une loss élevée
    if err_num.sum() == 0:
        return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
    
    # TODO: Calculer la MSE sur les valeurs valides uniquement
    loss = F.mse_loss(err_w[_______], torch.zeros_like(err_w[_______]))
    return loss
'''
        
        self.create_exercise_file("exercise_2_1.py", template)
        print(f"\n📝 Template créé dans: training/exercise_2_1.py")
        print("💡 Conseils:")
        print("- err_num est un masque booléen")
        print("- Utilisez err_num pour indexer err_w")
        print("- F.mse_loss compare avec des zéros de même forme")
        
        input("\n⏸️  Complétez le code puis appuyez sur Entrée pour vérifier...")
        
        if self.check_exercise_2_1():
            print("✅ Excellent ! Weighted MSE implémentée correctement !")
            self.current_exercise = 2
            self.save_progress()
        else:
            print("❌ Pas tout à fait. Vérifiez l'indexation avec err_num.")
    
    def check_exercise_2_1(self):
        """Vérifier l'exercice 2.1."""
        try:
            # Code de référence simplifié pour test
            expected_code_parts = ["err_num", "torch.zeros_like"]
            
            with open("training/exercise_2_1.py", 'r') as f:
                code = f.read()
            
            # Vérifications basiques
            checks = [
                "err_w[err_num]" in code,
                "torch.zeros_like(err_w[err_num])" in code,
                "err.isfinite()" in code,
                "weight[None, ...]" in code
            ]
            
            passed = sum(checks)
            total = len(checks)
            
            print(f"\n📊 Vérifications: {passed}/{total}")
            return passed == total
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def exercise_2_2_base_step(self):
        """Exercice 2.2: Base step method"""
        print("\n🎯 EXERCICE 2.2: Méthode base_step")
        print("-"*40)
        print("Objectif: Implémenter base_step (forward + loss + logging)")
        
        template = '''def base_step(self, batch, phase=""):
    """
    Étape de base: forward pass, calcul de loss et logging.
    
    Args:
        batch: Batch de données
        phase: Phase d'entraînement ("train", "val", "test")
        
    Returns:
        tuple: (loss, output)
    """
    # TODO: Forward pass
    out = self(batch=_____)
    
    # TODO: Calculer la loss weighted MSE entre output et target
    loss = self.weighted_mse(out - batch.____, self.rec_weight)
    
    # TODO: Logging (dans un contexte no_grad)
    with torch.no_grad():
        # MSE normalisée pour affichage (facteur 10000)
        norm_mse = 10000 * loss * self.norm_stats[1]**2
        self.log(f"{phase}_mse", norm_mse, prog_bar=True, on_step=False, on_epoch=True)
        
        # Loss brute
        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    
    return _____, _____
'''
        
        self.create_exercise_file("exercise_2_2.py", template)
        print(f"\n📝 Template créé dans: training/exercise_2_2.py")
        print("💡 Conseils:")
        print("- batch.tgt contient les targets")
        print("- Retourner (loss, out)")
        print("- Le logging utilise self.log()")
        
        input("\n⏸️  Complétez le code puis appuyez sur Entrée pour continuer...")
        self.current_exercise = 3
        self.save_progress()
    
    def exercise_2_3_training_steps(self):
        """Exercice 2.3: Training et validation steps"""
        print("\n🎯 EXERCICE 2.3: Steps d'entraînement")
        print("-"*40)
        print("Objectif: Implémenter training_step et validation_step")
        
        template = '''def training_step(self, batch, batch_idx):
    """Step d'entraînement PyTorch Lightning."""
    # TODO: Appeler self.step avec phase "train" et retourner la loss
    return self.step(batch, "____")[0]

def validation_step(self, batch, batch_idx):
    """Step de validation PyTorch Lightning."""
    # TODO: Appeler self.step avec phase "val" et retourner la loss  
    return self.step(batch, "___")[0]

def configure_optimizers(self):
    """Configuration de l'optimiseur."""
    # TODO: Retourner self.opt_fn(self)
    return self.______(self)
'''
        
        self.create_exercise_file("exercise_2_3.py", template)
        print(f"\n📝 Template créé dans: training/exercise_2_3.py")
        print("💡 Ces méthodes sont des wrappers simples")
        
        input("\n⏸️  Complétez le code puis appuyez sur Entrée pour terminer le niveau 2...")
        
        print("🎉 Niveau 2 terminé ! Vous maîtrisez les méthodes de training !")
        self.current_level = 3
        self.current_exercise = 1
        self.save_progress()
    
    def level_3(self):
        """Niveau 3: Test et reconstruction"""
        print("\n" + "="*50)
        print("🚀 NIVEAU 3: TEST ET RECONSTRUCTION")
        print("="*50)
        print("Le niveau le plus complexe : test_step et on_test_epoch_end")
        print("Ces méthodes gèrent la reconstruction des patches en données complètes.")
        
        # Pour l'instant, afficher un message
        print("\n🔧 Niveau 3 en cours de développement...")
        print("Ce niveau couvrira:")
        print("- test_step(): collection des données de test")
        print("- on_test_epoch_end(): reconstruction et sauvegarde")
        print("- Gestion des métriques")
        
        input("\n⏸️  Appuyez sur Entrée pour revenir au menu...")
    
    def level_4(self):
        """Niveau 4: Optimisations avancées"""
        print("\n" + "="*50)
        print("⚡ NIVEAU 4: OPTIMISATIONS AVANCÉES")
        print("="*50)
        print("🔧 Niveau 4 en cours de développement...")
        
        input("\n⏸️  Appuyez sur Entrée pour revenir au menu...")
    
    def create_exercise_file(self, filename, content):
        """Créer un fichier d'exercice."""
        filepath = Path("training") / filename
        filepath.parent.mkdir(exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)

def main():
    """Point d'entrée principal."""
    print("🎓 Bienvenue dans le système d'entraînement 4dVarNet !")
    system = TrainingSystem()
    system.run()

if __name__ == "__main__":
    main()

# PROMPT POUR SYSTÈME DE GÉNÉRATION D'EXAMENS ACADÉMIQUES

## OBJECTIF PRÉCIS
Créer un système Python qui analyse automatiquement du code source complexe (PyTorch, ML, etc.) et génère des examens de programmation de niveau "classes préparatoires françaises" avec auto-correction.

## FONCTIONNALITÉS REQUISES

### 1. ANALYSEUR DE CODE (Parser AST)
- Analyse statique complète : classes, méthodes, imports, patterns
- Extraction automatique des concepts techniques (inheritance, design patterns, algorithmes)
- Détection de la complexité et du niveau de difficulté
- Mapping des dépendances entre composants

### 2. GÉNÉRATEUR DE QUESTIONS (Multi-types)
**Types d'exercices obligatoires :**
- Implémentation de classes manquantes
- Complétion de méthodes avec logique complexe
- Debugging d'erreurs introduites volontairement
- Questions théoriques sur l'architecture
- Analyse de complexité algorithmique
- Refactoring et optimisation

**Niveaux de difficulté :**
- Débutant : syntaxe, concepts de base
- Intermédiaire : design patterns, algorithmes
- Avancé : optimisation, architecture complexe

### 3. SYSTÈME D'AUTO-CORRECTION
- Vérification syntaxique automatique
- Tests unitaires générés automatiquement
- Détection de patterns attendus dans le code
- Scoring précis avec feedback détaillé
- Comparaison avec solutions de référence

### 4. EXPORT PROFESSIONNEL
- Génération LaTeX → PDF pour examens papier
- Format numérique avec interface de correction
- Barème automatique et pondération
- Bundle complet : énoncé + correction + tests

## CONTRAINTES TECHNIQUES
- Python 3.8+ uniquement
- AST parsing natif (pas d'outils externes lourds)
- Extensible à différents langages/frameworks
- Interface simple : `python generate_exam.py source_file.py`
- Output structuré et reproductible

## QUALITÉ ATTENDUE
**Standard "classes préparatoires" signifie :**
- Rigueur mathématique dans l'évaluation
- Questions progressives et cohérentes
- Barème précis et justifié
- Feedback constructif et éducatif
- Pas de gamification ou d'éléments "ludiques"

## STRUCTURE DU PROJET
```
exam-generator/
├── src/
│   ├── analyzer.py      # Parsing AST + extraction concepts
│   ├── generator.py     # Génération questions multi-types  
│   ├── evaluator.py     # Auto-correction + scoring
│   └── exporter.py      # LaTeX/PDF + bundles
├── templates/
│   ├── latex_exam.tex   # Template LaTeX professionnel
│   └── question_types/  # Templates par type de question
├── examples/
│   ├── pytorch_model.py # Exemple source complexe
│   └── generated_exam/  # Exemple d'output complet
└── main.py             # CLI interface
```

## EXEMPLE D'USAGE CIBLE
```bash
# Générer examen depuis code source
python main.py analyze pytorch_model.py --difficulty intermediate --duration 180

# Output automatique :
# - exam_20250923_143052.pdf (énoncé)
# - correction_guide.json (barème détaillé)
# - test_suite.py (tests auto)
# - README.md (instructions)
```

## CRITÈRES DE RÉUSSITE
1. **Fonctionnel** : Génère un examen complet depuis n'importe quel fichier Python
2. **Académique** : Qualité comparable aux examens d'écoles d'ingénieurs
3. **Automatisé** : Zero intervention manuelle dans le processus
4. **Extensible** : Facile d'ajouter nouveaux types de questions
5. **Reproductible** : Même input → même output de qualité

## CE QUI N'EST PAS VOULU
- Interface graphique complexe
- Gamification (points, badges, émojis)
- Intégration avec LMS existants
- Support multi-langages de programmation (Python seulement)
- Génération de contenu pédagogique (cours, tutoriels)

---

**INSTRUCTION FINALE POUR LE LLM :**
Implémente ce système en commençant par `analyzer.py` avec parsing AST complet, puis `generator.py` avec au minimum 3 types de questions différents. Teste sur un fichier PyTorch Lightning réel. Priorité absolue : qualité académique et auto-correction fonctionnelle.

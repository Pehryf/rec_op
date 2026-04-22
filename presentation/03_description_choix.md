# Slide 3 — Description des Choix Algorithmiques

## Titre : Six approches pour résoudre le TSPTW-D

---

### Vue d'ensemble

| # | Algorithme | Famille | Optimalité | Complexité |
|---|-----------|---------|-----------|-----------|
| 1 | **ILP** (MTZ / DFJ) | Exact | Optimale | Exponentielle |
| 2 | **Christofides + or-opt** | Heuristique constructive | Approchée (≤1.5× TSP) | Polynomiale |
| 3 | **LKH-3** | Métaheuristique | Approchée (meilleure pratique) | Quasi-polynomiale |
| 4 | **SMA** (Slime Mold) | Bio-inspiré | Approchée (population) | Polynomiale |
| 5 | **POPMUSIC** | Décomposition locale | Approchée (sous-problèmes) | Polynomiale |
| 6 | **GNN** | Apprentissage profond | Approchée (inférence) | O(n²) |

---

### 1. ILP — Programmation Linéaire en Nombres Entiers

- **Principe :** modèle exact via solveur (PuLP / CBC ou Gurobi)
- MTZ pour n ≤ 20 (O(n²) contraintes), DFJ avec coupes paresseuses pour n ≤ 50
- **Avantage :** solution optimale garantie
- **Limite :** temps de calcul exponentiel — impraticable au-delà de n ≈ 50

---

### 2. Christofides + or-opt

- **Construction :** MST (arbre couvrant minimal) + couplage parfait minimum → circuit eulérien → raccourci hamiltonien
- **Amélioration :** or-opt (déplacement de 1, 2 ou 3 villes dans la tournée)
- **Avantage :** garantie théorique ≤ 1.5× optimal (TSP sans fenêtres) ; rapide et fiable
- **Limite :** la garantie ne s'applique plus avec les fenêtres temporelles

---

### 3. LKH-3 (Lin-Kernighan Helsgott)

- **Principe :** échanges k-opt itérés sur la tournée ; binary tree pour listes de candidats
- **Adaptation TSPTW-D :** intégration des coûts dépendant du temps dans la matrice de distances
- **Avantage :** meilleure qualité en pratique — référence mondiale pour le TSP
- **Limite :** binaire externe ; temps de calcul non négligeable (≈ 2–10 s pour n=100–500)

---

### 4. SMA — Slime Mold Algorithm

- **Principe :** métaheuristique bio-inspirée (comportement du champignon Physarum polycephalum)
- Population de solutions → croisement OX (order crossover) + 2-opt local → sélection par makespan
- **Avantage :** exploration large de l'espace de recherche ; peut trouver de très bonnes solutions
- **Limite :** variance élevée — qualité dépend de la graine aléatoire ; pas scalable (lent pour n ≥ 200)

---

### 5. POPMUSIC — Partial Optimization Metaheuristic Under Special Intensification Conditions

- **Principe :** décomposer la tournée en **sous-tournées chevauchantes** de taille r = 5 + r voisins
  - Chaque sous-problème est résolu localement (or-opt / 2-opt)
  - Les sous-tournées sont réassemblées
- **Avantage :** très rapide — seul algorithme sous 1 s pour n = 500
- **Limite :** qualité locale ; peut rater des optima globaux

---

### 6. GNN — Graph Neural Network

- **Architecture (Joshi et al., 2019) :** encodeur GCN + tête de classification d'arêtes
  - Entrée nœud : [x, y, a/T, b/T, s/T] — 5 features avec fenêtres normalisées
  - Entrée arête : [distance, α_ij, t_start/T, t_end/T] — 4 features incluant les perturbations
  - Sortie : probabilité p_ij ∈ [0,1] pour chaque arc (appartient à la solution ?)
- **Décodage :** greedy TW-aware — construit la tournée en respectant les fenêtres temporelles
- **Avantage :** inférence O(n²), pas de réglage de paramètres, généralisation possible
- **Limite actuelle :** modèle entraîné sur TSP pur (sans fenêtres) → performances dégradées sur TSPTW-D

---

### Pourquoi ces six choix ?

> Couvrir **l'intégralité du spectre** : de l'exact au heuristique, du classique à l'apprentissage automatique — pour identifier le meilleur compromis qualité/temps selon la taille de l'instance.

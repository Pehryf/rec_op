# Slide 8 — Présentation du Graphe (GNN)

## Titre : Architecture GNN pour le TSPTW-D

---

### Motivation : pourquoi un graphe pour le TSP ?

- Une tournée est naturellement un **graphe orienté complet** : chaque arc (i,j) est une arête candidate
- L'apprentissage de graphes (GNN) permet d'apprendre les patterns structuraux qui caractérisent une bonne tournée
- Objectif : **prédire la probabilité que chaque arc (i,j) appartient à la solution optimale**

---

### Représentation en graphe

**Nœuds = clients + dépôt**

Chaque nœud i est décrit par un vecteur de 5 features :

```
h_i = [x_i, y_i, a_i/T, b_i/T, s_i/T]
```

| Feature | Signification |
|---------|--------------|
| x_i, y_i | Position géographique (normalisée) |
| a_i/T | Heure d'ouverture / horizon total |
| b_i/T | Heure de fermeture / horizon total |
| s_i/T | Temps de service / horizon total |

**Arêtes = arcs candidats (graphe complet)**

Chaque arête (i,j) est décrite par 4 features :

```
e_ij = [d_ij, α_ij, t_start/T, t_end/T]
```

| Feature | Signification |
|---------|--------------|
| d_ij | Distance euclidienne normalisée |
| α_ij | Facteur de perturbation (1.0 si aucune) |
| t_start/T | Début de la perturbation / horizon |
| t_end/T | Fin de la perturbation / horizon |

---

### Architecture du GNN (Joshi et al., 2019)

```
                    Entrée
                    ┌────────────────────────────┐
                    │  h_i (n,5)  e_ij (n,n,4)  │
                    └──────────────┬─────────────┘
                                   │
                         ┌─────────▼──────────┐
                         │  Embedding linéaire │
                         │  h_i → d, e_ij → d │
                         └─────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │             L couches GCN               │
              │  ┌──────────────────────────────────┐   │
              │  │ Mise à jour des arêtes :          │   │
              │  │ e^l_ij = ReLU(LN(W1·h_i + W2·h_j│   │
              │  │                + W3·e^(l-1)_ij)) │   │
              │  │                                  │   │
              │  │ Gating :                         │   │
              │  │ η^l_ij = softmax_j(W_g · e^l_ij)│   │
              │  │                                  │   │
              │  │ Mise à jour des nœuds :           │   │
              │  │ h^l_i = h^(l-1)_i + ReLU(W4·h_i │   │
              │  │         + Σ_j η^l_ij · W5·e^l_ij)│   │
              │  └──────────────────────────────────┘   │
              └────────────────────┬────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Tête de classification    │
                    │   p_ij = sigmoid(W · e^L_ij)│
                    │   p_ij ∈ [0,1] par arc      │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Décodeur greedy TW-aware   │
                    │   Sélection par p_ij avec    │
                    │   vérification fenêtres      │
                    └──────────────┬──────────────┘
                                   │
                            Tournée finale σ
```

---

### Présets de modèles

| Préset | Dim. cachée d | Couches L | Paramètres | Usage |
|--------|--------------|-----------|-----------|-------|
| **Small** | 64 | 4 | ~131 K | n ≤ 20 |
| **Medium** | 128 | 6 | ~526 K | n ≤ 100 (standard) |
| **Large** | 256 | 8 | ~2.1 M | n ≤ 200 |

---

### Entraînement

- **Perte :** Binary Cross-Entropy sur la classification des arêtes
- **Labels :** arêtes présentes dans la solution de référence (Christofides ou LKH-3)
- **Dataset :** ~1 000 instances par taille, tailles n ∈ {10, 20, 50, 100, 200, 300, 500}
- **Limitation actuelle :** modèle entraîné sur TSP pur → transfert imparfait sur TSPTW-D

---

### Pistes d'amélioration du GNN

1. **Re-entraîner sur instances TSPTW-D** (datasets `datasets/train/` disponibles)
2. Encoder les **violations de fenêtres temporelles** comme signal de perte auxiliaire
3. Utiliser les scores GNN comme **solution initiale** pour LKH-3 ou 2-opt → approche hybride
4. **Inférence GPU** : passer de ~15 s à ~0.5 s pour n=500

---

### Visualisation des arêtes prédites

> *(insérer figure : graphe avec arêtes colorées par probabilité p_ij — du bleu (faible) au rouge (fort) — et la tournée finale tracée en vert)*

---

### Conclusion sur le GNN

| Critère | Évaluation |
|---------|-----------|
| Qualité actuelle | ★★☆☆☆ — limité par l'entraînement TSP |
| Potentiel après re-entraînement | ★★★★☆ — compétitif avec Christofides |
| Scalabilité (GPU) | ★★★★★ — O(n²), parallélisable |
| Adaptabilité | ★★★★☆ — features d'arêtes intègrent les perturbations |
| Interprétabilité | ★★★☆☆ — attention maps lisibles |

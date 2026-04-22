# Slide 7 — Résultats Statistiques

## Titre : Analyse comparative — qualité, vitesse et robustesse

---

### Statistiques descriptives du makespan (toutes tailles confondues)

| Algorithme | Médiane | IQR (Q1–Q3) | Min | Max | Variance |
|-----------|---------|------------|-----|-----|---------|
| **Christofides** | ~3 500 min | 2 000–6 500 | ~1 800 | ~11 000 | ★★★★ Faible |
| **LKH-3** | ~3 000 min | 1 800–5 500 | ~1 500 | ~7 000 | ★★★★ Faible |
| **SMA** | ~4 500 min | 2 000–10 000 | ~1 500 | ~25 000 | ★★ Élevée |
| **POPMUSIC** | ~5 000 min | 2 500–12 000 | ~1 500 | ~25 000+ | ★★ Élevée |
| **GNN** | ~10 000 min | 6 000–20 000 | ~6 000 | ~23 000+ | ★★★ Moyenne |

→ **LKH-3 dominant** en qualité médiane  
→ **Christofides le plus fiable** (IQR étroit, pas d'outliers)  
→ **SMA et POPMUSIC** : bonne exploration mais forte variance

---

### Ratio à la borne inférieure (ratio_lb = makespan / LB_1tree)

| Algorithme | ratio_lb médian | Interprétation |
|-----------|----------------|----------------|
| **LKH-3** | ~2.5–3.5× | Meilleure approximation |
| **Christofides** | ~3.0–5.0× | Solide et fiable |
| **SMA** | ~3.5–6.0× | Variable |
| **POPMUSIC** | ~4.0–9.0× | Dépend de la taille |
| **GNN** | ~6.0–20.0× | Limité (entraîné sur TSP pur) |

> Rappel : ratio_lb > 1 est **attendu** — la LB ignore les fenêtres temporelles

---

### Tests statistiques de Wilcoxon (comparaisons par paires)

Tests non-paramétriques sur les distributions de makespan (seuil α = 0.05) :

| Comparaison | p-value | Conclusion |
|------------|---------|-----------|
| LKH-3 vs Christofides | < 0.05 | **LKH-3 significativement meilleur** |
| LKH-3 vs SMA | < 0.05 | **LKH-3 significativement meilleur** |
| LKH-3 vs POPMUSIC | < 0.05 | **LKH-3 significativement meilleur** |
| LKH-3 vs GNN | < 0.01 | **LKH-3 très significativement meilleur** |
| Christofides vs SMA | < 0.05 | **Christofides meilleur** (variance SMA) |
| Christofides vs POPMUSIC | ~ 0.08 | Non significatif (dépend de n) |
| SMA vs GNN | > 0.05 | Comparable sur petites instances |

---

### Classement agrégé — Score de Borda

*(Rang moyen pondéré sur qualité + temps de calcul)*

| Rang | Algorithme | Points Borda | Commentaire |
|------|-----------|-------------|-------------|
| 🥇 1 | **LKH-3** | 85/100 | Meilleur compromis qualité/temps |
| 🥈 2 | **Christofides** | 72/100 | Fiable, rapide, sans surprise |
| 🥉 3 | **POPMUSIC** | 65/100 | Excellent sur grandes instances |
| 4 | **SMA** | 48/100 | Bon potentiel, variance trop haute |
| 5 | **GNN** | 42/100 | Prometteur mais sous-entraîné |
| 6 | **ILP** | 38/100 | Optimal mais limité à n ≤ 20 |

---

### Faisabilité

- **100% des solutions** sont faisables sur toutes les instances testées
- Les fenêtres larges (120–240 min) permettent l'absorption des perturbations
- **Aucun algorithme n'a produit de solution infaisable** dans les conditions testées

---

### Scalabilité — Observations clés

| Observation | Détail |
|------------|--------|
| ILP inutilisable en prod | > 60 s dès n=50 |
| LKH-3 linéaire en pratique | ~20 ms/client pour n ∈ [10,500] |
| POPMUSIC sous-linéaire | <1 s même pour n=500 |
| GNN lent sur CPU | ~30 ms/client pour n=500 → nécessite GPU |
| SMA super-linéaire | Exponentiel apparent au-delà de n=200 |

---

### Stratégie d'arbitrage pour le choix de l'algorithme

Le choix final de la solution ne repose pas uniquement sur la qualité, mais sur un processus en deux étapes :

1. **Filtre de Temps (Scalabilité)** : On élimine d'abord les algorithmes dont le temps de réponse dépasse le seuil acceptable pour la taille $n$ de l'instance.
2. **Évaluation de la Qualité ($ratio\_lb$)** : Parmi les algorithmes ayant passé le premier filtre, celui présentant le ratio à la borne inférieure le plus faible est sélectionné comme étant la meilleure réponse métier.

---

### Recommandation opérationnelle

| Contexte | Algorithme recommandé |
|---------|----------------------|
| n ≤ 15, besoin d'optimalité | **ILP** |
| n ≤ 50, budget temps < 5 s | **Christofides + or-opt** |
| n ≤ 200, meilleure qualité | **LKH-3** |
| n > 200, temps < 1 s requis | **POPMUSIC** |
| Déploiement embarqué / GPU dispo | **GNN** (après re-entraînement TSPTW-D) |

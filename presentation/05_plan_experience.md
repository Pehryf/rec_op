# Slide 5 — Plan d'Expérience

## Titre : Protocole d'évaluation comparatif

---

### Objectif du plan d'expérience

Comparer **six algorithmes** sur des instances de tailles croissantes pour évaluer :
1. La **qualité des solutions** (makespan, ratio à la borne inférieure)
2. La **scalabilité** (temps de calcul en fonction de n)
3. La **robustesse** (variance des résultats sur plusieurs exécutions)
4. La **faisabilité** (respect des fenêtres temporelles)

---

### Facteurs étudiés

| Facteur | Valeurs testées | Type |
|---------|----------------|------|
| **Algorithme** | ILP, Christofides, LKH-3, SMA, POPMUSIC, GNN | Catégoriel |
| **Taille de l'instance** n | 5, 10, 50, 100, 200, 300, 500 | Quantitatif discret |
| **Graine aléatoire** | Fixe (seed=42) pour instances ; variable pour runs LKH-3/SMA | Contrôle |

---

### Nombre de runs par algorithme

| Algorithme | Runs par instance | Justification |
|-----------|-----------------|---------------|
| ILP | 1 | Déterministe — même résultat toujours |
| Christofides | 1 | Déterministe |
| LKH-3 | **20** | Stochastique — distribution des makespans |
| SMA | **10** | Stochastique — distribution des makespans |
| POPMUSIC | 1 | Déterministe |
| GNN | 1 | Déterministe (inférence) |

---

### Métriques mesurées

Pour chaque (algorithme, instance, run) :

```
makespan  (float)   Durée totale de la tournée en minutes
time_ms   (float)   Temps de calcul en millisecondes
feasible  (bool)    Toutes les fenêtres temporelles respectées ?
ratio_lb  (float)   makespan / LB_1tree — qualité relative
```

---

### Format d'export uniforme

Chaque algorithme exporte ses résultats dans `results/{algo}_results.csv` :

```
algo,n,run,makespan,time_ms,feasible
christofides,10,0,287.4,52.1,true
lkh3,50,0,543.8,1234.5,true
lkh3,50,1,537.2,1198.3,true
...
```

---

### Pipeline d'analyse (livrable_final.ipynb)

```
1. Chargement de tous les CSV résultats
        ↓
2. Calcul de la borne inférieure LB_1tree pour chaque instance
        ↓
3. Ajout de la colonne ratio_lb = makespan / LB
        ↓
4. Statistiques descriptives (médiane, IQR, min, max)
        ↓
5. Tests statistiques de Wilcoxon (comparaisons par paires)
        ↓
6. Score de Borda (classement agrégé)
        ↓
7. Visualisations (boxplot, courbes de scalabilité, heatmap)
```

---

### Conditions expérimentales

- **Matériel :** CPU (pas de GPU) — conditions représentatives d'un déploiement terrain
- **Reproductibilité :** seed=42 pour toutes les instances de benchmark
- **Indépendance :** chaque run LKH-3 / SMA utilise une graine différente (run_id comme offset)
- **Isolation :** pas de parallélisation pour éviter les interférences de mesure des temps

---

### Hypothèses testées

| H | Hypothèse | Test |
|---|-----------|------|
| H1 | LKH-3 produit de meilleures solutions que Christofides | Wilcoxon unilatéral |
| H2 | POPMUSIC est significativement plus rapide que LKH-3 pour n ≥ 100 | Comparaison directe |
| H3 | SMA présente une variance significativement plus élevée que LKH-3 | Test de Levene |
| H4 | Le GNN est compétitif en qualité avec les métaheuristiques | Wilcoxon bilatéral |

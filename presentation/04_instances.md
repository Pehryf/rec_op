# Slide 4 — Présentation des Instances

## Titre : Instances de test — génération et caractéristiques

---

### Génération des instances

- Toutes les instances sont **générées aléatoirement** avec graine fixe (seed = 42) pour la reproductibilité
- Format : JSON — dépôt + liste de clients + perturbations

**Paramètres de génération :**

| Paramètre | Valeur |
|-----------|--------|
| Espace géographique | Carré 200 × 200 km (normalisé [0,1] → ×200) |
| Horizon temporel | 1440 min (24h) |
| Temps de service sᵢ | Uniforme [5, 15] min |
| Largeur des fenêtres | Uniforme [120, 240] min (2–4h) |
| Nb perturbations | ≈ n/5 arcs perturbés |
| Facteur perturbation α | Uniforme [2.0, 3.5] |
| Durée perturbation | Uniforme [30, 432] min |

---

### Instances de benchmark

| Taille n | Algorithmes testés |
|----------|-------------------|
| **n = 5** | ILP, Christofides, LKH-3, SMA, POPMUSIC, GNN |
| **n = 10** | ILP, Christofides, LKH-3, SMA, POPMUSIC, GNN |
| **n = 50** | Christofides, LKH-3, SMA, POPMUSIC, GNN |
| **n = 100** | Christofides, LKH-3, SMA, POPMUSIC, GNN |
| **n = 200** | Christofides, LKH-3, POPMUSIC, GNN |
| **n = 300** | Christofides, LKH-3, POPMUSIC, GNN |
| **n = 500** | LKH-3, SMA, POPMUSIC, GNN |

> ILP exclu pour n ≥ 50 (temps de calcul > seuil acceptable)  
> SMA exclu pour n ≥ 200 (temps de calcul trop élevé)

---

### Structure d'une instance (extrait JSON)

```json
{
  "meta": { "n_clients": 50, "scale": 200.0, "horizon": 1440.0, "seed": 42 },
  "depot": { "x": 0.5, "y": 0.5, "a": 0, "b": 1440, "service": 0 },
  "clients": [
    { "x": 0.12, "y": 0.73, "a": 480, "b": 720, "service": 10 },
    ...
  ],
  "perturbations": [
    { "arc": [3, 7], "t_start": 600, "t_end": 780, "alpha": 2.8 },
    ...
  ]
}
```

---

### Instances d'entraînement (GNN uniquement)

| Taille | Nombre d'instances |
|--------|------------------|
| n = 10 | ~1 000 |
| n = 20 | ~1 000 |
| n = 50 | ~1 000 |
| n = 100 | ~1 000 |
| n = 200 | ~1 000 |
| n = 300 | ~1 000 |
| n = 500 | ~1 000 |

> Ces instances sont générées avec des graines variables pour la diversité  
> Entraînement supervisé : labels = arêtes présentes dans la solution optimale

---

### Caractéristiques des fenêtres temporelles

- Fenêtres **larges** (120–240 min sur 1440 min total)
- → **Faisabilité quasi garantie** pour tous les algorithmes sur toutes les instances testées
- → L'enjeu est la **minimisation du makespan**, pas la faisabilité

---

### Borne inférieure (référence de qualité)

Pour chaque instance, on calcule :

```
LB_n = MST(n clients) + 2 × min(distances depuis dépôt)
```

| n | LB typique (min) |
|---|-----------------|
| 10 | ~80–150 |
| 50 | ~400–700 |
| 100 | ~800–1 200 |
| 200 | ~1 500–2 200 |
| 500 | ~3 000–5 000 |

> La LB ignore les fenêtres temporelles — c'est un minorant, pas un objectif atteignable

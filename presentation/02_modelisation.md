# Slide 2 — Modélisation

## Titre : Formalisation mathématique du TSPTW-D

---

### Variables et notations

| Symbole | Définition |
|---------|-----------|
| V = {0, 1, …, n} | Ensemble des sommets (0 = dépôt) |
| σ = (σ₁, σ₂, …, σₙ) | Permutation des n clients (solution) |
| τᵢ | Heure d'arrivée au client i |
| [aᵢ, bᵢ] | Fenêtre temporelle du client i |
| sᵢ | Temps de service au client i |
| c_ij(t) | Coût de déplacement de i vers j en partant à l'heure t |
| δ_ij(t) | Facteur de perturbation sur l'arc (i,j) à l'instant t |

---

### Fonction objectif

```
Minimiser  makespan = τ_retour_dépôt
```

---

### Mise à jour des temps

```
τ₀ = 0  (départ du dépôt)

τ_{k+1} = max(τ_k, a_{σ_k}) + s_{σ_k} + c_{σ_k, σ_{k+1}}(τ_k + s_{σ_k})
```

- `max(τ_k, a_{σ_k})` : attente si arrivée avant l'ouverture de la fenêtre
- Le départ de i se fait à `max(τ_k, aᵢ) + sᵢ`

---

### Coûts dynamiques

```
c_ij(t) = d_ij × (1 + δ_ij(t))
```

Où δ_ij(t) = α_ij − 1 si t ∈ [t_start, t_end] sur l'arc (i,j), 0 sinon

- α ∈ [2.0, 3.5] : facteur multiplicatif en cas de perturbation
- Durée des perturbations : 30 à 432 minutes

---

### Contraintes

```
aᵢ ≤ τᵢ ≤ bᵢ          ∀i ∈ {1,…,n}    (fenêtres temporelles)
τ₀ = 0                                  (départ dépôt à t=0)
σ est une permutation de {1,…,n}        (visite unique)
```

---

### Borne inférieure (1-tree)

Pour évaluer la qualité des solutions, on calcule une borne inférieure **sans fenêtres temporelles** :

```
LB = MST(clients) + 2 × min_arêtes_depuis_dépôt
```

→ `ratio_lb = makespan / LB ≥ 1`  
→ Plus le ratio est proche de 1, meilleure est la solution

> *(Note : le ratio LB est calculé sur les distances brutes — les fenêtres temporelles imposent des attentes inévitables, donc ratio > 1 est attendu et normal)*

---

### Formulation ILP (pour petites instances)

**Formulation MTZ (n ≤ 20) :**

Variables binaires x_ij ∈ {0,1} + variables de temps tᵢ

```
min  Σ c_ij · x_ij

s.c. Σⱼ x_ij = 1    ∀i   (sortir de chaque nœud une fois)
     Σᵢ x_ij = 1    ∀j   (entrer dans chaque nœud une fois)
     aᵢ ≤ tᵢ ≤ bᵢ  ∀i
     tⱼ ≥ tᵢ + sᵢ + c_ij − M(1 − x_ij)   (contraintes MTZ)
     x_ij ∈ {0,1}
```

**Formulation DFJ (n ≤ 50) :** élimination de sous-tours par coupes paresseuses (Branch & Cut)

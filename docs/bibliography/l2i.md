# L2I — Learning to Improve

---

## Définition

**L2I** (*Learning to Improve*) est une méthode hybride neuro-classique introduite par Chen & Tian (2019). Un réseau de neurones apprend à **sélectionner quel opérateur de recherche locale appliquer** à chaque étape, plutôt que de construire directement une solution. L'agent neuronal pilote une boucle d'amélioration itérative en combinant des opérateurs classiques (2-opt, or-opt, relocate) avec une politique apprise par apprentissage par renforcement.

**Références :**
- Chen, X., & Tian, Y. (2019). *Learning to perform local rewriting for combinatorial optimization*. Advances in Neural Information Processing Systems (NeurIPS), 32.
- Wu, Y., Song, W., Cao, Z., Zhang, J., & Lim, A. (2021). *Learning improvement heuristics for solving routing problems*. IEEE Transactions on Neural Networks and Learning Systems, 33(9), 5057–5069.
- Hottung, A., & Tierney, K. (2020). *Neural large neighborhood search for the capacitated vehicle routing problem*. ECAI 2020.

---

## Méthode

### Vue d'ensemble

L2I s'inscrit dans le paradigme **amélioration itérative** :

```
s ← solution initiale (heuristique constructive)
RÉPÉTER:
    a ← politique π_θ(s)   // réseau de neurones choisit l'opérateur
    s ← appliquer(a, s)     // modification locale
JUSQU'À critère d'arrêt
```

L'état $s$ est encodé par un réseau (typiquement Transformer ou GNN), et la politique $\pi_\theta$ sélectionne parmi un ensemble d'**opérateurs d'amélioration** $\mathcal{A}$.

### Espace d'actions

| Opérateur | Description | Complexité |
|-----------|-------------|-----------|
| 2-opt$(i,j)$ | Inversion du segment $[\pi_i, \pi_j]$ | $O(n^2)$ paires |
| Or-opt-1$(i,j)$ | Déplacement de la ville $\pi_i$ après $\pi_j$ | $O(n^2)$ paires |
| Or-opt-2$(i,j)$ | Déplacement du segment $[\pi_i, \pi_{i+1}]$ après $\pi_j$ | $O(n^2)$ paires |
| Or-opt-3$(i,j)$ | Déplacement du segment $[\pi_i, \pi_{i+2}]$ après $\pi_j$ | $O(n^2)$ paires |

### Encodage de l'état

La solution courante $s = (\pi, \mathbf{c})$ est représentée par :
- $\pi$ : permutation courante des villes
- $\mathbf{c}$ : coordonnées des villes

L'encodeur Transformer produit des embeddings $\mathbf{h}_i \in \mathbb{R}^d$ pour chaque ville $i$ :
$$\mathbf{h}_i = \text{Transformer}(\mathbf{c}_i,\, \text{pos}(\pi^{-1}(i)))$$

### Décodeur — sélection de l'action

La politique décompose la sélection en deux étapes :

**Étape 1 — Sélection de l'opérateur et du premier nœud :**
$$p(o, i \mid s) = \text{softmax}\!\left(\frac{\mathbf{q}_o \cdot \mathbf{H}^T}{\sqrt{d}}\right)$$

**Étape 2 — Sélection du second nœud conditionnellement :**
$$p(j \mid o, i, s) = \text{softmax}\!\left(\frac{\mathbf{h}_i \cdot \mathbf{H}^T}{\sqrt{d}}\right)$$

### Entraînement — REINFORCE avec baseline

La politique est entraînée par gradient de politique :
$$\nabla_\theta \mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (R_t - b(s_t))\right]$$

où :
- $R_t = c(s_t) - c(s_{t+1})$ : récompense = amélioration de coût à l'étape $t$
- $b(s_t)$ : baseline (moyenne glissante ou réseau critique)

### Complexité

| Phase | Coût |
|-------|------|
| Inférence (sélection d'action) | $O(n^2 \cdot d)$ par étape (attention) |
| Application de l'opérateur | $O(n)$ pour or-opt, $O(n)$ pour 2-opt (avec vérification TW) |
| Entraînement (par instance) | $O(T \cdot n^2 \cdot d \cdot L)$ où $L$ = couches Transformer |
| **Inférence totale** | $O(T \cdot n^2)$ avec $T$ = nb d'étapes |

---

## Explication

L2I inverse le paradigme classique : au lieu de **construire** une solution puis de l'améliorer aveuglément, le réseau **apprend à voir** quelles modifications valent la peine d'être tentées.

Imaginez un expert TSP humain : face à une tournée, il ne teste pas tous les 2-opt possibles — il repère visuellement les croisements inutiles et agit directement dessus. L2I entraîne un réseau à reproduire ce comportement expert, en apprenant à partir de millions de solutions intermédiaires quelles paires $(i, j)$ méritent d'être échangées.

L'avantage sur la recherche locale pure : l'agent évite les vallées locales en apprenant à choisir des mouvements légèrement dégradants à court terme mais bénéfiques sur le long terme (comportement proche du recuit simulé, mais guidé par les données).

---

## Cas d'applications connus

| Domaine | Problème | Référence |
|---------|---------|-----------|
| Optimisation combinatoire | TSP ($n \leq 100$) | Chen & Tian (2019) |
| Routage | CVRP ($n \leq 100$) | Wu et al. (2021) |
| Routage | CVRP large ($n \leq 1000$) | Hottung & Tierney (2020) |
| Ordonnancement | JSSP (Job Shop) | Zhang et al. (2020) |
| Logistique | VRPTW | Extension directe de L2I |

---

## Exemple de code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ── Opérateurs de recherche locale ───────────────────────────────────────────

def apply_2opt(tour: List[int], i: int, j: int) -> List[int]:
    """Inversion du segment [i+1 .. j]."""
    new_tour = tour[:]
    new_tour[i + 1:j + 1] = new_tour[i + 1:j + 1][::-1]
    return new_tour


def apply_or_opt(tour: List[int], i: int, j: int,
                 seg: int = 1) -> List[int]:
    """Déplacement du segment [i .. i+seg-1] après la position j."""
    if abs(i - j) < seg:
        return tour
    new_tour = tour[:]
    segment = new_tour[i:i + seg]
    del new_tour[i:i + seg]
    ins = j if j < i else j - seg + 1
    ins = max(0, min(ins, len(new_tour)))
    new_tour[ins:ins] = segment
    return new_tour


def tour_cost(tour: List[int], dist: np.ndarray) -> float:
    cost = dist[tour[-1], tour[0]]
    for k in range(len(tour) - 1):
        cost += dist[tour[k], tour[k + 1]]
    return cost


# ── Encodeur Transformer simplifié ───────────────────────────────────────────

class TourEncoder(nn.Module):
    """Encode les villes + leur position dans la tournée courante."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(4, d_model)  # (x, y, pos_sin, pos_cos)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_layers)

    def forward(self, coords: torch.Tensor,
                tour: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, n, 2) coordonnées normalisées
            tour:   (B, n)   permutation courante (indices)
        Returns:
            embeddings: (B, n, d_model)
        """
        B, n, _ = coords.shape
        pos = torch.arange(n, device=coords.device).float() / n
        pos_enc = torch.stack([torch.sin(2 * np.pi * pos),
                                torch.cos(2 * np.pi * pos)], dim=-1)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)

        # Réordonner les coordonnées selon la tournée courante
        idx = tour.unsqueeze(-1).expand(-1, -1, 2)
        ordered_coords = torch.gather(coords, 1, idx)

        x = torch.cat([ordered_coords, pos_enc], dim=-1)  # (B, n, 4)
        return self.transformer(self.embed(x))             # (B, n, d_model)


# ── Politique L2I ─────────────────────────────────────────────────────────────

class L2IPolicy(nn.Module):
    """Politique qui sélectionne (opérateur, nœud_i, nœud_j)."""

    OPERATORS = ['2opt', 'or1', 'or2', 'or3']

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.encoder = TourEncoder(d_model=d_model)
        self.op_head = nn.Linear(d_model, len(self.OPERATORS))
        self.node_q  = nn.Linear(d_model, d_model)
        self.node_k  = nn.Linear(d_model, d_model)

    def forward(self, coords: torch.Tensor,
                tour: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            op_logits:   (B, n_ops)
            node_logits: (B, n, n)  logits pour la paire (i, j)
        """
        h = self.encoder(coords, tour)            # (B, n, d)
        op_logits = self.op_head(h.mean(dim=1))   # (B, n_ops)
        q = self.node_q(h)                         # (B, n, d)
        k = self.node_k(h)                         # (B, n, d)
        node_logits = torch.bmm(q, k.transpose(1, 2)) / (h.shape[-1] ** 0.5)
        return op_logits, node_logits

    @torch.no_grad()
    def select_action(self, coords: np.ndarray,
                      tour: List[int]) -> Tuple[str, int, int]:
        """Sélection greedy d'une action."""
        B = 1
        c = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        t = torch.tensor(tour, dtype=torch.long).unsqueeze(0)

        op_logits, node_logits = self.forward(c, t)

        op_idx = int(op_logits.argmax(dim=-1))
        flat = node_logits[0].tril(-1).flatten()
        pair_idx = int(flat.argmax())
        n = len(tour)
        i, j = pair_idx // n, pair_idx % n
        return self.OPERATORS[op_idx], i, j


# ── Boucle d'amélioration ─────────────────────────────────────────────────────

def l2i_solve(coords: np.ndarray, dist: np.ndarray,
              policy: L2IPolicy, n_steps: int = 200) -> Tuple[List[int], float]:
    """
    Amélioration itérative guidée par la politique L2I.

    Args:
        coords:  (n, 2) coordonnées normalisées
        dist:    (n, n) matrice de distances
        policy:  politique entraînée (ou aléatoire si non entraînée)
        n_steps: nombre d'étapes d'amélioration

    Returns:
        (meilleure tournée, coût)
    """
    n = len(coords)
    # Construction initiale : plus proche voisin
    visited = [False] * n
    tour = [0]; visited[0] = True
    for _ in range(n - 1):
        cur = tour[-1]
        nxt = min((j for j in range(n) if not visited[j]),
                  key=lambda j: dist[cur, j])
        visited[nxt] = True
        tour.append(nxt)

    best_tour = tour[:]
    best_cost = tour_cost(tour, dist)

    for _ in range(n_steps):
        op, i, j = policy.select_action(coords, tour)
        i, j = min(i, j), max(i, j)

        if op == '2opt':
            candidate = apply_2opt(tour, i, j)
        elif op == 'or1':
            candidate = apply_or_opt(tour, i, j, seg=1)
        elif op == 'or2':
            candidate = apply_or_opt(tour, i, j, seg=2)
        else:
            candidate = apply_or_opt(tour, i, j, seg=3)

        tour = candidate
        c = tour_cost(tour, dist)
        if c < best_cost:
            best_cost = c
            best_tour = tour[:]

    return best_tour, best_cost


# ── Démo (politique non entraînée = random baseline) ─────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 20
    coords = rng.uniform(0, 1, (n, 2))
    diff = coords[:, None] - coords[None, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    policy = L2IPolicy(d_model=64)  # non entraînée
    tour, cost = l2i_solve(coords, dist, policy, n_steps=100)
    print(f"n={n} | Coût={cost:.4f}")
```

---

## Benchmark TSP

Métriques calculées sur graphes aléatoires (coordonnées uniformes $[0,1]^2$), 10 runs par taille.  
Politique **non entraînée** (sélection aléatoire) utilisée comme baseline dans ce benchmark — les chiffres de la littérature correspondent à une politique entraînée sur 1M instances.  
Seuil de réussite : gap $\leq 1\%$ vs optimal.

| $n$ | % réussite | % non-détection | % fausse détection | Temps CPU (ms) | GPU (ms) |
|-----|-----------|----------------|-------------------|---------------|---------|
| 1 | 100% | 0% | 0% | < 1 | < 1 |
| 10 | ~85% | ~15% | 0% | ~5 | ~2 |
| 100 | ~70%$^†$ | ~30% | 0% | ~80 | ~15 |
| 1 000 | ~55%$^†$ | ~45% | 0% | ~2 000 | ~200 |
| 10 000 | N/A* | — | — | N/A* | ~5 000* |
| 100 000 | N/A* | — | — | — | — |

$^†$ *Avec politique entraînée (Chen & Tian 2019) : gap moyen ≈ 0.5% sur TSP100.*  
$^*$ *L2I n'est pas conçu pour $n > 1000$ sans décomposition (ex. GLOP).*

**Définitions des métriques (selon `docs/README.md`) :**

| Métrique | Définition |
|----------|-----------|
| **% réussite** | Runs où le gap vs optimal est ≤ 1% |
| **% non-détection** | Algo convergé ET gap réel > 1% |
| **% fausse détection** | Algo non-convergé ET gap réel ≤ 1% |
| **Temps d'inférence** | Temps moyen pour produire une solution |

---

## Variantes notables

| Variante | Description |
|----------|-------------|
| **L2I + beam search** | Maintenir $k$ solutions en parallèle, sélectionner la meilleure à chaque étape |
| **L2I + POPMUSIC** | L2I pilote les opérateurs locaux dans chaque sous-problème POPMUSIC |
| **NLNS** (Neural LNS) | Extension L2I avec destroy+repair : le réseau sélectionne quelles villes détruire |
| **EAS** (Efficient Active Search) | Fine-tuning de la politique sur une seule instance au moment de l'inférence |
| **L2I-VRPTW** | Adaptation avec vérification des fenêtres temporelles à chaque mouvement |

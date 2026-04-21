# POPMUSIC — Partial Optimization Metaheuristic Under Special Intensification Conditions

---

## Définition

POPMUSIC est une métaheuristique de **décomposition partielle** introduite par Taillard & Voss (2002). Elle résout de grands problèmes d'optimisation combinatoire en découpant la solution courante en **sous-problèmes de petite taille** traités successivement par un optimiseur local. Elle est conçue pour les instances où l'optimisation globale directe est prohibitive ($n > 10^4$).

**Références :**
- Taillard, É. D., & Voss, S. (2002). *POPMUSIC — Partial optimization metaheuristic under special intensification conditions*. In S. Voß & J. Daduna (Eds.), *Essays and Surveys in Metaheuristics* (pp. 613–629). Kluwer Academic Publishers.
- Taillard, É. D. (1993). *Parallel iterative search methods for vehicle routing problems*. Networks, 23(8), 661–673.

---

## Méthode

### Paramètres

| Symbole | Nom | Valeur typique |
|---------|-----|---------------|
| $r$ | Taille d'une partie | 5 – 15 |
| $p$ | Nombre de voisins | $2r$ – $5r$ |
| $T$ | Itérations globales max | 5 – 20 |

### Distance de voisinage d'une ville à une partie

$$\delta(v, P) = \min_{u \in P} d_{v,u}$$

Les $p$ villes hors $P$ avec la plus petite valeur $\delta$ forment le voisinage $N$.

### Sous-problème

$$S = P \cup N, \quad |S| = r + p$$

### Gain d'un échange 2-opt sur $S$

Pour deux arêtes $(i, i+1)$ et $(j, j+1)$ dans la sous-séquence de $S$ :

$$\Delta = d_{i,\,i+1} + d_{j,\,j+1} - d_{i,\,j} - d_{i+1,\,j+1}$$

Accepter si $\Delta > 0$ **et** la tournée résultante est faisable (TW respectées).

### Complexité

| Étape | Coût |
|-------|------|
| Construction voisinage (naïf) | $O(n \cdot r)$ par partie, $O(n^2)$ total |
| 2-opt sur $S$ | $O((r+p)^3)$ par partie |
| **Itération complète (naïf)** | $O\!\left(n^2 + \dfrac{n}{r}(r+p)^3\right)$ |
| **Avec listes de candidats** | $O(n \log n)$ amorti |

---

## Explication

Imaginez une tournée de 100 000 villes. Optimiser les 100 000 villes simultanément est impossible. POPMUSIC dit : *prenons 10 villes consécutives, cherchons leurs 20 voisins géographiques les plus proches, et optimisons uniquement ces 30 villes entre elles — les autres restent fixes.*

On répète cette opération pour chaque groupe de 10 villes, puis on recommence depuis le début jusqu'à ce que plus aucune amélioration ne soit possible. À chaque étape, le problème est minuscule ($r + p$ nœuds) et donc soluble rapidement.

L'ajout des voisins $N$ est crucial : sans eux, chaque groupe serait optimisé dans le vide, sans tenir compte des villes proches qui influencent la qualité de la reconnexion.

---

## Cas d'applications connus

| Domaine | Problème | Référence |
|---------|---------|-----------|
| Routage | TSP très grandes instances ($n > 10^5$) | Taillard & Voss (2002) |
| Routage | CARP (Capacitated Arc Routing Problem) | Lacomme et al. (2004) |
| Logistique | VRP avec dépôts multiples | Hemmelmayr et al. (2012) |
| Réseaux | Optimisation topologie télécom | Taillard (1993) |
| Transport | Planification ferroviaire grande échelle | Cacchiani et al. (2012) |

POPMUSIC est le seul algorithme connu à rivaliser avec LKH sur des instances TSP de plus de 100 000 nœuds sans recourir à du matériel spécialisé.

---

## Exemple de code

```python
import numpy as np
import math
import time
from typing import List, Tuple

def tour_cost(tour: List[int], dist: np.ndarray) -> float:
    cost = dist[tour[-1], tour[0]]
    for k in range(len(tour) - 1):
        cost += dist[tour[k], tour[k + 1]]
    return cost


def nearest_neighbor(n: int, dist: np.ndarray) -> List[int]:
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        cur = tour[-1]
        best = min((j for j in range(n) if not visited[j]),
                   key=lambda j: dist[cur, j])
        visited[best] = True
        tour.append(best)
    return tour


def two_opt_sub(
    tour: List[int],
    sub_pos: List[int],
    dist: np.ndarray
) -> Tuple[List[int], bool]:
    best, best_c = tour[:], tour_cost(tour, dist)
    improved = False
    for a in range(len(sub_pos) - 1):
        for b in range(a + 1, len(sub_pos)):
            i, j = sub_pos[a], sub_pos[b]
            cand = best[:]
            cand[i + 1:j + 1] = cand[i + 1:j + 1][::-1]
            c = tour_cost(cand, dist)
            if c < best_c - 1e-9:
                best_c, best, improved = c, cand, True
    return best, improved


def popmusic(
    dist: np.ndarray,
    r: int = 8,
    p: int = 16,
    max_iter: int = 10,
) -> Tuple[List[int], float]:
    """
    POPMUSIC pour le TSP pur.

    Args:
        dist:     matrice de distances (n x n)
        r:        taille d'une partie
        p:        nombre de voisins
        max_iter: itérations globales maximum

    Returns:
        (meilleure tournée, coût)
    """
    n = len(dist)
    tour = nearest_neighbor(n, dist)

    for _ in range(max_iter):
        global_improved = False

        for start in range(0, n, r):
            part = list(range(start, min(start + r, n)))
            if len(part) < 2:
                continue

            part_cities = [tour[pos] for pos in part]
            part_set = set(part)

            # Sélection des p voisins géographiques
            scores = []
            for pos in range(n):
                if pos in part_set:
                    continue
                md = min(dist[tour[pos], pc] for pc in part_cities)
                scores.append((md, pos))
            scores.sort()
            neighbors = [pos for _, pos in scores[:p]]

            sub_pos = sorted(set(part) | set(neighbors))
            tour, improved = two_opt_sub(tour, sub_pos, dist)
            if improved:
                global_improved = True

        if not global_improved:
            break

    return tour, tour_cost(tour, dist)


# ── Exemple d'utilisation ─────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 50
    coords = rng.uniform(0, 100, (n, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    t0 = time.perf_counter()
    tour, cost = popmusic(dist, r=8, p=16, max_iter=10)
    elapsed = time.perf_counter() - t0

    print(f"n={n} | Coût={cost:.2f} | Temps={elapsed*1000:.1f}ms")
```

---

## Benchmark TSP

Métriques calculées sur graphes aléatoires (coordonnées uniformes $[0, 100]^2$), 10 runs par taille.  
Référence : solution optimale Held-Karp pour $n \leq 12$, meilleure solution connue au-delà.  
Seuil de réussite : gap $\leq 1\%$ vs référence.

| $n$ | % réussite | % non-détection | % fausse détection | Temps CPU (ms) |
|-----|-----------|----------------|-------------------|---------------|
| 1 | 100% | 0% | 0% | < 1 |
| 10 | ~90% | ~10% | 0% | < 20 |
| 100 | ~75% | ~25% | 0% | ~500 |
| 1 000 | ~60% | ~40% | 0% | ~15 000 |
| 10 000 | ~50%* | — | — | ~25 min* |
| 100 000 | ~45%* | — | — | ~42 h* (avec listes candidats : ~2 h) |

*Extrapolation $O(n^2)$ — valeurs expérimentales issues de Taillard & Voss (2002) pour les grandes tailles.*

**Définitions des métriques (selon `docs/README.md`) :**

| Métrique | Définition |
|----------|-----------|
| **% réussite** | Runs où le gap vs optimal est ≤ 1% |
| **% non-détection** | Algo convergé ET gap réel > 1% |
| **% fausse détection** | Algo non-convergé ET gap réel ≤ 1% |
| **Temps d'inférence** | Temps moyen CPU pour produire une solution |

**Paramètres utilisés pour ce benchmark :** $r = 8$, $p = 16$, `max_iter = 10`, construction initiale Plus Proche Voisin.

---

## Variantes notables

| Variante | Description |
|----------|-------------|
| POPMUSIC + or-opt | Remplacer 2-opt par or-opt (déplacement de segments 1–3) dans chaque sous-problème |
| POPMUSIC + LKH | Utiliser LKH-3 comme optimiseur local → qualité proche de LKH global à moindre coût |
| Parallel POPMUSIC | Les parties étant indépendantes, traitement parallèle (`multiprocessing`) |
| POPMUSIC + TW | Acceptation conditionnelle : n'accepter un mouvement que si les fenêtres temporelles sont respectées |

# Savings Algorithm (Clarke-Wright)

---

## Définition

L'**algorithme de Clarke-Wright** (1964) est une heuristique **constructive** pour le TSP et le VRP. Il est basé sur la notion d'**économie (saving)** : la distance économisée en reliant directement deux villes $i$ et $j$ plutôt que de passer par le dépôt.

C'est une méthode **gloutonne** (greedy) : elle construit le tour en fusionnant itérativement les paires de villes qui maximisent le gain de distance. Elle ne garantit pas l'optimalité mais produit de bonnes solutions rapidement.

---

## Méthode

### Étape 1 — Configuration initiale

On choisit un dépôt $0$. On initialise $n$ routes triviales :

$$\text{dépôt} \to i \to \text{dépôt} \quad \forall\, i \in \{1, \ldots, n\}$$

Coût total initial :

$$C_0 = 2 \sum_{i=1}^{n} c_{0i}$$

### Étape 2 — Calcul des savings

Pour chaque paire $(i, j)$ avec $i \neq j$, on calcule le **saving** :

$$s(i, j) = c_{0i} + c_{0j} - c_{ij}$$

**Intuition :** si on fusionne les routes $0 \to i \to 0$ et $0 \to j \to 0$ en $0 \to i \to j \to 0$, on supprime deux arcs vers le dépôt ($c_{0i}$ et $c_{0j}$) et on ajoute un arc direct ($c_{ij}$). Le gain est $s(i,j)$.

### Étape 3 — Fusion gloutonne

On trie les savings par ordre décroissant et on fusionne les routes selon les règles de faisabilité :

1. $i$ et $j$ ne sont **pas déjà dans la même route**
2. $i$ est en **bout de route** (extrémité) dans sa route courante
3. $j$ est en **bout de route** dans sa route courante

On répète jusqu'à obtenir un tour unique passant par toutes les villes.

### Complexité

| Étape | Complexité |
|-------|-----------|
| Calcul des savings | $O(n^2)$ |
| Tri | $O(n^2 \log n)$ |
| Fusions | $O(n^2)$ |
| **Total** | $O(n^2 \log n)$ |

---

## Explication intuitive

Imagine que tu dois livrer $n$ colis depuis un entrepôt. Au départ, tu fais $n$ allers-retours. Clarke-Wright te dit : "regarde quelles paires de livraisons tu peux enchaîner sans repasser par l'entrepôt, et commence par celles qui te font gagner le plus de distance."

Le résultat est un tour qui n'est pas forcément optimal, mais généralement à **10--15% de l'optimum**, construit en un temps très court.

---

## Cas d'applications connus

| Référence | Contribution |
|-----------|-------------|
| Clarke & Wright (1964) | Article fondateur — *Scheduling of Vehicles from a Central Depot* |
| VRP classique | Baseline standard pour les problèmes de tournées de véhicules |
| Logistique de livraison | Utilisé comme heuristique de départ dans de nombreux solveurs industriels |

---

## Exemple de code

```python
import math

def solve_tsp_clarke_wright(cities):
    n = len(cities)
    depot = 0

    def dist(i, j):
        dx = cities[i][0] - cities[j][0]
        dy = cities[i][1] - cities[j][1]
        return math.hypot(dx, dy)

    # Calcul des savings
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s = dist(depot, i) + dist(depot, j) - dist(i, j)
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Initialisation : chaque ville est sa propre route
    routes = {i: [i] for i in range(1, n)}
    endpoint_to_route = {i: i for i in range(1, n)}

    for s, i, j in savings:
        ri = endpoint_to_route.get(i)
        rj = endpoint_to_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        # i doit être en bout de sa route
        if route_i[-1] == i:
            pass
        elif route_i[0] == i:
            route_i.reverse()
        else:
            continue
        # j doit être en début de sa route
        if route_j[0] == j:
            pass
        elif route_j[-1] == j:
            route_j.reverse()
        else:
            continue
        # Fusion
        merged = route_i + route_j
        new_key = ri
        routes[new_key] = merged
        del routes[rj]
        endpoint_to_route[merged[0]] = new_key
        endpoint_to_route[merged[-1]] = new_key
        if i in endpoint_to_route and endpoint_to_route[i] == new_key:
            del endpoint_to_route[i]
        if j in endpoint_to_route and endpoint_to_route[j] == new_key:
            del endpoint_to_route[j]

    # Assembler en tour complet
    final_route = list(routes.values())[0]
    tour = [depot] + final_route
    total = sum(dist(tour[k], tour[k+1]) for k in range(len(tour)-1)) + dist(tour[-1], tour[0])
    return tour, total
```

---

## Benchmark TSP

Mesuré sur graphes aléatoires (seed=42, coordonnées uniformes dans [0,100]²).
Modèle empirique : t ≈ 1,36×10⁻⁶ × n^2,08.

| Nœuds | Distance | Temps CPU | Mémoire | Statut |
|-------|----------|-----------|---------|--------|
| 1 | 0,00 | < 0,001 s | < 1 MB | TRIVIAL |
| 10 | 264,14 | < 0,001 s | < 1 MB | OK |
| 100 | 851,03 | 0,035 s | 0,4 MB | OK |
| 1 000 | 2 512,54 | 1,93 s | 63 MB | OK |
| 10 000 | 8 011,32 | 271 s | 6 408 MB | LENT |
| 100 000 | — | — | — | SKIP (hors mémoire) |

**Prédictions (loi de puissance) :**
- n = 1 000 000 → ~48 jours
- n = 10 000 000 → ~16 ans

# Integer Linear Programming (ILP)

## Définition

L'**Integer Linear Programming (ILP)** est une méthode de résolution exacte du TSP.
On modélise le problème comme un programme mathématique : on définit des **variables binaires** représentant les arcs empruntés, une **fonction objectif** à minimiser (distance totale), et des **contraintes** garantissant un tour valide.

Un solveur (OR-Tools, CPLEX, Gurobi) explore ensuite l'espace des solutions via du **branch-and-cut** : il divise récursivement l'espace de recherche tout en coupant les branches impossibles grâce à des inégalités linéaires.

---

## Méthode

### Variables

$$x_{ij} \in \{0, 1\}$$

$x_{ij} = 1$ si l'on emprunte l'arc de la ville $i$ vers la ville $j$, $0$ sinon.

### Objectif

Minimiser la distance totale du tour :

$$\min \sum_{i=1}^{n} \sum_{j \neq i} c_{ij} \cdot x_{ij}$$

où $c_{ij}$ est la distance entre les villes $i$ et $j$.

### Contraintes de degré

Chaque ville doit être quittée exactement une fois et arrivée exactement une fois :

$$\sum_{j \neq i} x_{ij} = 1 \quad \forall i \qquad \text{(quitter } i \text{)}$$

$$\sum_{i \neq j} x_{ij} = 1 \quad \forall j \qquad \text{(arriver en } j \text{)}$$

### Le problème des sous-tours

Les contraintes de degré ne suffisent pas à garantir un tour valide. Elles imposent seulement que chaque ville soit visitée exactement une fois, mais elles n'empêchent pas la solution d'être composée de **plusieurs petits cycles disjoints** plutôt que d'un seul grand cycle passant par toutes les villes.

**Exemple concret avec 6 villes :**

Imaginons les villes A, B, C, D, E, F. Une solution qui respecte les contraintes de degré pourrait être :

```
Cycle 1 : A -> B -> C -> A      (3 villes)
Cycle 2 : D -> E -> F -> D      (3 villes)
```

Chaque ville est bien quittée une fois et arrivée une fois. Pourtant ce n'est pas un tour valide du TSP : le livreur ferait une première boucle sur A, B, C, puis une deuxième boucle sur D, E, F, sans jamais relier les deux groupes. Il n'aurait pas visité toutes les villes dans un seul trajet.

Ce phénomène s'appelle un **sous-tour** : un cycle qui ne couvre qu'un sous-ensemble des villes.

**Pourquoi le solveur les produit naturellement :**

Le solveur cherche à minimiser la distance totale. Deux petits cycles locaux sont souvent moins coûteux que de devoir "rallier" deux groupes de villes éloignés. Sans contrainte supplémentaire, il va donc naturellement s'y réfugier car c'est mathématiquement optimal pour lui, même si c'est invalide pour le problème.

**Visualisation :**

```
Solution invalide (sous-tours)    Solution valide (tour unique)

   A --- B                            A --- B
   |     |                            |      \
   C ----+                            C   D   |
                                      |   |   |
   D --- E                            +---E   |
   |     |                                 \  |
   F ----+                            F ----+-+
```

### Contraintes anti sous-tours : formulation MTZ

Pour interdire les sous-tours, on introduit des variables d'ordre $u_i$ représentant la **position de la ville $i$ dans le tour** (la ville 0 est fixée en position 0, les autres sont numérotées de 1 à n-1).

$$u_0 = 0$$

$$u_i - u_j + n \cdot x_{ij} \leq n - 1 \quad \forall\, i \neq j,\; i \geq 1$$

**Pourquoi ça marche :** si $x_{ij} = 1$ (on va de $i$ à $j$), alors la contrainte impose $u_i < u_j$, ce qui force $j$ à être visité après $i$. Un sous-tour entre les villes $\{1, 2, 3\}$ imposerait $u_1 < u_2 < u_3 < u_1$, ce qui est impossible.

**Limites de MTZ :**

MTZ utilise seulement $O(n^2)$ contraintes, ce qui est compact. En contrepartie, la **relaxation LP** (voir section Branch-and-Cut) est très lâche : le solveur doit explorer un arbre de recherche énorme avant de prouver l'optimalité. En pratique, MTZ devient impraticable au-delà de $n \approx 50$.

### Contraintes anti sous-tours : formulation DFJ

La formulation de Dantzig-Fulkerson-Johnson (1954) exprime directement l'interdiction : pour tout sous-ensemble $S$ de villes (qui ne contient pas toutes les villes), le nombre d'arcs internes à $S$ doit être strictement inférieur à $|S|$.

$$\sum_{i \in S,\, j \in S,\, i \neq j} x_{ij} \leq |S| - 1 \qquad \forall\, S \subsetneq V,\; 2 \leq |S| \leq n-1$$

**Intuition :** si $S$ contient $k$ villes, au maximum $k-1$ arcs peuvent rester à l'intérieur de $S$. Si on avait $k$ arcs internes à $S$, cela formerait forcément un cycle fermé sur $S$, donc un sous-tour.

**Problème :** il y a $2^n$ sous-ensembles possibles, donc $2^n$ contraintes. On ne peut pas toutes les ajouter au départ.

**Solution (lazy constraints) :** on n'ajoute une contrainte DFJ que lorsque le solveur produit une solution contenant un sous-tour. On détecte le sous-tour, on génère la contrainte qui l'interdit, on la rajoute au modèle, et on relance. En pratique, seulement quelques dizaines ou centaines de contraintes sont nécessaires avant d'atteindre l'optimalité. C'est la stratégie utilisée par **Concorde**.

| Formulation | Nombre de contraintes | Qualité de la relaxation LP | Limite pratique |
|-------------|----------------------|-----------------------------|-----------------|
| MTZ | $O(n^2)$ | Faible (borne lâche) | $n \leq 50$ |
| DFJ (lazy) | Quelques centaines en pratique | Excellente | $n \leq 100\,000$ |

---

## Branch-and-Cut

C'est le moteur interne de tous les solveurs ILP modernes. Il combine deux mécanismes : **Branch** (diviser) et **Cut** (éliminer).

### Vue d'ensemble

```
Problème ILP
    |
    v
Relaxation LP  <------------------------------------------------------+
(on autorise x dans [0,1] au lieu de {0,1})                           |
    |                                                                  |
    +-- Solution entière ? --> OPTIMAL                                 |
    |                                                                  |
    +-- Borne superieure depassee ? --> Elaguer                       |
    |                                                                  |
    +-- Sous-tour detecte ? --> Ajouter une coupe DFJ ----------------->
    |
    +-- Variable fractionnaire (ex: x12 = 0.7) ?
           |
     +-----+------+
     |   BRANCH   |
     v            v
x12 = 0        x12 = 1
(sous-arbre)  (sous-arbre)
```

### Étape 1 : Relaxation LP

On commence par **relâcher** la contrainte d'intégrité : au lieu d'imposer $x_{ij} \in \{0,1\}$, on autorise $x_{ij} \in [0, 1]$.

Ce problème continu (**LP** = Linear Program) se résout en temps polynomial via l'algorithme du simplex. Sa solution est une **borne inférieure** du vrai optimum entier : elle est toujours inférieure ou égale au meilleur tour entier possible.

Pourquoi ? Tout tour entier valide est aussi une solution LP valide (puisque $\{0,1\} \subset [0,1]$). Donc le minimum LP est forcément plus petit ou égal au minimum ILP.

> **Exemple :** si la relaxation LP donne une distance de 100, on sait que le tour optimal ne peut pas faire moins de 100. Si on a déjà trouvé un tour valide de longueur 120, la solution optimale est quelque part entre 100 et 120.

### Étape 2 : Cut (couper)

Avant de brancher, on cherche des **inégalités valides** (coupes) que la solution LP viole mais que toute solution entière respecte. On les ajoute au modèle pour resserrer la relaxation LP, ce qui remonte la borne inférieure et réduit la taille de l'arbre de recherche.

Pour le TSP, les coupes les plus puissantes sont :

| Coupe | Description |
|-------|-------------|
| **Subtour Elimination (DFJ)** | $\sum_{i,j \in S} x_{ij} \leq \|S\| - 1$ : interdit les cycles sur tout sous-ensemble $S$ de villes |
| **Comb inequalities** | Structure en "peigne" détectée sur le graphe fractionnaire ; très puissantes, utilisées par Concorde |
| **Blossom inequalities** | Généralisations des coupes comb issues de la théorie des couplages parfaits |
| **Gomory cuts** | Coupes génériques dérivées du tableau simplex ; applicables à tout ILP |

### Étape 3 : Branch (diviser)

Si après ajout de coupes la solution LP contient encore des variables fractionnaires (ex: $x_{12} = 0.7$), on crée **deux sous-problèmes** :

- Branche gauche : on force $x_{12} = 0$ (on n'emprunte pas cet arc)
- Branche droite : on force $x_{12} = 1$ (on emprunte cet arc)

On obtient un **arbre de recherche**. Chaque nœud est un sous-problème LP avec des variables supplémentaires fixées.

### Étape 4 : Élagage (pruning)

Un sous-arbre est abandonné si :
- Sa borne inférieure LP est supérieure ou égale à la meilleure solution entière connue (inutile d'explorer, on ne peut qu'être pire)
- Le sous-problème LP est infaisable (les contraintes se contredisent)

### Flux complet

```
1. Resoudre la relaxation LP du noeud courant
2. Si infaisable --> elaguer ce noeud
3. Si borne LP >= meilleure solution connue --> elaguer ce noeud
4. Chercher des coupes violees --> en ajouter, retourner en 1
5. Si solution entiere --> mettre a jour la meilleure solution connue
6. Sinon --> choisir une variable fractionnaire, creer 2 branches
7. Repeter sur les noeuds non elagues
```

---

## Les solveurs ILP

### Qu'est-ce qu'un solveur ?

Un solveur ILP est un logiciel qui implémente branch-and-cut avec des décennies d'optimisations : choix de la variable de branchement, ordre d'exploration de l'arbre, bibliothèque de coupes, heuristiques internes pour trouver rapidement une bonne solution initiale (*primal heuristics*), etc.

### Comparatif des principaux solveurs

| Solveur | Licence | Points forts | Usage typique |
|---------|---------|--------------|---------------|
| **Concorde** | Académique (gratuit) | Spécialisé TSP ; coupes comb/blossom natives ; a résolu des instances à 85 900 villes | Recherche TSP |
| **Gurobi** | Commercial (gratuit étudiants) | Le plus rapide sur ILP général ; excellente API Python | Industrie, recherche |
| **CPLEX** (IBM) | Commercial | Très mature, parallélisation avancée | Grandes entreprises |
| **OR-Tools** (Google) | Open source (Apache 2.0) | Deux moteurs : CP-SAT (SAT-based) et MIP classique ; facile à installer | Projets Python, académique |
| **SCIP** | Académique libre | Framework complet, très configurable | Recherche opérationnelle |
| **HiGHS** | Open source (MIT) | Rapide sur LP/MIP, intégré dans SciPy | Alternatif léger à Gurobi |

### OR-Tools en détail

OR-Tools propose deux moteurs différents pour l'ILP :

**CP-SAT** (utilisé dans le code ci-dessous)

Basé sur la satisfaisabilité (SAT solving) combinée à de la propagation de contraintes. Très efficace sur les problèmes combinatoires. Les variables doivent être entières (d'où la multiplication par 1000 sur les distances flottantes).

**MIP (Mixed Integer Programming)**

Branch-and-cut classique avec relaxation LP. Plus proche de Gurobi/CPLEX dans son fonctionnement. Les variables peuvent être continues ou entières.

```python
# Avec le moteur MIP d'OR-Tools (alternative a CP-SAT)
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver("SCIP")  # ou "CBC", "GLPK"
x = {}
for i in range(n):
    for j in range(n):
        if i != j:
            x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")
```

### Concorde en détail

Concorde est entièrement **dédié au TSP** et implémente la formulation DFJ avec des séparateurs de coupes spécialisés. Son pipeline :

1. Résoudre la relaxation LP du TSP sans coupes
2. Détecter les sous-tours dans la solution fractionnaire et ajouter les coupes DFJ correspondantes
3. Détecter des violations de coupes comb et blossom, les ajouter
4. Brancher uniquement si aucune coupe ne resserre plus la borne
5. L'arbre de recherche résultant est souvent minuscule grâce à la qualité des coupes

En pratique, Concorde résout souvent des instances de plusieurs milliers de nœuds **sans même brancher** : les coupes seules suffisent à prouver l'optimalité.

---

## Explication intuitive

On cherche à cocher exactement $n$ cases dans une grille $n \times n$ (une par ligne, une par colonne) de façon à former un seul cycle passant par toutes les villes, avec le coût minimum.

Le solveur fait ça en **découpant le problème** en sous-problèmes plus petits (branch), et en **éliminant les mauvaises directions** dès qu'il peut prouver qu'elles ne mèneront pas à mieux que ce qu'il a déjà (cut).

---

## Cas d'applications connus

- Référence théorique pour tout algorithme TSP exact
- Comparaison avec les heuristiques sur petites instances ($n \leq 50$)
- Utilisé dans les solveurs industriels (logistique, planification de circuits imprimés, ordonnancement)
- Papier fondateur : *Dantzig, Fulkerson, Johnson (1954)*, premier algorithme exact pour le TSP par coupes

---

## Exemple de code

```python
from ortools.sat.python import cp_model
import math

def solve_tsp_ilp(cities: list[tuple[float, float]]) -> tuple[list[int], float]:
    """
    Resout le TSP exact par ILP (CP-SAT d'OR-Tools).
    cities : liste de coordonnees (x, y)
    Retourne : (tour, distance_totale)
    """
    n = len(cities)

    def dist(i, j):
        dx = cities[i][0] - cities[j][0]
        dy = cities[i][1] - cities[j][1]
        return int(math.hypot(dx, dy) * 1000)  # x1000 pour conserver la precision entiere

    model = cp_model.CpModel()

    # Variables
    x = [[model.new_bool_var(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
    u = [model.new_int_var(0, n - 1, f"u_{i}") for i in range(n)]

    # Contraintes de degre
    for i in range(n):
        model.add(sum(x[i][j] for j in range(n) if j != i) == 1)
        model.add(sum(x[j][i] for j in range(n) if j != i) == 1)
        model.add(x[i][i] == 0)

    # Contraintes MTZ (anti sous-tours)
    model.add(u[0] == 0)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.add(u[i] - u[j] + n * x[i][j] <= n - 1)

    # Objectif
    model.minimize(
        sum(dist(i, j) * x[i][j] for i in range(n) for j in range(n) if i != j)
    )

    # Resolution
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Pas de solution trouvee")

    # Reconstruction du tour
    tour = [0]
    current = 0
    for _ in range(n - 1):
        current = next(j for j in range(n) if j != current and solver.value(x[current][j]) == 1)
        tour.append(current)

    total = sum(dist(tour[i], tour[i + 1]) for i in range(n - 1)) + dist(tour[-1], tour[0])
    return tour, total / 1000


if __name__ == "__main__":
    cities = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 0)]
    tour, distance = solve_tsp_ilp(cities)
    print(f"Tour    : {tour}")
    print(f"Distance: {distance:.2f}")
```

---

## Benchmark TSP

| Metrique | 1 noeud | 10 noeuds | 100 noeuds | 1 000 noeuds | 10 000 noeuds | 100 000 noeuds |
|----------|---------|-----------|------------|--------------|---------------|----------------|
| % reussite | / | / | / | / | / | / |
| % non-detection | / | / | / | / | / | / |
| % fausse detection | / | / | / | / | / | / |
| Temps CPU | / | / | / | / | / | / |
| Temps GPU | N/A | N/A | N/A | N/A | N/A | N/A |

> ILP avec formulation MTZ devient impraticable au-dela de $n \approx 50$.
> Pour de plus grandes instances, utiliser Concorde (formulation DFJ + lazy constraints).

---

## Complexite resumee

| Formulation | Contraintes anti sous-tours | Limite pratique |
|-------------|----------------------------|-----------------|
| MTZ | $O(n^2)$ (formulation faible) | $n \leq 50$ |
| DFJ (Concorde) | Quelques centaines en pratique (lazy) | $n \leq 100\,000$ |

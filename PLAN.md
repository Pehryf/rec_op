# Plan de livraison — Projet RO TSPTW-D

## État actuel

Notebooks existants (branche `rendu`) :

| Fichier | Algorithme | Statut chemins |
|---------|------------|----------------|
| `docs/livrable_1_modelisation.ipynb` | Modélisation formelle | ✅ |
| `heuristique/christofides.ipynb` | Christofides + or-opt TSPTW-D | ✅ |
| `meta_heuristique/lkh3_tsptwd.ipynb` | LKH-3 (ILK) TSPTW-D | ✅ |
| `meta_heuristique/sma.ipynb` | Slime Mold Algorithm TSPTW-D | ✅ corrigé |
| `meta_heuristique/popmusic.ipynb` | POPMUSIC TSPTW-D | ✅ corrigé |
| `DL_MODEL/gnn/gnn_tsptwd.ipynb` | GNN (deep learning) TSPTW-D | ✅ |

---

## Priorité 1 — ILP (bloquant pour la grille)

La grille exige *"le programme linéaire et au moins une méta-heuristique"*.

**Fichier à créer :** `exact/ilp_tsptwd.ipynb`

Contenu attendu :
- Formulation ILP (MTZ ou DFJ) avec contraintes TW et coûts dynamiques
- Solver : PuLP ou OR-Tools (CBC/GLPK)
- Benchmark sur petites instances (n ≤ 15) — comparaison avec optimal Held-Karp
- Section complexité : justification pourquoi l'ILP est exponentiel en pire cas
- Export du ratio `coût_ILP / 1-tree LB` sur chaque instance

---

## Priorité 2 — Corrections notebook LKH-3

**Fichier :** `meta_heuristique/lkh3_tsptwd.ipynb`

- [ ] Élargir `SIZES = [1, 10]` → `[10, 50, 100, 200]` (plan d'expérience représentatif)
- [ ] Ajouter calcul du ratio `coût / 1-tree LB` dans `benchmark_datasets()`
- [ ] Vérifier que `benchmark_datasets()` exporte un DataFrame avec colonne `ratio_lb`

---

## Priorité 3 — Correction notebook GNN

**Fichier :** `DL_MODEL/gnn/gnn_tsptwd.ipynb`

- [ ] Cellule `bm_run_500` (n=500) : remplacer ou commenter — trop lente sur CPU
  - Option A : remplacer par n=100 pour la démo
  - Option B : ajouter `N_REPS = 1` pour le timing et désactiver le loop de repetitions
- [ ] Ajouter calcul du ratio `makespan / 1-tree LB` dans `run_benchmark()` et `show_results()`

---

## Priorité 4 — Merge en un notebook livrable final

**Fichier à créer :** `livrable_final.ipynb`

Structure :

```
1. Modélisation (reprise de livrable_1_modelisation.ipynb)
   - Contexte ADEME
   - Formulation TSPTW-D (fenêtres glissantes multi-jours — aligner avec les implémentations)
   - Complexité NP-complète
   - Références

2. Méthodes de résolution
   - 2.1 ILP (exact, n ≤ 15)
   - 2.2 Christofides (heuristique, garantie 1.5×)
   - 2.3 LKH-3 (métaheuristique)
   - 2.4 SMA — Slime Mold Algorithm (métaheuristique)
   - 2.5 POPMUSIC (métaheuristique)
   - 2.6 GNN (apprentissage profond)

3. Datasets
   - Génération aléatoire (datasetsgenerator.ipynb)
   - Tailles : n ∈ {10, 50, 100, 200, 300, 500, 1000}

4. Plan d'expérience
   - Métriques : makespan (min), ratio coût/1-tree LB, temps d'exécution (ms)
   - Répétitions : 10 runs par instance par algo
   - Variables : taille n, tightness des fenêtres TW

5. Tableau comparatif inter-algorithmes
   - Colonnes : n, algo, makespan_mean, makespan_std, ratio_lb_mean, time_ms_mean
   - Source : résultats agrégés depuis chaque notebook (ou Excel)
   - Graphiques : boxplot par algo, courbe scalabilité, heatmap ratio/n

6. Analyse statistique
   - Statistiques descriptives (moyenne, écart-type, médiane, MAD)
   - Test de Wilcoxon entre paires d'algorithmes
   - Interprétation + pistes d'amélioration

7. Conclusion
```

**Règle de merge :** les outputs des cellules ne sont PAS copiés — le notebook livrable est re-exécuté proprement de bout en bout.

---

## Priorité 5 — Alignement modélisation / implémentation

**Fichier :** `docs/livrable_1_modelisation.ipynb`

- [ ] Section 2.4 : noter que la formulation stricte (`τ_i ≤ b_i`) est utilisée dans l'ILP
- [ ] Section 2.6 : promouvoir les "fenêtres glissantes multi-jours" comme modèle principal (pas juste une extension), car c'est ce que tous les algos heuristiques implémentent
- [ ] Ajouter un tableau de correspondance modèle / implémentation

---

## Ratio 1-tree lower bound

Le 1-tree LB est calculé dans `popmusic.ipynb` (fonction `one_tree_lb`) et dans `lkh3_tsptwd.ipynb` (Held-Karp pour n ≤ 12).

**Définition utilisée :**
```
1-tree LB = MST sur nœuds {1..n} + 2 arêtes minimum depuis le dépôt (nœud 0)
```

Le ratio `coût_algo / 1-tree_LB` mesure l'écart à l'optimale :
- Ratio = 1.0 → solution optimale (ou très proche)
- Ratio > 1.5 → solution dégradée (Christofides garantit ≤ 1.5 sur TSP symétrique)

Ce ratio doit être exporté par chaque notebook dans un format uniforme pour alimenter le tableau comparatif.

---

## Format d'export uniforme (inter-notebooks)

Chaque notebook doit produire un DataFrame avec ce schéma minimal :

```python
{
    'algo':        str,   # 'christofides', 'lkh3', 'sma', 'popmusic', 'gnn', 'ilp'
    'n':           int,   # nombre de clients
    'run':         int,   # indice du run (0..N_RUNS-1)
    'makespan':    float, # durée totale en minutes
    'ratio_lb':    float, # makespan / 1-tree LB
    'time_ms':     float, # temps d'exécution en ms
    'feasible':    bool,  # solution respecte toutes les TW
}
```

Fichier de sortie suggéré : `results/{algo}_results.csv`

---

## Checklist finale avant soutenance

- [ ] ILP implémenté et fonctionnel sur n ≤ 15
- [ ] Tous les chemins relatifs corrigés (done ✅ sma, popmusic)
- [ ] LKH-3 benchmark élargi à n ∈ {10, 50, 100, 200}
- [ ] GNN cellule n=500 gérée
- [ ] Notebook livrable final mergé et re-exécuté proprement
- [ ] Tableau comparatif inter-algos avec ratio LB
- [ ] Analyse statistique (Wilcoxon + stats descriptives)
- [ ] Alignement modélisation ↔ implémentation (TW glissantes)
- [ ] Orthographe et grammaire relues
- [ ] Présentation orale préparée (résultats, défis, planning)

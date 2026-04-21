# Plan de livraison — Projet RO TSPTW-D

## État actuel

| Fichier | Algorithme | Statut |
|---------|------------|--------|
| `docs/livrable_1_modelisation.ipynb` | Modélisation formelle | ✅ |
| `heuristique/christofides.ipynb` | Christofides + or-opt TSPTW-D | ✅ |
| `meta_heuristique/lkh3_tsptwd.ipynb` | LKH-3 (ILK) TSPTW-D | ✅ export uniforme |
| `meta_heuristique/sma.ipynb` | Slime Mold Algorithm TSPTW-D | ✅ corrigé |
| `meta_heuristique/popmusic.ipynb` | POPMUSIC TSPTW-D | ✅ corrigé |
| `DL_MODEL/gnn/gnn_tsptwd.ipynb` | GNN (deep learning) TSPTW-D | ✅ |
| `ilp_tsptwd.ipynb` | ILP exact MTZ + DFJ TSPTW-D | ✅ complet |

---

## Priorité 1 — ILP ✅ TERMINÉ

**Fichier :** `ilp_tsptwd.ipynb`

- ✅ Formulations MTZ et DFJ avec contraintes TW cycliques et coûts dynamiques (OR-Tools CP-SAT)
- ✅ Benchmark sur petites instances (n ≤ 20, skip_n=20)
- ✅ Section complexité : O(n!), Held-Karp O(n²·2ⁿ), branch-and-bound ILP exponentiel
- ✅ Export format uniforme → `results/ilp_results.csv`

---

## Priorité 2 — Corrections notebook LKH-3 ✅ TERMINÉ

**Fichier :** `meta_heuristique/lkh3_tsptwd.ipynb`

- ✅ Benchmark couvre n ∈ {10, 50, 100, 200, 500} (cellules individuelles par taille)
- ✅ `benchmark_datasets()` exporte le format uniforme `{algo, n, run, makespan, time_ms, feasible}`
- ✅ Export → `results/lkh3_results.csv`
- ℹ️ `ratio_lb` non calculé ici — centralisé dans `livrable_final.ipynb`

---

## Priorité 3 — Correction notebook GNN ✅ TERMINÉ

**Fichier :** `DL_MODEL/gnn/gnn_tsptwd.ipynb`

- ✅ Cellule `bm_run_500` remplacée par n=100 (Option A)
- ✅ `bm_run_all` : `n_reps=1`, capped `max_n=500`
- ✅ Export format uniforme `{algo, n, run, makespan, time_ms, feasible}`
- ✅ Export → `results/gnn_results.csv`
- ℹ️ `ratio_lb` calculé dans `livrable_final.ipynb`

---

## Priorité 4 — Merge en un notebook livrable final

**Fichier à créer :** `livrable_final.ipynb`

Structure :

```
1. Modélisation (reprise de livrable_1_modelisation.ipynb)
   - Contexte ADEME
   - Formulation TSPTW-D (fenêtres glissantes multi-jours)
   - Complexité NP-complète
   - Références

2. Méthodes de résolution
   - 2.1 ILP (exact, n ≤ 20)
   - 2.2 Christofides (heuristique, garantie 1.5×)
   - 2.3 LKH-3 (métaheuristique)
   - 2.4 SMA — Slime Mold Algorithm (métaheuristique)
   - 2.5 POPMUSIC (métaheuristique)
   - 2.6 GNN (apprentissage profond)

3. Datasets
   - Génération aléatoire (datasetsgenerator.ipynb)
   - Tailles : n ∈ {5, 10, 50, 100, 200, 300, 500, 1000}

4. Plan d'expérience
   - Métriques : makespan (min), ratio_lb, temps d'exécution (ms)
   - Variables : taille n, tightness des fenêtres TW

5. Calcul du ratio 1-tree LB (centralisé ici)
   - Chargement des results/{algo}_results.csv
   - one_tree_lb(dist_matrix) calculé une fois par instance
   - ratio_lb = makespan / 1-tree_LB ajouté au DataFrame consolidé
   - Note : LB basée sur distances statiques (borne sur la distance de transit pure,
     sans les attentes TW) — ratio > 1 reflète aussi les attentes inévitables

6. Tableau comparatif inter-algorithmes
   - Colonnes : n, algo, makespan_mean, makespan_std, ratio_lb_mean, time_ms_mean
   - Graphiques : boxplot par algo, courbe scalabilité, heatmap ratio/n

7. Analyse statistique
   - Statistiques descriptives (moyenne, écart-type, médiane, MAD)
   - Test de Wilcoxon entre paires d'algorithmes
   - Interprétation + pistes d'amélioration

8. Conclusion
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

**Calculé une seule fois dans `livrable_final.ipynb`** (pas dans les notebooks individuels).

**Définition :**
```
1-tree LB = MST de Prim sur nœuds {1..n-1} + 2 arêtes minimales depuis le dépôt (nœud 0)
```

Utilise les distances statiques (`dist_base`) — borne inférieure sur la distance de transit pure.

**Interprétation du ratio `makespan / 1-tree_LB` :**
- Ratio proche de 1.0 → solution quasi-optimale sur la distance de transit
- Ratio > 1.0 : peut refléter les attentes obligatoires aux fenêtres TW (inévitables, pas une dégradation)
- Ratio > 1.5 → solution potentiellement dégradée (Christofides garantit ≤ 1.5 sur TSP symétrique pur)

---

## Format d'export uniforme (inter-notebooks)

Chaque notebook produit un CSV avec ce schéma :

```python
{
    'algo':     str,   # 'christofides', 'lkh3', 'sma', 'popmusic', 'gnn', 'ilp'
    'n':        int,   # nombre de clients
    'run':      int,   # indice du run (0..N_RUNS-1)
    'makespan': float, # durée totale en minutes
    'time_ms':  float, # temps d'exécution en ms
    'feasible': bool,  # solution respecte toutes les TW
}
```

`ratio_lb` est ajouté par `livrable_final.ipynb` après chargement des CSVs.

Fichiers de sortie : `results/{algo}_results.csv`

| Notebook | Fichier CSV | Statut |
|----------|-------------|--------|
| `ilp_tsptwd.ipynb` | `results/ilp_results.csv` | ✅ |
| `lkh3_tsptwd.ipynb` | `results/lkh3_results.csv` | ✅ |
| `christofides.ipynb` | `results/christofides_results.csv` | ✅ |
| `sma.ipynb` | `results/sma_results.csv` | ✅ |
| `popmusic.ipynb` | `results/popmusic_results.csv` | ✅ |
| `gnn_tsptwd.ipynb` | `results/gnn_results.csv` | ✅ |

---

## Checklist finale avant soutenance

- [x] ILP implémenté et fonctionnel (MTZ + DFJ, n ≤ 20)
- [x] Tous les chemins relatifs corrigés (sma, popmusic)
- [x] LKH-3 benchmark couvre n ∈ {10, 50, 100, 200, 500}
- [x] LKH-3 export format uniforme
- [x] GNN cellule n=500 gérée + export uniforme
- [x] Exports uniformes : christofides, sma, popmusic
- [ ] Notebook livrable final créé et re-exécuté proprement
- [ ] Calcul ratio_lb centralisé dans livrable_final.ipynb
- [ ] Tableau comparatif inter-algos
- [ ] Analyse statistique (Wilcoxon + stats descriptives)
- [ ] Alignement modélisation ↔ implémentation (TW glissantes)
- [ ] Orthographe et grammaire relues
- [ ] Présentation orale préparée (résultats, défis, planning)

# Présentation TSPTW-D — Index des slides

## Projet : Optimisation de tournées de livraison (ADEME)
## Algorithmes comparés : ILP · Christofides · LKH-3 · SMA · POPMUSIC · GNN

---

| # | Fichier | Contenu |
|---|---------|---------|
| 1 | [01_probleme.md](01_probleme.md) | Présentation du problème TSPTW-D, contexte ADEME, complexité |
| 2 | [02_modelisation.md](02_modelisation.md) | Formalisation mathématique, variables, objectif, ILP |
| 3 | [03_description_choix.md](03_description_choix.md) | Description des 6 algorithmes choisis et justification |
| 4 | [04_instances.md](04_instances.md) | Génération et caractéristiques des instances de test |
| 5 | [05_plan_experience.md](05_plan_experience.md) | Protocole d'évaluation, métriques, hypothèses testées |
| 6 | [06_impact_parametres.md](06_impact_parametres.md) | Influence de n, des runs, des perturbations, des hyperparamètres |
| 7 | [07_resultats_statistiques.md](07_resultats_statistiques.md) | Stats descriptives, Wilcoxon, Borda, recommandations |
| 8 | [08_presentation_graphe.md](08_presentation_graphe.md) | Architecture GNN, représentation graphe, décodage TW-aware |

---

### Figures à insérer (disponibles dans `docs/`)

| Figure | Slide | Fichier |
|--------|-------|---------|
| Illustration TSPTW-D | 1 | `docs/tsptwd_illustration.png` |
| Boxplot makespans + ratio_lb | 7 | `docs/standalone_boxplot.png` |
| Courbes de scalabilité | 6 & 7 | `docs/standalone_scalabilite.png` |
| Benchmark POPMUSIC | 3 & 5 | `docs/popmusic_benchmark_graphs.png` |

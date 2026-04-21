# Documentation — Règles de contribution

Chaque branche dédiée à un modèle doit avoir sa propre entrée dans `docs/bibliography/`.
La documentation doit être écrite en LaTex and the dossier `docs/src/latex`, puis convertit en pdf et placée dans `docs/src/`.
Aucune PR ne sera mergée sans cette entrée.

## Structure attendue pour chaque modèle

- **Définition** : description claire de la méthode
- **Méthode** : formules mathématiques clés
- **Explication** : intuition sans jargon
- **Cas d'applications connus** : papiers / domaines de référence
- **Exemple de code** : snippet Python minimal et fonctionnel
- **Benchmark TSP** : tableau rempli avec les métriques ci-dessous

### Métriques benchmark

| Métrique | Définition |
|----------|-----------|
| **% réussite** | Proportion de runs où le gap avec l'optimal est ≤ 1 %, pour 1 / 10 / 100 / 1 000 / 10 000 / 100 000 nœuds |
| **% non-détection** | L'algo croit avoir convergé mais le gap réel est > 1 % |
| **% fausse détection** | L'algo arrête en pensant échouer alors que la solution est ≤ 1 % de l'optimal |
| **Temps passé par inférence** | Temps moyen pour produire une solution, exprimé en ms ou s, sur CPU et GPU si applicable |

**Les métriques suivantes doivent être calculées sur des graphs aléatoires contenant 1 / 10 / 100 / 1 000 / 10 000 / 100 000 nœuds ainsi que sur le dataset de test fourni par l'équipe.**

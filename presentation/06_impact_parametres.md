# Slide 6 — Impact des Paramètres

## Titre : Influence des paramètres sur la qualité et le temps de calcul

---

### 1. Impact de la taille de l'instance (n)

**Sur le makespan :**

| n | Christofides | LKH-3 (médiane) | POPMUSIC | GNN |
|---|---|---|---|---|
| 10 | ~200 min | ~180 min | ~220 min | ~350 min |
| 50 | ~900 min | ~750 min | ~1 100 min | ~1 800 min |
| 100 | ~2 000 min | ~1 700 min | ~2 500 min | ~4 500 min |
| 300 | ~5 500 min | ~4 500 min | ~7 000 min | ~12 000 min |
| 500 | — | ~8 000 min | ~12 000 min | ~20 000 min |

→ Le makespan croît **super-linéairement** avec n (accumulation des attentes + distances)

**Sur le temps de calcul (log-scale) :**

| n | ILP | Christofides | LKH-3 | SMA | POPMUSIC | GNN |
|---|---|---|---|---|---|---|
| 5 | ~250 ms | ~50 ms | ~100 ms | ~7 ms | ~0.1 ms | ~50 ms |
| 10 | ~500 ms | ~100 ms | ~200 ms | ~9 ms | ~0.5 ms | ~100 ms |
| 50 | >60 s | ~200 ms | ~1 000 ms | ~70 ms | ~10 ms | ~300 ms |
| 100 | ∞ | ~600 ms | ~2 000 ms | ~700 ms | ~40 ms | ~1 000 ms |
| 500 | ∞ | >30 s | ~10 000 ms | >5 min | ~800 ms | ~15 000 ms |

→ POPMUSIC est le seul à maintenir un temps **sub-seconde** jusqu'à n=500

---

### 2. Impact du nombre de runs (LKH-3, SMA)

**LKH-3 — 20 runs par instance :**
- Écart-type faible : ~5–8% du makespan médian
- Convergence rapide : les 5 premiers runs donnent déjà un bon estimateur
- Distribution proche d'une normale tronquée

**SMA — 10 runs par instance :**
- Écart-type élevé : ~20–40% du makespan médian
- Présence de **valeurs aberrantes** (solutions très dégradées)
- Distribution asymétrique à droite (longue queue)

---

### 3. Impact des perturbations dynamiques

Les perturbations affectent les coûts de transit sur certains arcs pendant des plages horaires définies :
- Sans perturbation → makespans ≈ 10–15% inférieurs (estimation)
- Avec perturbation → l'ordre optimal de visite change car les coûts varient selon l'heure de passage
- **Algorithmes les plus sensibles :** SMA (recalcul coûteux), LKH-3 (recalcul des listes de candidats)
- **Algorithmes les plus robustes :** POPMUSIC (sous-problèmes locaux), GNN (les perturbations sont encodées dans les features d'arêtes)

---

### 4. Impact de la largeur des fenêtres temporelles

- Fenêtres **étroites** (< 60 min) → forte infaisabilité, peu d'algorithmes trouvent une solution
- Fenêtres **larges** (120–240 min, utilisées ici) → faisabilité quasi garantie → focus sur le makespan
- Fenêtres **très larges** (> 600 min) → problème se rapproche du TSP pur → tous les algos convergent

---

### 5. Impact des hyperparamètres internes

**LKH-3 :**
- Taille des listes de candidats (k=5 par défaut) → compromis qualité/vitesse
- Nombre maximum d'itérations → amélioration marginale au-delà de 100 itérations

**SMA :**
- Taille de la population → plus grande = meilleure exploration mais plus lent
- Taux de croisement OX → influent sur la diversité génétique

**POPMUSIC :**
- Taille des sous-tournées r → r=5 optimal en pratique (trop grand = sous-problèmes NP-durs)
- Nombre de voisins considérés → r voisins garantit la connectivité

**GNN :**
- Dimension cachée d ∈ {64, 128, 256} → medium (d=128, L=6) offre le meilleur compromis
- Profondeur L ∈ {4, 6, 8} couches → L=6 en pratique
- Taille du dataset d'entraînement → 1 000 instances par taille suffisant

---

### Synthèse : paramètre le plus impactant

> **La taille n est le facteur dominant** sur le temps de calcul.  
> **L'algorithme choisi** est le facteur dominant sur la qualité.  
> Les perturbations dynamiques ont un effet modéré mais réel sur tous les algorithmes.

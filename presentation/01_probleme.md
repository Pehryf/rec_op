# Slide 1 — Présentation du Problème

## Titre : TSPTW-D — Voyageur de Commerce avec Fenêtres Temporelles et Perturbations Dynamiques

---

### Contexte

- Projet financé par l'**ADEME** (Agence de la transition écologique)
- Objectif industriel : **optimiser les tournées de livraison** pour réduire les émissions carbone et la consommation de carburant
- Problème réel : un livreur doit visiter n clients dans une journée en respectant des contraintes horaires et en faisant face aux aléas de la route

---

### Définition du problème

**Données d'entrée :**
- Un **dépôt** (point de départ et d'arrivée)
- **n clients** à visiter, chacun avec :
  - Une position géographique (x, y)
  - Une **fenêtre temporelle** [aᵢ, bᵢ] : créneau d'ouverture (ex. 8h–12h)
  - Un **temps de service** sᵢ (durée de la livraison : 5–15 min)
- Des **perturbations dynamiques** sur les arcs : accidents, embouteillages → multiplicateur de coût α ∈ [2×, 3.5×] sur des plages horaires

**Objectif :**
- Trouver la **permutation σ des n clients** qui minimise le **makespan** (durée totale de la tournée)

**Contraintes :**
- Chaque client est visité **exactement une fois**
- L'heure d'arrivée chez chaque client respecte sa fenêtre temporelle
- Le véhicule repart du dépôt et y revient

---

### Complexité

| Approche | Complexité |
|----------|-----------|
| Force brute (exhaustive) | O(n!) |
| Programmation dynamique (Held-Karp) | O(n² · 2ⁿ) → faisable jusqu'à n≈20 |
| Heuristiques / métaheuristiques | Polynomial (approché) |

→ **Problème NP-complet** (réduction depuis le cycle hamiltonien)  
→ Pour n ≥ 50, les méthodes exactes sont **impraticables** en conditions réelles

---

### Ce qui distingue TSPTW-D du TSP classique

1. **Fenêtres temporelles** → restriction drastique de l'espace des solutions feasibles  
2. **Coûts dynamiques** c_ij(t) → le coût d'un arc dépend de l'heure de passage  
3. **Horizon glissant** → modèle sur 1440 min (24h), extensible à plusieurs jours

---

### Illustration

> *(insérer `docs/tsptwd_illustration.png` — réseau 6 nœuds + courbe de perturbation)*

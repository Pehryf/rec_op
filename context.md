# Contexte du projet — Optimisation de tournées de livraison

## Introduction

Depuis les années 90, il y a eu une véritable prise de conscience mondiale de la nécessité de réduire la consommation d'énergie et les émissions de gaz à effet de serre. Les premiers engagements sont apparus lors de la signature du protocole de Kyoto en 1997. Mais son entrée en vigueur n'a finalement eu lieu qu'en 2005 et de nombreux scientifiques ont jugé les efforts insuffisants pour ralentir le réchauffement climatique.

Depuis, d'autres engagements plus ambitieux ont vu le jour (division par 4 des émissions d'ici 2050 pour la France, engagements de certaines grandes villes comme Paris). Mais la tâche est compliquée : les pouvoirs publics et les collectivités territoriales n'ont pas la possibilité d'obliger les entreprises et les particuliers à changer leurs habitudes. L'action se porte donc avant tout à faire évoluer les comportements. L'économie et le recyclage des matières premières, l'amélioration des modes de transport et des performances énergétiques des bâtiments doivent devenir des priorités.

---

## Sujet

### Énoncé

L'ADEME (Agence de l'Environnement et de la Maîtrise de l'Énergie) a récemment lancé un appel à manifestation d'intérêt pour promouvoir la réalisation de démonstrateurs et d'expérimentations de nouvelles solutions de mobilité pour les personnes et les marchandises adaptées à différents types de territoires.

Votre structure **CesiCDP** est déjà bien implantée dans le domaine. Aidé de nombreux partenaires, vous avez réalisé plusieurs études sur le thème de la Mobilité Multimodale Intelligente. Les nouvelles technologies de transport, plus économiques et moins polluantes, ne sont pas sans poser de nouveaux défis, notamment du point de vue de l'optimisation de la gestion des ressources. Mais ces problèmes de logistique du transport présentent un enjeu majeur pour l'avenir : leurs applications sont nombreuses (distribution du courrier, livraison de produits, traitement du réseau routier, ramassage des ordures) et leur impact sur l'environnement peut être véritablement significatif.

Vous faites partie de l'équipe mise en place par CesiCDP pour répondre à l'appel de l'ADEME. L'enjeu est d'obtenir de nouveaux marchés avec des financements très intéressants pour continuer à développer votre activité.

**Objectif :** limiter les déplacements et la consommation des véhicules lors de livraisons. Le problème algorithmique consiste à calculer sur un réseau routier une tournée permettant de relier entre elles un sous-ensemble de villes, puis de revenir à son point de départ, de manière à **minimiser la durée totale de la tournée**.

---

## Contenu de l'étude

Après répartition des tâches entre les différents acteurs du projet, CesiCDP doit fournir à l'ADEME :

1. La **modélisation** du problème
2. Une **analyse de la complexité** du problème
3. Le **code en Python** capable de :
   - Générer des instances aléatoires (à vous de choisir et justifier les paramètres : graphes complets ou non, taille des liens, etc.)
   - Résoudre le problème à l'aide d'au moins deux méthodes de résolution différentes
4. Une **étude statistique** du comportement expérimental de vos algorithmes

### Contraintes supplémentaires choisies

Il est demandé de rajouter au moins deux contraintes parmi les suivantes. **Contraintes retenues pour ce projet :**

- **Fenêtres temporelles** (*Time Windows*) : chaque ville doit être visitée dans un certain intervalle de temps. Par exemple, si une ville est disponible uniquement de 8 h à 10 h, le parcours doit respecter cette contrainte.
- **Routes dynamiques ou perturbations** : simuler des changements dynamiques dans les coûts ou la disponibilité des routes pendant la résolution.

Les autres contraintes disponibles (non retenues) :

- Coût ou restriction de passage sur certaines arêtes (travaux, routes bloquées)
- Dépendances entre visites (une livraison doit précéder une collecte)
- Utilisation de plusieurs véhicules avec sous-tournées
- Capacités du véhicule (limite pour transporter des marchandises)
- Équilibrage de la charge des véhicules

Chaque étudiant doit pouvoir assurer la responsabilité et la production d'une solution pour ce problème.

---

## Organisation

L'échéance pour le dépôt des projets auprès de l'ADEME est fixée au **1er septembre**. Feuille de route :

1. Modélisation formelle
2. Conception algorithmique et implémentation
3. Étude expérimentale
4. **Fin juin :** Point d'avancement — présentation du travail réalisé à votre équipe
5. **Fin juillet :** Dépôt du dossier à l'ADEME *(hors périmètre du bloc)*

Chacune de ces étapes aboutira à un livrable sous forme de **Notebook Jupyter** présentant à la fois la démarche et le code correspondant (storytelling privilégié). Le code doit être lisible, commenté et orienté performance. Il est fortement recommandé de suivre les recommandations [PEP](https://openclassrooms.com/fr/courses/235344-apprenez-a-programmer-en-python/235263-de-bonnes-pratiques).

---

## Livrables attendus

### Livrable 1 — Modélisation *(check, non évalué)*

**Objectif :** Modéliser le problème au travers d'un Notebook Jupyter.

**Description :**

- Présenter le problème et son contexte, le reformuler de manière formelle.
- Étudier les propriétés théoriques, notamment la complexité.
- Proposer une représentation formelle des données, du problème et de l'objectif à optimiser.
- S'appuyer sur cette représentation pour démontrer la complexité théorique du problème.
- Intégrer des références bibliographiques vers des articles ou des ouvrages scientifiques.
- Ne pas se concentrer sur la version de base du problème : y ajouter au moins une contrainte supplémentaire (plusieurs véhicules, fenêtres de temps, etc.).

> Les méthodes de résolution ne sont pas à aborder lors de ce livrable.

Le point est réalisé avec le tuteur de projet pour valider la modélisation.

---

### Livrable final *(évalué)*

**Objectif :** Présenter l'ensemble de la démarche, la réalisation technique et conclure sur les résultats obtenus.

**Partie 1 — Modélisation :**

- Reprend les éléments de modélisation formelle du Livrable 1, mis à jour.
- Décrit les méthodes de résolution choisies (détails sur les algorithmes utilisés).

**Partie 2 — Implémentation et exploitation :**

- Implémentation des algorithmes.
- Démonstration du fonctionnement sur différents cas de test.
- Étude expérimentale : plan d'expérience complet démontrant les performances, limitations et perspectives d'amélioration, justifiés par une analyse statistique détaillée.

---

### Soutenance *(évaluée)*

**Objectif :** Démontrer les capacités à présenter oralement un travail dans un contexte professionnel.

La présentation doit être orientée résultats, avec démonstration de l'exécution du code (sur des cas suffisamment petits) et présentation des résultats. Elle doit rappeler le contexte et les objectifs, mettre en valeur les réalisations, et être transparente sur les défis rencontrés, les étapes achevées, en cours et à venir (planning apprécié).

---

## Grille d'évaluation

| Note | Critère |
|------|---------|
| **A** | Programme fonctionnel avec étude théorique de complexité et étude statistique |
| **B** | Programme fonctionnel sans étude théorique et avec étude statistique |
| **C** | Programme fonctionnel sans études |
| **D** | Programme non fonctionnel |

# Accessibilité WCAG - Dashboard Prêt à Dépenser

## Objectif

Ce document présente les mesures d'accessibilité mises en œuvre dans le dashboard de scoring crédit du projet *Prêt à Dépenser*.

L'objectif est de garantir une utilisation correcte du dashboard par le plus grand nombre d'utilisateurs, conformément aux critères WCAG sélectionnés pour le projet.

---

# Critère 1.1.1 - Contenu non textuel

## Statut

✅ Conforme

## Justification

Les graphiques présents dans le dashboard sont systématiquement accompagnés d'une description textuelle.

Graphiques concernés :

- Distribution d'une variable
- Analyse bi-variée
- Importance globale du modèle
- Explication locale SHAP

Chaque graphique possède un résumé textuel permettant de comprendre l'information principale sans dépendre uniquement de la représentation visuelle.

## Résultat

✅ Les informations essentielles restent accessibles même sans interpréter le graphique.

---

# Critère 1.4.1 - Utilisation de la couleur

## Statut

✅ Conforme

## Justification

Les couleurs ne constituent jamais l'unique moyen de transmettre une information.

Exemples :

### Décision du modèle

Les décisions sont affichées sous forme textuelle :

```text
Crédit accepté
Crédit refusé
```

et non uniquement en vert ou rouge.

### Analyse SHAP

Les contributions sont distinguées :

```text
Variables qui augmentent le risque
Variables qui réduisent le risque
```

avec des titres explicites.

### Analyse bi-variée

Le client sélectionné est identifié :

- par sa couleur ;
- par sa forme spécifique ;
- par une annotation textuelle.

## Résultat

✅ Les informations restent compréhensibles sans distinction des couleurs.

---

# Critère 1.4.3 - Contraste minimum

## Statut

✅ Conforme

## Justification

Le dashboard utilise principalement :

- le thème standard Streamlit ;
- du texte sombre sur fond clair ;
- des composants natifs de Streamlit.

Aucun texte à faible contraste n'a été ajouté volontairement.

## Résultat

✅ Le niveau de contraste est jugé suffisant pour une lecture confortable.

---

# Critère 1.4.4 - Redimensionnement du texte

## Statut

✅ Conforme

## Test réalisé

Zoom navigateur :

```text
200 %
```

## Résultat observé

Le dashboard reste fonctionnel :

- les boutons demeurent accessibles ;
- les tableaux restent utilisables ;
- les filtres et menus restent sélectionnables ;
- les graphiques restent exploitables.

Des ajustements ont été réalisés pour améliorer l'affichage :

- augmentation de la hauteur des graphiques ;
- réduction de la longueur du titre principal ;
- utilisation d'un sous-titre distinct.

## Résultat

✅ Le contenu demeure exploitable avec un zoom à 200 %.

---

# Critère 2.4.2 - Titre de page

## Statut

✅ Conforme

## Justification

Le titre de page est défini dans Streamlit via :

```python
st.set_page_config(
    page_title="Dashboard scoring crédit - Prêt à Dépenser"
)
```

Le titre de l'onglet navigateur est explicite et permet d'identifier rapidement la page.

## Résultat

✅ Critère respecté.

---

# Bonnes pratiques complémentaires

## Titres explicites

Le dashboard est structuré en sections clairement identifiables :

- Profil client
- Analyse métier du client
- Score et décision
- Explication locale de la décision
- Importance globale du modèle
- Comparaison à la population
- Analyse bi-variée

✅ Conforme

---

## Terminologie métier

Les variables techniques du modèle sont traduites en vocabulaire métier.

Exemples :

```text
PAYMENT_RATE
→ Ratio annuité / crédit

EXT_SOURCE_MEAN
→ Score externe moyen

INSTAL_DPD_MEAN
→ Retard moyen de paiement historique
```

✅ Conforme

---

## Résumés textuels

Chaque visualisation importante dispose d'une explication textuelle complémentaire.

✅ Conforme

---

# Synthèse

| Critère WCAG | Statut |
|-------------|---------|
| 1.1.1 Contenu non textuel | ✅ Conforme |
| 1.4.1 Utilisation de la couleur | ✅ Conforme |
| 1.4.3 Contraste minimum | ✅ Conforme |
| 1.4.4 Redimensionnement du texte | ✅ Conforme |
| 2.4.2 Titre de page | ✅ Conforme |

---

# Conclusion

Le dashboard respecte les critères d'accessibilité WCAG retenus pour le projet.

Les principales mesures mises en œuvre sont :

- descriptions textuelles des graphiques ;
- informations non dépendantes uniquement de la couleur ;
- contrastes suffisants ;
- affichage compatible avec un zoom à 200 % ;
- titres explicites ;
- vocabulaire métier compréhensible.

L'accessibilité a été intégrée dès la conception des visualisations et de l'interface utilisateur afin de garantir une expérience cohérente pour l'ensemble des utilisateurs.
---
title: "La régression linéaire : de l'intuition au code"
title_en: "Linear Regression: from Intuition to Code"
description: "Le guide complet pour comprendre la régression linéaire — l'intuition géométrique, les mathématiques (moindres carrés et descente de gradient), les hypothèses, les métriques d'évaluation et l'implémentation en Python."
description_en: "The complete guide to understanding linear regression — geometric intuition, the mathematics (least squares and gradient descent), assumptions, evaluation metrics and a Python implementation."
date: 2026-07-30
tags: ["Machine Learning", "Python", "Statistiques", "Data Science"]
---

La **régression linéaire** est souvent le premier algorithme que l'on rencontre en machine learning. Ce n'est pas un hasard : derrière sa simplicité apparente se cachent presque tous les concepts fondamentaux du domaine — une fonction de coût, une méthode d'optimisation, des hypothèses statistiques et des métriques d'évaluation. La comprendre en profondeur, c'est se construire une base solide pour tout le reste.

Dans cet article, on part de l'intuition, on descend jusqu'aux mathématiques, puis on remonte vers le code.

## 1. L'intuition

Imaginez un nuage de points : en abscisse la surface d'un appartement, en ordonnée son prix. On devine une tendance — plus la surface augmente, plus le prix monte. La régression linéaire cherche simplement **la droite qui passe « au mieux » à travers ce nuage**.

<figure class="post-figure">
<svg viewBox="0 0 420 260" role="img" aria-label="Nuage de points et droite de régression">
  <line x1="40" y1="220" x2="400" y2="220" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
  <line x1="40" y1="220" x2="40" y2="20" stroke="currentColor" stroke-opacity="0.4" stroke-width="1.5"/>
  <line x1="55" y1="205" x2="390" y2="55" style="stroke: rgb(var(--color))" stroke-width="3" stroke-linecap="round"/>
  <g style="fill: rgb(var(--color-2))" fill-opacity="0.85">
    <circle cx="70" cy="200" r="4"/><circle cx="105" cy="185" r="4"/><circle cx="130" cy="175" r="4"/>
    <circle cx="160" cy="150" r="4"/><circle cx="185" cy="160" r="4"/><circle cx="210" cy="130" r="4"/>
    <circle cx="240" cy="120" r="4"/><circle cx="270" cy="95" r="4"/><circle cx="300" cy="105" r="4"/>
    <circle cx="330" cy="75" r="4"/><circle cx="360" cy="70" r="4"/><circle cx="380" cy="58" r="4"/>
  </g>
  <text x="220" y="248" text-anchor="middle" font-size="12" fill="currentColor" fill-opacity="0.6">Surface (m²)</text>
</svg>
<figcaption>La régression linéaire trouve la droite qui minimise l'écart avec les points observés.</figcaption>
</figure>

Mais que veut dire « au mieux » ? C'est toute la question, et c'est là que les mathématiques entrent en jeu.

## 2. Le modèle mathématique

### Régression simple

Avec une seule variable explicative $x$, le modèle s'écrit :

$$
\hat{y} = \beta_0 + \beta_1 x
$$

- $\hat{y}$ est la valeur **prédite** ;
- $\beta_0$ est l'**ordonnée à l'origine** (l'*intercept*, le prix d'un appartement de 0 m²) ;
- $\beta_1$ est la **pente** (de combien le prix augmente par m² supplémentaire).

### Régression multiple

En pratique, le prix ne dépend pas que de la surface : nombre de pièces, étage, quartier… Avec $p$ variables, on généralise :

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p = \mathbf{x}^\top \boldsymbol{\beta}
$$

où $\mathbf{x} = (1, x_1, \dots, x_p)^\top$ et $\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_p)^\top$. Cette écriture vectorielle est essentielle : elle nous permettra de tout calculer d'un coup avec de l'algèbre linéaire.

## 3. La fonction de coût

Pour chaque point $i$, l'erreur (le **résidu**) est la différence entre la vraie valeur et la prédiction : $y_i - \hat{y}_i$. On veut que ces erreurs soient globalement les plus petites possibles.

On mesure cela avec l'**erreur quadratique moyenne** (*Mean Squared Error*, MSE) :

$$
J(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

Pourquoi élever au carré plutôt que prendre la valeur absolue ?

- Le carré **pénalise fortement les grandes erreurs** (une erreur de 10 compte 100, pas 10) ;
- La fonction est **dérivable partout**, ce qui rend l'optimisation simple et élégante ;
- Elle est **convexe** : un seul minimum, pas de piège local.

> Minimiser $J(\boldsymbol{\beta})$ revient à trouver les paramètres $\boldsymbol{\beta}$ qui rendent nos prédictions les plus fidèles. C'est le cœur de l'apprentissage.

## 4. Trouver les paramètres

Il existe deux grandes façons de minimiser $J$.

### Méthode 1 — L'équation normale (solution exacte)

En écriture matricielle, avec $\mathbf{X}$ la matrice des données ($n$ lignes, $p+1$ colonnes) et $\mathbf{y}$ le vecteur des cibles, la solution optimale se calcule **directement** :

$$
\boldsymbol{\beta} = \left( \mathbf{X}^\top \mathbf{X} \right)^{-1} \mathbf{X}^\top \mathbf{y}
$$

C'est la méthode des **moindres carrés ordinaires** (*Ordinary Least Squares*, OLS). Élégante et exacte, mais elle a un coût : inverser $\mathbf{X}^\top\mathbf{X}$ est en $O(p^3)$. Au-delà de quelques dizaines de milliers de variables, ça devient impraticable.

### Méthode 2 — La descente de gradient (solution itérative)

L'idée : partir de paramètres au hasard, puis les ajuster petit à petit dans la direction qui fait **descendre** le coût. On calcule le gradient (la pente de $J$) et on avance à l'opposé :

$$
\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \alpha \, \nabla J(\boldsymbol{\beta})
$$

où $\alpha$ est le **taux d'apprentissage** (*learning rate*). Le gradient vaut :

$$
\nabla J(\boldsymbol{\beta}) = -\frac{2}{n} \mathbf{X}^\top \left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right)
$$

C'est cette approche qui passe à l'échelle et qui, généralisée, entraîne aujourd'hui les réseaux de neurones.

| | Équation normale | Descente de gradient |
|---|---|---|
| Solution | Exacte | Approchée |
| Coût | $O(p^3)$ | $O(n p)$ par itération |
| Hyperparamètre | Aucun | Taux d'apprentissage $\alpha$ |
| Passe à l'échelle | Non | Oui |

## 5. Les hypothèses à ne pas oublier

La régression linéaire n'est fiable que si certaines hypothèses sont raisonnablement respectées :

1. **Linéarité** — la relation entre $\mathbf{x}$ et $y$ est bien linéaire.
2. **Indépendance** — les observations ne sont pas corrélées entre elles.
3. **Homoscédasticité** — la variance des résidus est constante.
4. **Normalité des résidus** — utile pour les intervalles de confiance.
5. **Absence de multicolinéarité** — les variables explicatives ne sont pas redondantes.

Ignorer ces hypothèses ne fait pas planter le code, mais rend les conclusions trompeuses. **Toujours tracer les résidus.**

## 6. Évaluer le modèle

Une fois entraîné, comment savoir s'il est bon ? Voici les trois métriques incontournables :

| Métrique | Formule | Interprétation |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert y_i - \hat{y}_i \rvert$ | Erreur moyenne, dans l'unité de $y$ |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Comme MAE mais pénalise les gros écarts |
| **R²** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Part de la variance expliquée (0 à 1) |

Le **coefficient de détermination** $R^2$ répond à la question : « quelle proportion de la variation du prix mon modèle explique-t-il ? ». Un $R^2$ de $0.85$ signifie 85 % de variance expliquée.

## 7. Implémentation en Python

### À la main, avec NumPy

Rien de tel que de coder l'équation normale pour démystifier la formule :

```python
import numpy as np

# Données : X (n, p) et y (n,)
X = np.array([[50], [62], [75], [80], [95], [110]], dtype=float)
y = np.array([120, 145, 170, 180, 210, 245], dtype=float)

# Ajout de la colonne de 1 pour l'intercept β0
X_b = np.c_[np.ones((X.shape[0], 1)), X]     # (n, p+1)

# Équation normale : β = (XᵀX)⁻¹ Xᵀ y
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"Intercept β0 = {beta[0]:.2f}")
print(f"Pente     β1 = {beta[1]:.2f}")

# Prédiction pour un appartement de 100 m²
x_new = np.array([1, 100])
print(f"Prix estimé (100 m²) = {x_new @ beta:.0f} k€")
```

Et la version **descente de gradient**, pour voir l'optimisation en action :

```python
def gradient_descent(X_b, y, alpha=1e-4, n_iter=10_000):
    n, d = X_b.shape
    beta = np.zeros(d)
    for _ in range(n_iter):
        gradient = -(2 / n) * X_b.T @ (y - X_b @ beta)
        beta -= alpha * gradient
    return beta

beta_gd = gradient_descent(X_b, y)
print("β (descente de gradient) :", beta_gd.round(2))
```

### Avec scikit-learn (en production)

En pratique, on ne réinvente pas la roue. `scikit-learn` fait tout, proprement :

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"MAE  : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE : {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"R²   : {r2_score(y_test, y_pred):.3f}")
```

> **Point clé** : on évalue toujours le modèle sur des données qu'il n'a **jamais vues** (le jeu de test). Un bon score sur les données d'entraînement ne prouve rien — c'est sur le test que se juge la généralisation.

## 8. Régularisation : quand le modèle sur-apprend

Avec beaucoup de variables, la régression peut « coller » au bruit des données d'entraînement : c'est le **surapprentissage** (*overfitting*). La parade consiste à ajouter une pénalité sur la taille des coefficients :

- **Ridge (L2)** ajoute $\lambda \sum \beta_j^2$ — réduit les coefficients sans les annuler.
- **Lasso (L1)** ajoute $\lambda \sum \lvert \beta_j \rvert$ — peut mettre des coefficients exactement à zéro, faisant ainsi de la **sélection de variables**.

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
```

Le paramètre $\lambda$ (ou `alpha`) contrôle la force de la pénalité : trop faible, on sur-apprend ; trop fort, on sous-apprend. On le règle par validation croisée.

## 9. Pièges et bonnes pratiques

- **Standardisez vos variables** avant une descente de gradient ou une régularisation (sinon les échelles faussent tout).
- **Vérifiez la linéarité** : si la relation est courbe, ajoutez des termes polynomiaux ou changez de modèle.
- **Traitez les valeurs aberrantes** : le MSE y est très sensible.
- **Ne vous fiez pas au seul $R^2$** : regardez aussi les résidus et l'erreur sur le jeu de test.

## Conclusion

La régression linéaire n'est pas qu'un modèle « pour débuter ». C'est un objet complet qui condense l'essence du machine learning : formuler une hypothèse, définir un coût, l'optimiser, puis évaluer honnêtement. Maîtriser chacune de ces étapes ici, c'est se donner les clés pour aborder sereinement des modèles bien plus complexes.

La prochaine étape logique ? La **régression logistique** — la même mécanique, mais pour la classification. On en reparle dans un prochain article.

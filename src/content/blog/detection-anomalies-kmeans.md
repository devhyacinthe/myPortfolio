---
title: "Détecter des anomalies bancaires avec K-means"
title_en: "Detecting Banking Anomalies with K-means"
description: "20 000 transactions, aucune étiquette, six clusters. Comment le clustering non supervisé repère les montants extrêmes — et pourquoi il laisse passer 4 309 transactions manifestement incohérentes."
description_en: "20,000 transactions, no labels, six clusters. How unsupervised clustering catches extreme amounts — and why it misses 4,309 blatantly inconsistent transactions."
date: 2026-08-03
tags: ["Machine Learning", "Clustering", "Détection d'anomalies", "Python"]
pdf: "files/rapport-detection-anomalies-kmeans.pdf"
pdfLabel: "Rapport complet (29 pages) : méthodologie détaillée, figures, tableaux de résultats et l'intégralité du code source Python."
pdfLabel_en: "Full report (29 pages): detailed methodology, figures, result tables and the complete Python source code."
pdfMeta: "PDF · 29 pages · 0,9 Mo · Français"
---

Comment repérer une transaction suspecte quand **personne ne vous a dit à quoi ressemble une fraude** ? C'est le point de départ de ce projet : 20 000 transactions bancaires, quatre montants par ligne, et aucune étiquette. Pas de « fraude / pas fraude », rien. De l'apprentissage non supervisé, donc.

Le pari est simple : la grande majorité des transactions suivent des schémas réguliers et se regroupent naturellement en familles. Une transaction qui s'écarte fortement du cœur de son groupe ne ressemble à rien de connu — elle mérite un regard.

Cet article résume la démarche, les résultats… et surtout la limite que l'analyse a fini par révéler, qui s'est avérée plus instructive que les anomalies elles-mêmes. Le rapport complet, code source inclus, est téléchargeable en fin d'article.

> Projet réalisé avec **AREGBA-TISSOU Ulrich** dans le cadre du cours MTH2228.

## 1. Les données

Le fichier `transactions.csv` contient 20 000 lignes et 8 colonnes. Seuls les quatre montants entrent dans le clustering — les identifiants sont de simples numéros sans signification métrique, et la date n'a pas de sens dans un calcul de distance.

| Colonne | Signification | Utilisée |
|---|---|---|
| `global_id`, `sender_id`, `receiver_id` | Identifiants | Non |
| `initial_amount` | Solde initial du compte | **Oui** |
| `transfer_amount` | Montant transféré | **Oui** |
| `amount_received` | Montant réellement reçu | **Oui** |
| `final_amount` | Solde final du compte | **Oui** |
| `date` | Horodatage | Non |

Avant toute modélisation, on pose les **relations comptables attendues**. En situation normale :

- `final_amount ≈ initial_amount − transfer_amount` ;
- `amount_received = transfer_amount` (l'argent reçu égale l'argent envoyé) ;
- `transfer_amount < initial_amount` (on ne transfère pas plus qu'on ne possède) ;
- tous les montants sont positifs.

Toute violation de ces règles est une **incohérence métier**. Retenez-les : c'est autour d'elles que se joue la chute de l'histoire.

## 2. Le pipeline

Six étapes, une fonction par étape :

1. **Chargement** du CSV ;
2. **Sélection** des quatre montants ;
3. **Normalisation** via `StandardScaler` ;
4. **Clustering** K-means en $k = 6$ groupes ;
5. **Détection** des anomalies par distance au centroïde ;
6. **Visualisation** après réduction PCA.

### Pourquoi normaliser ?

Les échelles n'ont rien de comparable : le solde initial monte à 375 000 €, alors que certains transferts valent quelques euros. Or K-means raisonne en distances — sans mise à l'échelle, `initial_amount` écraserait purement et simplement les trois autres variables. Le `StandardScaler` centre (moyenne 0) et réduit (écart-type 1), plaçant les quatre montants sur un pied d'égalité.

```python
from sklearn.preprocessing import StandardScaler

def normalize_features(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)   # moyenne 0, écart-type 1
```

### Détecter par la distance

Le principe tient en trois lignes : pour chaque transaction, on mesure la distance euclidienne à **son propre** centroïde, et on déclare anormales celles qui dépassent un seuil.

```python
import numpy as np

def detect_anomalies(data, model, threshold):
    labels = model.predict(data)
    centroids = model.cluster_centers_
    # distance de chaque point à SON propre centroïde
    distances = np.linalg.norm(data - centroids[labels], axis=1)
    return np.where(distances > threshold)[0]
```

<figure class="post-figure">
<svg viewBox="0 0 420 260" role="img" aria-label="Distance au centroïde et seuil de détection">
  <circle cx="185" cy="130" r="88" fill="none" stroke="currentColor" stroke-opacity="0.35" stroke-width="1.5" stroke-dasharray="5 5"/>
  <g style="fill: rgb(var(--color-2))" fill-opacity="0.8">
    <circle cx="170" cy="118" r="4"/><circle cx="200" cy="140" r="4"/><circle cx="155" cy="150" r="4"/>
    <circle cx="210" cy="105" r="4"/><circle cx="188" cy="98" r="4"/><circle cx="145" cy="125" r="4"/>
    <circle cx="225" cy="160" r="4"/><circle cx="163" cy="175" r="4"/><circle cx="215" cy="80" r="4"/>
    <circle cx="130" cy="160" r="4"/><circle cx="240" cy="125" r="4"/><circle cx="175" cy="60" r="4"/>
  </g>
  <g fill="red" fill-opacity="0.85">
    <circle cx="330" cy="70" r="5.5"/><circle cx="80" cy="55" r="5.5"/><circle cx="345" cy="195" r="5.5"/>
  </g>
  <line x1="185" y1="130" x2="330" y2="70" style="stroke: rgb(var(--color))" stroke-width="2" stroke-dasharray="4 3"/>
  <circle cx="185" cy="130" r="6" style="fill: rgb(var(--color))"/>
  <text x="185" y="248" text-anchor="middle" font-size="12" fill="currentColor" fill-opacity="0.6">seuil = distance limite au centroïde</text>
</svg>
<figcaption>Les points au-delà du seuil (en rouge) sont déclarés anormaux. Tout se joue sur le rayon du cercle.</figcaption>
</figure>

## 3. Ce que l'exploration révèle déjà

Les données sont propres : aucune valeur manquante, aucun doublon. Mais trois constats orientent toute la suite.

**Les distributions sont très asymétriques et bimodales.** Coefficient d'asymétrie proche de 2,0, et deux populations bien distinctes : une masse de petites transactions, puis une seconde bosse de très gros montants. Cette coexistence annonce à elle seule la pertinence d'un clustering.

**Les variables sont massivement corrélées.** La corrélation entre montant transféré et montant reçu atteint **1,00**, celle entre solde initial et solde final **0,997**. Quatre colonnes, mais seulement deux dimensions d'information indépendantes — c'est pourquoi la projection PCA en 2D conservera plus tard la *totalité* de l'information (93,1 % + 6,9 % = 100 %).

**Des règles métier simples classent déjà la population.** Avant tout modèle, en appliquant les relations comptables de la section 1 :

| Catégorie (règle métier) | Transactions | Part |
|---|---|---|
| Normale | 7 354 | 36,8 % |
| Découvert (`final < 0`) | 4 355 | 21,8 % |
| Transaction fantôme (`reçu ≠ envoyé`) | 4 289 | 21,4 % |
| Sur-transfert (`transfer > initial`) | 3 904 | 19,5 % |
| Transfert négatif (incohérent) | 98 | 0,5 % |

Ce tableau servira de juge de paix : que capte réellement la détection automatique face à ces catégories connues ?

## 4. Choisir le seuil sans le choisir arbitrairement

Le seuil est le paramètre décisif : trop bas, l'analyse se noie sous les fausses alertes ; trop haut, on rate les vraies anomalies. Plutôt que de fixer une valeur à la main, on la calcule à partir de la distribution des distances. Deux méthodes s'offrent à nous.

### Méthode 1 — La règle des trois sigma

```python
def suggest_threshold(data, model, k=3.0):
    labels = model.predict(data)
    distances = np.linalg.norm(data - model.cluster_centers_[labels], axis=1)
    return distances.mean() + k * distances.std()
```

Le raisonnement est classique : pour une distribution approximativement normale, environ **99,7 %** des valeurs se situent en deçà de $\mu + 3\sigma$. Le seuil est adaptatif (il se recalcule sur les données) et ajustable via $k$.

### Méthode 2 — Le 98ᵉ percentile

```python
def suggest_threshold_percentile(data, model):
    distances = np.min(model.transform(data), axis=1)
    return np.percentile(distances, 98)
```

Ici, on ne raisonne plus en écarts-types mais **en rang** : on déclare anormales les 2 % de transactions les plus éloignées de leur centre, point final.

### Verdict

| Méthode | Anomalies | % du total |
|---|---|---|
| $\mu + 2\sigma$ (permissif) | 1 159 | 5,80 % |
| $\mu + 3\sigma$ | 442 | 2,21 % |
| **98ᵉ percentile** | **400** | **2,00 %** |
| $\mu + 4\sigma$ (strict) | 219 | 1,10 % |

Les deux méthodes automatiques donnent des volumes très proches — 442 contre 400 — ce qui rassure sur la stabilité de la détection : autour de 2 % des transactions sont atypiques quelle que soit l'approche. Mais elles ne fonctionnent pas de la même façon : **le percentile fixe le nombre d'anomalies** et laisse le seuil s'ajuster, là où $\mu + 3\sigma$ **fixe le seuil** et laisse le nombre varier.

> **Méthode retenue : le 98ᵉ percentile.** Il est non paramétrique (il ne suppose pas la normalité des distances — hypothèse clairement fausse ici vu l'asymétrie) et robuste aux valeurs extrêmes, deux propriétés mieux adaptées à nos données. Quelques distances gigantesques suffiraient à gonfler artificiellement un seuil $\mu + 3\sigma$ ; un percentile, lui, ne regarde que le rang.

## 5. Les six clusters

Un cluster n'a pas d'étiquette : son comportement se déduit des moyennes de ses caractéristiques. On les ramène en euros en inversant le pipeline (`scaler.inverse_transform` sur les centroïdes).

| Cluster | Initial | Transfert | Final | Taille | Comportement |
|---|---|---|---|---|---|
| 0 | 390 € | 369 € | 51 € | 13 429 (67 %) | Normal |
| 2 | 2 256 € | 6 431 € | −4 175 € | 3 076 (15 %) | **Découvert** |
| 1 | 247 083 € | 33 090 € | 213 993 € | 967 (5 %) | Gros volume |
| 3 | 153 776 € | 28 464 € | 125 312 € | 985 (5 %) | Gros volume |
| 4 | 213 256 € | 18 002 € | 195 254 € | 819 (4 %) | Gros volume |
| 5 | 182 312 € | 43 558 € | 138 754 € | 724 (4 %) | Gros volume actif |

**Cluster 0 — le quotidien (67 %).** De petites opérations courantes, cohérentes entre elles, avec un compte qui ne bascule pas dans le rouge. Le comportement de référence.

**Cluster 2 — le découvert (15 %).** Solde final négatif de −4 175 € en moyenne. Le mécanisme est limpide : on transfère 6 431 € avec 2 256 € en poche, soit un ratio transfert/solde de **285 %**. Ces comptes dépensent près de trois fois ce qu'ils possèdent. Comportement à risque, isolé par l'algorithme **sans qu'on le lui ait indiqué** — c'est le beau résultat de cette segmentation.

**Clusters 1, 3, 4, 5 — le gros volume (4 à 5 % chacun).** Quatre groupes de 150 000 à 247 000 € de solde initial. Attention à l'interprétation : ils ne traduisent **pas** quatre comportements distincts. K-means a simplement découpé en tranches la population continue des grosses transactions, parce qu'on lui a imposé $k = 6$ et qu'il doit bien produire six groupes. Il les trouve en subdivisant finement la zone la plus étalée de l'espace.

En synthèse : **trois régimes financiers**, pas six — le quotidien de faible montant, le découvert, et le gros volume.

## 6. Où sont les anomalies ?

Les 400 anomalies ne sont pas réparties uniformément, loin de là.

| Cluster | Anomalies | Taille | % du cluster |
|---|---|---|---|
| 0 — normales | 4 | 13 429 | 0,0 % |
| 2 — découverts | 11 | 3 076 | 0,4 % |
| 3 — gros volume | 72 | 985 | 7,3 % |
| 1 — très gros volume | 87 | 967 | 9,0 % |
| 5 — gros volume actif | 105 | 724 | 14,5 % |
| 4 — gros volume | 121 | 819 | 14,8 % |

Elles se concentrent presque exclusivement dans les clusters de gros montant : **385 des 400 cas**. Le montant initial moyen des anomalies atteint **197 246 €**, contre 35 458 € pour l'ensemble du jeu — près de six fois plus.

Le résultat est cohérent avec lui-même : plus un cluster regroupe des montants élevés et dispersés, plus il contient d'anomalies en proportion. La détection ne les distribue pas au hasard, elle suit fidèlement la dispersion de chaque groupe.

Tout va bien, donc. Sauf qu'en regardant la *nature* de ces anomalies, une surprise attend.

## 7. Le constat essentiel

Parmi les 400 anomalies détectées, **seules 7 présentent un transfert négatif, et aucune ne correspond à une transaction fantôme**.

Aucune. Alors que l'exploration en avait dénombré **4 309** dans le jeu de données — des transactions où l'argent reçu diffère de l'argent envoyé, ce qui est comptablement absurde.

Où sont-elles passées ? Presque toutes dans le cluster 0, celui des petites transactions « normales » :

| Cluster | Transactions fantômes | Transferts négatifs |
|---|---|---|
| 0 — normales | **4 309** | 93 |
| 1 à 5 — autres | 0 | 5 |

L'explication est **géométrique**. Ces transactions sont illogiques, mais elles portent sur de petits montants. Une fois les données normalisées, elles restent très proches du centre du cluster 0, où se masse l'essentiel des petites opérations. Leur distance au centroïde demeure faible, bien en deçà du seuil — **quelle que soit la méthode de seuillage retenue** — et la détection les classe comme parfaitement normales.

Autrement dit : le défaut d'une transaction fantôme ne réside pas dans **l'ampleur** de ses montants (ce que mesure la distance) mais dans la **relation** entre eux. L'égalité attendue entre l'argent envoyé et l'argent reçu est une dimension que la distance euclidienne ne capture tout simplement pas.

<figure class="post-figure">
<svg viewBox="0 0 420 240" role="img" aria-label="Une transaction fantôme de petit montant reste proche du centroïde">
  <circle cx="120" cy="120" r="62" fill="none" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.5" stroke-dasharray="5 5"/>
  <g style="fill: rgb(var(--color-2))" fill-opacity="0.55">
    <circle cx="112" cy="112" r="3.5"/><circle cx="128" cy="126" r="3.5"/><circle cx="105" cy="132" r="3.5"/>
    <circle cx="134" cy="106" r="3.5"/><circle cx="120" cy="98" r="3.5"/><circle cx="98" cy="118" r="3.5"/>
    <circle cx="140" cy="136" r="3.5"/><circle cx="110" cy="146" r="3.5"/><circle cx="136" cy="90" r="3.5"/>
  </g>
  <circle cx="126" cy="118" r="6" fill="orange" stroke="currentColor" stroke-opacity="0.5" stroke-width="1"/>
  <circle cx="120" cy="120" r="5" style="fill: rgb(var(--color))"/>
  <circle cx="300" cy="110" r="72" fill="none" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.5" stroke-dasharray="5 5"/>
  <g style="fill: rgb(var(--color-2))" fill-opacity="0.55">
    <circle cx="290" cy="95" r="3.5"/><circle cx="312" cy="120" r="3.5"/><circle cx="275" cy="128" r="3.5"/>
    <circle cx="322" cy="92" r="3.5"/><circle cx="298" cy="140" r="3.5"/><circle cx="330" cy="130" r="3.5"/>
  </g>
  <g fill="red" fill-opacity="0.85">
    <circle cx="382" cy="60" r="5"/><circle cx="228" cy="55" r="5"/><circle cx="372" cy="170" r="5"/>
  </g>
  <circle cx="300" cy="110" r="5" style="fill: rgb(var(--color))"/>
  <text x="120" y="215" text-anchor="middle" font-size="11" fill="currentColor" fill-opacity="0.65">cluster 0 — dense</text>
  <text x="120" y="231" text-anchor="middle" font-size="10" fill="orange" fill-opacity="0.9">fantôme : invisible</text>
  <text x="300" y="215" text-anchor="middle" font-size="11" fill="currentColor" fill-opacity="0.65">gros volume — étalé</text>
  <text x="300" y="231" text-anchor="middle" font-size="10" fill="red" fill-opacity="0.8">détectées</text>
</svg>
<figcaption>À gauche, la transaction incohérente (en orange) se cache dans la masse dense des petits montants. À droite, les extrêmes de magnitude sautent aux yeux.</figcaption>
</figure>

Il existe donc **deux natures d'anomalies radicalement différentes** dans ce jeu de données, et la méthode n'en voit qu'une :

- les **anomalies de magnitude** — des montants anormalement élevés : parfaitement détectées ;
- les **anomalies logiques** — transactions fantômes, transferts négatifs : totalement invisibles, car dissimulées dans la masse des petites transactions normales.

> **Le point à retenir.** La détection par distance répond à la question « *ce montant est-il inhabituellement grand ?* », mais pas à la question « *cette transaction est-elle cohérente ?* ». Les deux notions d'anomalie ne se recouvrent pas.

## 8. Pourquoi, au fond ?

Ce n'est pas un défaut de réglage — aucun choix de $k$ ni de seuil n'y changerait quoi que ce soit. La limite tient à la nature même de l'approche, pour deux raisons.

**La distance mesure l'amplitude, pas la cohérence.** La distance euclidienne au centroïde quantifie à quel point les montants d'une transaction s'écartent de la moyenne de son groupe. Elle ne sait rien des relations attendues *entre* ces montants. Une violation reste invisible tant qu'elle ne se traduit pas par un montant extrême.

**K-means présuppose des clusters sphériques.** L'algorithme attribue chaque point au centroïde le plus proche, ce qui revient à supposer des groupes grossièrement sphériques et de taille comparable. L'hypothèse convient mal à des densités aussi inégales que celles observées ici, entre un cluster 0 très dense et des clusters de gros volume très étalés.

**Conséquence pratique :** les 400 anomalies remontées ne doivent pas se lire comme « les transactions frauduleuses », mais comme « les transactions dont les montants sont exceptionnellement élevés ». C'est une information utile — les très gros montants méritent une vérification — mais partielle. Une banque qui s'y fierait seule ignorerait plus de 21 % de comportements manifestement problématiques.

## 9. Recommandations

1. **Compléter le clustering par des règles métier explicites** pour rattraper les anomalies logiques : signaler systématiquement `amount_received ≠ transfer_amount` et `transfer_amount < 0`. Deux conditions triviales à écrire, et 4 309 cas récupérés.
2. **Renforcer la surveillance des transactions de très gros montant** (clusters 1, 3, 4, 5), où se concentre l'essentiel des anomalies géométriques.
3. **Conserver le 98ᵉ percentile** comme règle de seuillage, tout en gardant la possibilité de l'ajuster (95ᵉ ou 99ᵉ) selon le niveau de vigilance souhaité.
4. **Envisager des méthodes complémentaires** — Isolation Forest, DBSCAN — qui ne présupposent pas la forme sphérique des groupes.

## Conclusion

K-means a structuré 20 000 transactions en six clusters interprétables — un comportement normal dominant, une famille de découverts, quatre profils de gros volume — et isolé 400 transactions atypiques par leur montant. C'est une grille de lecture précieuse des comportements financiers.

Mais son principal enseignement est aussi sa principale limite : **la détection par distance capte les extrêmes de magnitude et laisse passer les incohérences logiques de faible montant**. Le meilleur résultat de ce projet n'est pas la liste des 400 anomalies ; c'est d'avoir compris précisément *ce que la méthode ne peut pas voir*.

Ce qui plaide sans détour pour une approche **hybride** : la segmentation non supervisée pour découvrir les structures qu'on n'attendait pas, et des règles métier ciblées pour vérifier celles qu'on attendait.

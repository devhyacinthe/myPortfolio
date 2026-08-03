---
title: "Detecting Banking Anomalies with K-means"
description: "20,000 transactions, no labels, six clusters. How unsupervised clustering catches extreme amounts — and why it misses 4,309 blatantly inconsistent transactions."
---

How do you spot a suspicious transaction when **nobody has told you what fraud looks like**? That's the starting point of this project: 20,000 banking transactions, four amounts per row, and no labels. No "fraud / not fraud", nothing. Unsupervised learning, then.

The bet is simple: the vast majority of transactions follow regular patterns and naturally group into families. A transaction that drifts far from the core of its own group resembles nothing familiar — it deserves a second look.

This article summarises the approach, the results… and above all the limitation the analysis eventually exposed, which turned out to be more instructive than the anomalies themselves. The full report, source code included, is downloadable at the end.

> Project carried out with **AREGBA-TISSOU Ulrich** as part of the MTH2228 course.

## 1. The data

`transactions.csv` holds 20,000 rows and 8 columns. Only the four amounts feed the clustering — the identifiers are plain numbers with no metric meaning, and a date makes no sense inside a distance computation.

| Column | Meaning | Used |
|---|---|---|
| `global_id`, `sender_id`, `receiver_id` | Identifiers | No |
| `initial_amount` | Account balance before | **Yes** |
| `transfer_amount` | Amount transferred | **Yes** |
| `amount_received` | Amount actually received | **Yes** |
| `final_amount` | Account balance after | **Yes** |
| `date` | Timestamp | No |

Before any modelling, we state the **expected accounting relations**. Under normal circumstances:

- `final_amount ≈ initial_amount − transfer_amount`;
- `amount_received = transfer_amount` (money received equals money sent);
- `transfer_amount < initial_amount` (you don't transfer more than you own);
- all amounts are positive.

Any violation of these rules is a **business inconsistency**. Keep them in mind — the twist of this story hinges on them.

## 2. The pipeline

Six steps, one function each:

1. **Load** the CSV;
2. **Select** the four amounts;
3. **Normalise** with `StandardScaler`;
4. **Cluster** with K-means into $k = 6$ groups;
5. **Detect** anomalies by distance to centroid;
6. **Visualise** after PCA reduction.

### Why normalise?

The scales aren't remotely comparable: the initial balance reaches €375,000 while some transfers are worth a few euros. K-means reasons in distances — without scaling, `initial_amount` would simply crush the other three variables. `StandardScaler` centres (mean 0) and scales (standard deviation 1), putting the four amounts on equal footing.

```python
from sklearn.preprocessing import StandardScaler

def normalize_features(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)   # mean 0, std 1
```

### Detecting by distance

The principle fits in three lines: for each transaction, measure the Euclidean distance to **its own** centroid, and flag whatever exceeds a threshold.

```python
import numpy as np

def detect_anomalies(data, model, threshold):
    labels = model.predict(data)
    centroids = model.cluster_centers_
    # distance from each point to ITS OWN centroid
    distances = np.linalg.norm(data - centroids[labels], axis=1)
    return np.where(distances > threshold)[0]
```

<figure class="post-figure">
<svg viewBox="0 0 420 260" role="img" aria-label="Distance to centroid and detection threshold">
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
  <text x="185" y="248" text-anchor="middle" font-size="12" fill="currentColor" fill-opacity="0.6">threshold = maximum distance to centroid</text>
</svg>
<figcaption>Points beyond the threshold (in red) are flagged as anomalous. Everything hinges on the radius of that circle.</figcaption>
</figure>

## 3. What exploration already reveals

The data is clean: no missing values, no duplicates. But three findings shape everything that follows.

**The distributions are heavily skewed and bimodal.** Skewness close to 2.0, and two clearly distinct populations: a mass of small transactions, then a second bump of very large amounts. That coexistence alone justifies clustering.

**The variables are massively correlated.** The correlation between amount transferred and amount received reaches **1.00**, and between initial and final balance **0.997**. Four columns, but only two independent dimensions of information — which is why the 2D PCA projection will later preserve *all* the information (93.1% + 6.9% = 100%).

**Simple business rules already classify the population.** Before any model, applying the accounting relations from section 1:

| Category (business rule) | Transactions | Share |
|---|---|---|
| Normal | 7,354 | 36.8% |
| Overdraft (`final < 0`) | 4,355 | 21.8% |
| Ghost transaction (`received ≠ sent`) | 4,289 | 21.4% |
| Over-transfer (`transfer > initial`) | 3,904 | 19.5% |
| Negative transfer (inconsistent) | 98 | 0.5% |

This table is the yardstick: what does automatic detection actually catch, measured against these known categories?

## 4. Choosing the threshold without choosing arbitrarily

The threshold is the decisive parameter: too low and the analysis drowns in false alarms; too high and real anomalies slip through. Rather than picking a value by hand, we compute it from the distribution of distances. Two methods are available.

### Method 1 — The three-sigma rule

```python
def suggest_threshold(data, model, k=3.0):
    labels = model.predict(data)
    distances = np.linalg.norm(data - model.cluster_centers_[labels], axis=1)
    return distances.mean() + k * distances.std()
```

Classic reasoning: for an approximately normal distribution, roughly **99.7%** of values sit below $\mu + 3\sigma$. The threshold is adaptive (it recomputes on the data) and tunable through $k$.

### Method 2 — The 98th percentile

```python
def suggest_threshold_percentile(data, model):
    distances = np.min(model.transform(data), axis=1)
    return np.percentile(distances, 98)
```

Here we no longer reason in standard deviations but **in rank**: the 2% of transactions furthest from their centre are anomalous, full stop.

### Verdict

| Method | Anomalies | % of total |
|---|---|---|
| $\mu + 2\sigma$ (permissive) | 1,159 | 5.80% |
| $\mu + 3\sigma$ | 442 | 2.21% |
| **98th percentile** | **400** | **2.00%** |
| $\mu + 4\sigma$ (strict) | 219 | 1.10% |

Both automatic methods produce very similar volumes — 442 against 400 — which is reassuring about the stability of the detection: around 2% of transactions look atypical whichever route you take. But they don't work the same way: **the percentile fixes the number of anomalies** and lets the threshold adjust, whereas $\mu + 3\sigma$ **fixes the threshold** and lets the count vary.

> **Method retained: the 98th percentile.** It is non-parametric (it doesn't assume normality of distances — an assumption plainly false here given the skew) and robust to extreme values, two properties better suited to our data. A handful of enormous distances would be enough to artificially inflate a $\mu + 3\sigma$ threshold; a percentile only looks at rank.

## 5. The six clusters

A cluster carries no label: its behaviour is inferred from the means of its features. We bring them back to euros by inverting the pipeline (`scaler.inverse_transform` on the centroids).

| Cluster | Initial | Transfer | Final | Size | Behaviour |
|---|---|---|---|---|---|
| 0 | €390 | €369 | €51 | 13,429 (67%) | Normal |
| 2 | €2,256 | €6,431 | −€4,175 | 3,076 (15%) | **Overdraft** |
| 1 | €247,083 | €33,090 | €213,993 | 967 (5%) | High volume |
| 3 | €153,776 | €28,464 | €125,312 | 985 (5%) | High volume |
| 4 | €213,256 | €18,002 | €195,254 | 819 (4%) | High volume |
| 5 | €182,312 | €43,558 | €138,754 | 724 (4%) | Active high volume |

**Cluster 0 — everyday banking (67%).** Small routine operations, internally consistent, with an account that never goes into the red. The reference behaviour.

**Cluster 2 — overdrafts (15%).** Final balance of −€4,175 on average. The mechanism is transparent: €6,431 transferred with €2,256 in hand, a transfer-to-balance ratio of **285%**. These accounts spend nearly three times what they own. A risky behaviour, isolated by the algorithm **without ever being told about it** — the elegant result of this segmentation.

**Clusters 1, 3, 4, 5 — high volume (4–5% each).** Four groups from €150,000 to €247,000 of initial balance. Careful with the interpretation: they do **not** represent four distinct behaviours. K-means simply sliced the continuous population of large transactions into bands, because we forced $k = 6$ on it and it has to produce six groups. It finds them by finely subdividing the most spread-out region of the space.

In short: **three financial regimes**, not six — low-amount everyday banking, overdrafts, and high volume.

## 6. Where are the anomalies?

The 400 anomalies are far from uniformly distributed.

| Cluster | Anomalies | Size | % of cluster |
|---|---|---|---|
| 0 — normal | 4 | 13,429 | 0.0% |
| 2 — overdrafts | 11 | 3,076 | 0.4% |
| 3 — high volume | 72 | 985 | 7.3% |
| 1 — very high volume | 87 | 967 | 9.0% |
| 5 — active high volume | 105 | 724 | 14.5% |
| 4 — high volume | 121 | 819 | 14.8% |

They concentrate almost exclusively in the large-amount clusters: **385 of the 400 cases**. The mean initial amount among anomalies reaches **€197,246**, against €35,458 across the whole dataset — nearly six times higher.

The result is internally coherent: the higher and more dispersed a cluster's amounts, the more anomalies it holds proportionally. Detection isn't scattering them at random, it faithfully follows each group's dispersion.

All good, then. Except that looking at the *nature* of those anomalies, a surprise is waiting.

## 7. The essential finding

Among the 400 detected anomalies, **only 7 show a negative transfer, and not a single one is a ghost transaction**.

Not one. Whereas exploration had counted **4,309** of them in the dataset — transactions where the money received differs from the money sent, which is accounting nonsense.

Where did they go? Almost all into cluster 0, the one holding small "normal" transactions:

| Cluster | Ghost transactions | Negative transfers |
|---|---|---|
| 0 — normal | **4,309** | 93 |
| 1 to 5 — others | 0 | 5 |

The explanation is **geometric**. These transactions are illogical, but they involve small amounts. Once the data is normalised, they stay very close to the centre of cluster 0, where the bulk of small operations piles up. Their distance to the centroid remains low, well below the threshold — **whichever thresholding method you choose** — and detection classifies them as perfectly normal.

Put differently: the defect of a ghost transaction lies not in the **magnitude** of its amounts (what distance measures) but in the **relationship** between them. The expected equality between money sent and money received is a dimension Euclidean distance simply does not capture.

<figure class="post-figure">
<svg viewBox="0 0 420 240" role="img" aria-label="A small-amount ghost transaction stays close to the centroid">
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
  <text x="120" y="231" text-anchor="middle" font-size="10" fill="orange" fill-opacity="0.9">ghost: invisible</text>
  <text x="300" y="215" text-anchor="middle" font-size="11" fill="currentColor" fill-opacity="0.65">high volume — spread out</text>
  <text x="300" y="231" text-anchor="middle" font-size="10" fill="red" fill-opacity="0.8">detected</text>
</svg>
<figcaption>On the left, the inconsistent transaction (orange) hides inside the dense mass of small amounts. On the right, extremes of magnitude are impossible to miss.</figcaption>
</figure>

So there are **two radically different natures of anomaly** in this dataset, and the method sees only one:

- **magnitude anomalies** — abnormally large amounts: perfectly detected;
- **logical anomalies** — ghost transactions, negative transfers: entirely invisible, because they hide within the mass of small normal transactions.

> **The takeaway.** Distance-based detection answers the question "*is this amount unusually large?*", but not the question "*is this transaction consistent?*". The two notions of anomaly do not overlap.

## 8. Why, fundamentally?

This isn't a tuning problem — no choice of $k$ or threshold would change a thing. The limitation is inherent to the approach, for two reasons.

**Distance measures magnitude, not consistency.** The Euclidean distance to the centroid quantifies how far a transaction's amounts stray from its group's mean. It knows nothing of the expected relations *between* those amounts. A violation stays invisible as long as it doesn't translate into an extreme amount.

**K-means assumes spherical clusters.** The algorithm assigns each point to the nearest centroid, which amounts to assuming roughly spherical groups of comparable size. That assumption sits badly with densities as uneven as those here, between a very dense cluster 0 and very spread-out high-volume clusters.

**Practical consequence:** the 400 anomalies returned should not be read as "the fraudulent transactions", but as "the transactions with exceptionally large amounts". That's useful information — very large amounts warrant a check — but partial. A bank relying on it alone would ignore more than 21% of plainly problematic behaviour.

## 9. Recommendations

1. **Complement the clustering with explicit business rules** to recover the logical anomalies: systematically flag `amount_received ≠ transfer_amount` and `transfer_amount < 0`. Two trivial conditions to write, and 4,309 cases recovered.
2. **Tighten monitoring on very large transactions** (clusters 1, 3, 4, 5), where the bulk of geometric anomalies concentrates.
3. **Keep the 98th percentile** as the thresholding rule, while retaining the option to adjust it (95th or 99th) depending on the desired level of vigilance.
4. **Consider complementary methods** — Isolation Forest, DBSCAN — which don't presuppose spherical group shapes.

## Conclusion

K-means structured 20,000 transactions into six interpretable clusters — a dominant normal behaviour, an overdraft family, four high-volume profiles — and isolated 400 transactions made atypical by their amount. That's a valuable reading grid for financial behaviour.

But its main lesson is also its main limitation: **distance-based detection captures extremes of magnitude and lets low-amount logical inconsistencies through**. The best result of this project isn't the list of 400 anomalies; it's having understood precisely *what the method cannot see*.

Which argues plainly for a **hybrid** approach: unsupervised segmentation to discover the structures you weren't expecting, and targeted business rules to check the ones you were.

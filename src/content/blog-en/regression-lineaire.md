---
title: "Linear Regression: from Intuition to Code"
description: "The complete guide to understanding linear regression — geometric intuition, the mathematics (least squares and gradient descent), assumptions, evaluation metrics and a Python implementation."
---

**Linear regression** is often the very first algorithm you meet in machine learning. That's no accident: behind its apparent simplicity lie almost all of the field's core concepts — a cost function, an optimization method, statistical assumptions and evaluation metrics. Understanding it deeply gives you a solid foundation for everything else.

In this article, we start from intuition, go down to the mathematics, then climb back up to the code.

## 1. The intuition

Picture a cloud of points: on the x-axis the surface area of an apartment, on the y-axis its price. You can sense a trend — the larger the area, the higher the price. Linear regression simply looks for **the line that runs "best" through that cloud**.

<figure class="post-figure">
<svg viewBox="0 0 420 260" role="img" aria-label="Scatter plot and regression line">
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
<figcaption>Linear regression finds the line that minimizes the gap with the observed points.</figcaption>
</figure>

But what does "best" mean? That's the whole question — and it's where the mathematics comes in.

## 2. The mathematical model

### Simple regression

With a single explanatory variable $x$, the model reads:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

- $\hat{y}$ is the **predicted** value;
- $\beta_0$ is the **intercept** (the price of a 0 m² apartment);
- $\beta_1$ is the **slope** (how much the price rises per extra m²).

### Multiple regression

In practice, price doesn't depend only on surface area: number of rooms, floor, neighbourhood… With $p$ variables, we generalize:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p = \mathbf{x}^\top \boldsymbol{\beta}
$$

where $\mathbf{x} = (1, x_1, \dots, x_p)^\top$ and $\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_p)^\top$. This vector form is essential: it lets us compute everything at once with linear algebra.

## 3. The cost function

For each point $i$, the error (the **residual**) is the difference between the true value and the prediction: $y_i - \hat{y}_i$. We want these errors to be globally as small as possible.

We measure this with the **Mean Squared Error** (MSE):

$$
J(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

Why square the errors instead of taking the absolute value?

- Squaring **strongly penalizes large errors** (an error of 10 counts as 100, not 10);
- The function is **differentiable everywhere**, which makes optimization simple and elegant;
- It is **convex**: a single minimum, no local traps.

> Minimizing $J(\boldsymbol{\beta})$ means finding the parameters $\boldsymbol{\beta}$ that make our predictions the most faithful. This is the heart of learning.

## 4. Finding the parameters

There are two main ways to minimize $J$.

### Method 1 — The normal equation (exact solution)

In matrix form, with $\mathbf{X}$ the data matrix ($n$ rows, $p+1$ columns) and $\mathbf{y}$ the target vector, the optimal solution is computed **directly**:

$$
\boldsymbol{\beta} = \left( \mathbf{X}^\top \mathbf{X} \right)^{-1} \mathbf{X}^\top \mathbf{y}
$$

This is the **Ordinary Least Squares** (OLS) method. Elegant and exact, but it has a cost: inverting $\mathbf{X}^\top\mathbf{X}$ is $O(p^3)$. Beyond a few tens of thousands of variables, it becomes impractical.

### Method 2 — Gradient descent (iterative solution)

The idea: start from random parameters, then adjust them little by little in the direction that makes the cost **go down**. We compute the gradient (the slope of $J$) and step in the opposite direction:

$$
\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \alpha \, \nabla J(\boldsymbol{\beta})
$$

where $\alpha$ is the **learning rate**. The gradient is:

$$
\nabla J(\boldsymbol{\beta}) = -\frac{2}{n} \mathbf{X}^\top \left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right)
$$

This is the approach that scales and that, generalized, trains today's neural networks.

| | Normal equation | Gradient descent |
|---|---|---|
| Solution | Exact | Approximate |
| Cost | $O(p^3)$ | $O(n p)$ per iteration |
| Hyperparameter | None | Learning rate $\alpha$ |
| Scales | No | Yes |

## 5. The assumptions you shouldn't forget

Linear regression is only reliable if certain assumptions are reasonably met:

1. **Linearity** — the relationship between $\mathbf{x}$ and $y$ really is linear.
2. **Independence** — observations are not correlated with each other.
3. **Homoscedasticity** — the variance of the residuals is constant.
4. **Normality of residuals** — useful for confidence intervals.
5. **No multicollinearity** — the explanatory variables are not redundant.

Ignoring these assumptions won't crash your code, but it will make your conclusions misleading. **Always plot the residuals.**

## 6. Evaluating the model

Once trained, how do we know it's any good? Here are the three essential metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert y_i - \hat{y}_i \rvert$ | Average error, in the unit of $y$ |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Like MAE but penalizes large gaps |
| **R²** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Share of variance explained (0 to 1) |

The **coefficient of determination** $R^2$ answers the question: "what proportion of the variation in price does my model explain?". An $R^2$ of $0.85$ means 85% of variance explained.

## 7. Implementation in Python

### By hand, with NumPy

Nothing beats coding the normal equation to demystify the formula:

```python
import numpy as np

# Data: X (n, p) and y (n,)
X = np.array([[50], [62], [75], [80], [95], [110]], dtype=float)
y = np.array([120, 145, 170, 180, 210, 245], dtype=float)

# Add the column of 1s for the intercept β0
X_b = np.c_[np.ones((X.shape[0], 1)), X]     # (n, p+1)

# Normal equation: β = (XᵀX)⁻¹ Xᵀ y
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"Intercept β0 = {beta[0]:.2f}")
print(f"Slope     β1 = {beta[1]:.2f}")

# Prediction for a 100 m² apartment
x_new = np.array([1, 100])
print(f"Estimated price (100 m²) = {x_new @ beta:.0f} k€")
```

And the **gradient descent** version, to see the optimization in action:

```python
def gradient_descent(X_b, y, alpha=1e-4, n_iter=10_000):
    n, d = X_b.shape
    beta = np.zeros(d)
    for _ in range(n_iter):
        gradient = -(2 / n) * X_b.T @ (y - X_b @ beta)
        beta -= alpha * gradient
    return beta

beta_gd = gradient_descent(X_b, y)
print("β (gradient descent):", beta_gd.round(2))
```

### With scikit-learn (in production)

In practice, you don't reinvent the wheel. `scikit-learn` does it all, cleanly:

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

> **Key point**: always evaluate the model on data it has **never seen** (the test set). A good score on the training data proves nothing — generalization is judged on the test set.

## 8. Regularization: when the model overfits

With many variables, regression can "stick" to the noise in the training data: this is **overfitting**. The remedy is to add a penalty on the size of the coefficients:

- **Ridge (L2)** adds $\lambda \sum \beta_j^2$ — shrinks the coefficients without zeroing them.
- **Lasso (L1)** adds $\lambda \sum \lvert \beta_j \rvert$ — can set coefficients exactly to zero, performing **feature selection**.

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
```

The parameter $\lambda$ (or `alpha`) controls the strength of the penalty: too small and you overfit; too large and you underfit. It's tuned by cross-validation.

## 9. Pitfalls and best practices

- **Standardize your variables** before gradient descent or regularization (otherwise the scales throw everything off).
- **Check linearity**: if the relationship is curved, add polynomial terms or switch models.
- **Handle outliers**: MSE is very sensitive to them.
- **Don't rely on $R^2$ alone**: also look at the residuals and the test-set error.

## Conclusion

Linear regression isn't just a "beginner's" model. It's a complete object that condenses the essence of machine learning: state a hypothesis, define a cost, optimize it, then evaluate honestly. Mastering each of these steps here gives you the keys to approach far more complex models with confidence.

The natural next step? **Logistic regression** — the same machinery, but for classification. More on that in an upcoming article.

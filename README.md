# Half-Moon Challenge

Logistic regression, implemented from scratch, on data that no straight line can
separate — and a demonstration that the fix has nothing to do with changing the
model.

The classifier here is ~40 lines of NumPy: batch gradient descent on the
cross-entropy loss. It never changes. What changes is the number of columns fed
into it, and that alone takes test accuracy from 84.7% to 98.7%.

![Decision boundaries by polynomial degree](figures/halfmoon-degrees.png)

---

## The question

`make_moons` produces two interleaving crescents. A linear decision boundary is
structurally incapable of separating them, so the model plateaus well below its
ceiling no matter how long it trains.

The standard response is to reach for a more powerful model. This repo takes the
other route: keep the linear model and expand the feature table with polynomial
terms, so that a boundary which is linear *in the expanded space* is curved in the
original one.

The interesting part is not that this works. It is **where** it starts working.

---

## Result

500 samples, `noise=0.20`, 70/30 stratified split, `random_state=42`.

| Degree | Columns | Train acc. | Test acc. |
|---|---|---|---|
| 1 | 2 | 86.3% | 84.7% |
| 2 | 5 | 85.7% | 84.7% |
| 3 | 9 | 96.9% | **98.7%** |
| 4 | 14 | 98.3% | 98.7% |
| 5 | 20 | 98.3% | 98.0% |
| 8 | 44 | 98.3% | 98.0% |

### Degree 2 buys nothing

Identical test accuracy to degree 1, for three extra columns. The fitted boundary
bends into a shallow parabola and stops there.

The reason is geometric, not statistical. A degree-2 boundary is a conic, and a
conic changes direction exactly once. These crescents require the boundary to turn
**twice** — down between the upper moon's tail and the lower moon's body, then back
up. No amount of data or training fixes an expressiveness gap.

### Degree 3 jumps 14 points

Cubics are the lowest degree admitting an inflection point, and the boundary
immediately picks up the S-shape that threads the gap between the crescents.

The general rule: **a degree-`k` polynomial boundary can change direction at most
`k−1` times.** Counting the turns the data requires gives a principled starting
degree instead of a blind search. Here the count is two, which points at degree 3
— and degree 3 is what the experiment selects.

### Past degree 3, returns go negative

Degree 8 uses 44 columns, nearly five times degree 3, and scores *lower* on the
held-out set. Train accuracy stops moving after degree 4, so the extra parameters
are fitting the 20% label noise. This is the textbook case for L1 regularisation,
which would drive the redundant high-order coefficients to zero on its own.

---

## Background: what logistic regression actually is

### It is not "linear regression squashed into 0–1"

The modelling assumption is that the **log-odds** of the positive class are linear
in the features:

```
log( p / (1 − p) )  =  θ₀ + θ₁x₁ + θ₂x₂ + …
```

Solving that equation for `p` yields the sigmoid:

```
p = σ(z) = 1 / (1 + e^(−z))
```

So the sigmoid is a *consequence* of the assumption, not an aesthetic choice made
to keep outputs in range. Getting this ordering right explains everything else.

### Why the log-odds and not the probability

`p` is trapped in `[0, 1]` while `θ·x` ranges over the whole real line, so they
cannot be equated directly. The odds `p/(1−p)` unbounds the top but still floors at
zero. Taking the logarithm unbounds both ends and centres the scale at zero:

| p | odds | logit |
|---|---|---|
| 0.10 | 0.111 | −2.197 |
| 0.25 | 0.333 | −1.099 |
| **0.50** | **1** | **0** |
| 0.75 | 3 | +1.099 |
| 0.90 | 9 | +2.197 |

The transform is antisymmetric — `logit(1−p) = −logit(p)` — so swapping which class
is "positive" only flips the sign of every coefficient.

### The pipeline

```
x ──linear──▶ z = θ·x ──sigmoid──▶ p ──Bernoulli──▶ y ∈ {0,1}
```

The third stage is the generative assumption: the observed label is a draw from a
Bernoulli distribution parameterised by `p`. That is what makes the likelihood
well defined and gives the loss function below.

### Training

Parameters are fit by **maximum likelihood**, equivalently by minimising
cross-entropy:

```
L = −Σᵢ [ yᵢ·log(pᵢ) + (1−yᵢ)·log(1−pᵢ) ]
```

Squared error is not used, for two reasons:

1. **Convexity.** Cross-entropy paired with a sigmoid is convex — a single global
   minimum, so gradient descent cannot get stuck. Squared error paired with a
   sigmoid is not.
2. **Gradient behaviour.** The `p(1−p)` factors cancel (see the derivation under
   *The implementation*). With squared error that factor survives, and it vanishes
   exactly when the model is confidently wrong — learning stalls at the worst
   possible moment.

There is no closed-form solution, unlike ordinary least squares. Fitting is always
iterative.

### Reading the coefficients

Because log-odds add, odds multiply:

```
odds = e^θ₀ · e^(θ₁x₁) · e^(θ₂x₂) · …
```

Each `e^θⱼ` is therefore an **odds ratio**: the multiplicative effect on the odds
of a one-unit increase in `xⱼ`. A coefficient of 0.69 means `e^0.69 ≈ 2` — that
feature doubles the odds.

This is why logistic regression persists in medicine, credit scoring, and
insurance, where a model's reasoning must be auditable by a domain expert, not just
accurate.

*(In this repo the coefficients are not interpreted, because polynomial columns
like `x₁²x₂` have no physical meaning. Interpretability is the price paid for the
non-linear boundary — a trade-off worth being explicit about.)*

### Assumptions and failure modes

| Assumption | What breaks it | Remedy |
|---|---|---|
| Log-odds linear in features | Curved decision boundary | Polynomial features, splines — *this repo* |
| Independent observations | Repeated measures, time series | Mixed models, GEE |
| No perfect separation | Classes fully separable by a feature | Regularisation |
| Low multicollinearity | Correlated predictors | Check VIF, drop or regularise |

**Perfect separation** deserves the warning. On separable data the likelihood has
no finite maximum: coefficients diverge toward infinity and any reported odds ratio
is an artefact of that divergence rather than a finding. It is the reason
scikit-learn enables L2 regularisation by default, and the reason switching it off
without thinking is a mistake.

---

## The implementation

```python
def fit(self, X, y):
    n, d = X.shape
    self.w, self.b = np.zeros(d), 0.0
    for _ in range(self.n_iters):
        p = self._sigmoid(X @ self.w + self.b)
        err = p - y                                  # the entire gradient
        self.w -= self.lr * (X.T @ err / n + self.lambda_ * self.w)
        self.b -= self.lr * err.mean()
```

Two details worth pointing at.

**The gradient is one line.** Applying the chain rule through
`z → sigmoid → cross-entropy`, the `p(1−p)` factor from the sigmoid derivative
cancels against the `1/(p(1−p))` from the loss derivative, leaving

```
∂L/∂θⱼ = (p̂ − y) · xⱼ
```

Prediction error times the feature — the same form as linear regression, despite a
different loss and a different activation. If your implementation still carries a
`p * (1 - p)` term, it is minimising squared error, not cross-entropy.

**The sigmoid is branch-stable.** `1/(1+np.exp(-z))` overflows for strongly
negative `z`. Splitting on the sign avoids it:

```python
out[pos] = 1 / (1 + np.exp(-z[pos]))
ez = np.exp(z[neg]); out[neg] = ez / (1 + ez)
```

---

## Run it

```bash
pip install numpy scikit-learn matplotlib
python halfmoon_challenge.py
```

Prints the accuracy table and writes the boundary plots to `figures/`. No dataset
download — `make_moons` is generated on the fly.

---

## Caveats

- **Test accuracy exceeds train accuracy at degree 3** (98.7% vs 96.9%). This is
  sampling noise on a 150-point test split, not a real effect; changing the seed
  reverses it. A rigorous version would average over several seeds and report a
  standard deviation.
- **Scaling turned out to be optional here.** Re-running degree 3 without
  `StandardScaler` gave 98.0% — the moons span roughly `[-1.5, 2.5]`, so cubing
  them produces nothing large enough to destabilise gradient descent. Scaling
  becomes mandatory when raw features differ by orders of magnitude, or whenever
  regularisation is on.
- Accuracy is a fair metric here only because the classes are balanced 50/50. On
  imbalanced data it would be misleading and the threshold would need choosing
  deliberately.

---

## License

MIT

# Credit Derivatives & CPDO Simulator

A Python-based framework for:
- Simulating correlated Credit Default Swap (CDS) spread paths via copulas  
- Fitting univariate time‐series (ARIMA‐GARCH) to each CDS  
- Generating multi‐name scenarios with a skewed‐\(t\), Student‐\(t\) and compared with Gaussian copula  
- Evaluating dependence properties and stress‐testing joint extreme events  
- Modeling a Constant Proportion Debt Obligation (CPDO) following Dorn (2010)

---

## Installation & Requirements

1. **Python 3.8+**
2. **Dependencies** (install via `pip install -r requirements.txt`)

   * `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
   * `statsmodels`, `arch`, `statsforecast`, `sklarpy`
   * `tqdm`, `statsmodels`, `scipy`

---

## Usage

```bash
python driver.py
```

The `driver.py` script loads historical CDS data, fits each name’s ARIMA‐GARCH model, constructs a copula‐based simulator, generates paths, and finally runs the CPDO calculator.

---

## File Structure

```
.
├── basket.py           # CDSSimulator: copula sampling & path generation
├── cdo.py              # CPDO: Constant Proportion Debt Obligation simulator
├── cds_wrapper.py      # SingleCDS: data loading, ARIMA‐GARCH fitting & diagnostics
├── GJRGARCHMarket.py   # GJR‐GARCH(1,1) with market residual lags
└── driver.py           # Orchestrates end‐to‐end workflow & plots
```
## Key Dynamics
1. **ARIMA($p,d,q$) Mean Equation**
Let $r_t$ be the log‐return (or spread change). Then

$$
\phi(B)\,(1 - B)^d\,r_t
= \theta(B)\,\eta_t,
$$

where

* $B$ is the backshift operator,
* $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$,
* $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$,
* $\eta_t$ are the innovations.

2. **Student-$t$ Copula Shock**

$$
z\sim t_ν,\quad L\,z,\quad
\varepsilon = Lz \sqrt{\frac{ν}{χ^2_ν}}
$$

3. **GJR-GARCH w/ Market Lags**

$$
\sigma_t^2 = \omega + β\sigma^2_{t-1} + αε^2_{t-1} + γ\,ε^2_{t-1}\mathbf{1}\{ε_{t-1}>0\} + α_mε^2_{m,t-1} + γ_mε^2_{m,t-1}\mathbf{1}\{ε_{m,t-1}>0\}.
$$

4. **CPDO Capital Update**

   $$
     C_{t+1} = C_t + G\,(s_t - \bar s)\,Δt
               + rC_t\,Δt - c\,Δt
               - (1-R)\mathbf{1}\{\text{default}_t\}.
   $$

---

## Future Work / TODO

1. **Performance**

   * Vectorize Monte Carlo loops in `CDSSimulator`.
   * Parallelize across CPU cores (e.g. via `joblib`).

2. **Model Extensions**

   * Support other copulas (Clayton, Gumbel).
   * Add alternative volatility processes (EGARCH, FIGARCH).

3. **Analytics**

   * Implement `CopulaDependenceEvaluator` fully.
   * Add joint‐extreme stress‐testing module.

4. **CPDO Enhancements**

   * Multi‐tranche CPDO.
   * Include dynamic gearing `G(t)`.
   * Incorporate funding cost and liquidity spread.

5.**Visualization & Reporting**

6. Use **probability integral transforms (PITs)** column‐wise via empirical CDF.

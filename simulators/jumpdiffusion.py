import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsforecast import StatsForecast
from statsforecast.models import AutoRegressive, ARIMA, ARCH, GARCH
from scipy.stats import zscore
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MOUJumpDiffusion:
    def __init__(self, name, cds_series):
        # CDS Data
        print(f'Modeling CDS: {name} 5Y Mod-Mod Restructuring (Protocol 2014)....')
        self.name = name
        self.history = pd.read_csv(cds_series)
        self.history = self.history[["DATE", "Mid Spread", "Recovery Rate", "Benchmark Rate"]]
        self.history["DATE"] = pd.to_datetime(self.history["DATE"])
        self.history.sort_values("DATE", inplace=True)

        self.history_train, self.history_test = np.array_split(self.history, 2)
        self.recovery_rate = self.history_test["Recovery Rate"].mean()
        self.prediction_horizon = len(self.history_test)

        # Time series parameters
        self.sigma_residuals = None  # ARCH(1) residual volatility
        self.residuals = None

        # OU parameters (optional, not used at the moment)
        self.theta = None  # long term mean
        self.kappa = None  # mean reversion term

        # Jump parameters
        self.jump_prob = None
        self.jump_mean = None
        self.jump_std = None

        # Model selection parameters
        self.model_candidates = {}

        # Time series simulations
        self.evaluate_models()
        self.best_model = self.get_best_model()

        # Jump parameters
        self.jump_prob = 0.0
        self.jump_mean = 0.0
        self.jump_std = 0.0
        self.set_jump_params()

        # Misc
        self.sf = None

    # Calculate evaluation metrics
    def get_model_eval(self, actual, forecast, result, model):
        mae = mean_absolute_error(actual['y'], forecast[model])
        mse = mean_squared_error(actual['y'], forecast[model])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual['y'] - forecast[model]) / actual['y'])) * 100

        resid = result.get('residuals')
        self.residuals = resid
        self.sigma_residuals = np.std(resid, ddof=1)

        # Ljung-Box
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        print(f"[{self.name} - {model} model] Ljung-Box (lag=10):")
        print(lb)

        # ACF plot
        # fig, ax = plt.subplots(figsize=(6, 4))
        # plot_acf(resid, lags=20, ax=ax)
        # ax.set_title(f"ACF of residuals: {self.name}")
        # plt.tight_layout()
        # plt.show()

        return {'backtesting_scores': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape},
                'predictions': forecast[model],
                'residuals': zscore(result.get("residuals")),
                'coefficients': result.get("coef"),
                "aic": result.get('aic'),
                "sigma2": result.get('sigma2'),
                "var_coef": result.get('var_coef')}

    # TODO: Better Jumps needed
    def set_jump_params(self):
        self.sigma_residuals = np.std(self.best_model['residuals'], ddof=1)
        thresh = 3 * self.sigma_residuals
        jump_resid = self.best_model['residuals'][np.abs(self.best_model['residuals']) > thresh]
        n = len(self.best_model['residuals'])
        
        if len(jump_resid) > 0:
            # the size of postive and negative jump will be drawn from two different exponentail distribution, and that is why
            # in the paper it called Double Exponentail Distribution
            jump_up = jump_resid[jump_resid > 0]
            jump_down = -jump_resid[jump_resid < 0]
            self.jump_prob = len(jump_resid) / n
            # we decided the porpotion of positive and negative jump base on past data
            self.p_up = len(jump_up) / len(jump_resid) if len(jump_resid) > 0 else 0.5
            self.eta_up = 1.0 / np.mean(jump_up) if len(jump_up) > 0 else 1.0
            self.eta_down = 1.0 / np.mean(jump_down) if len(jump_down) > 0 else 1.0

    # New function, instead of random drawing number, we put the past data into consideration.
    def sample_jump(self):
        if np.random.rand() < self.p_up:
            return np.random.exponential(scale=1 / self.eta_up)
        else:
            return -np.random.exponential(scale=1 / self.eta_down)



    # TODO: Figure out how to eval and pick the best model
    def get_best_model(self):
        return list(self.model_candidates.values())[0]

    def evaluate_models(self):
        X_train = self.history_train[["DATE", "Mid Spread"]]
        X_test = self.history_test[["DATE", "Mid Spread"]]

        X_train.rename(columns={"DATE": "ds", "Mid Spread": "y"}, inplace=True)
        X_test.rename(columns={"DATE": "ds", "Mid Spread": "y"}, inplace=True)

        X_train["unique_id"] = 1
        X_test["unique_id"] = 1

        # mean‐based models
        mean_models = [
            AutoRegressive(lags=1, include_mean=True), # AR(1)
            # AutoRegressive(lags=2, include_mean=True), # AR(2)
            # ARIMA(order=(1, 0, 1)), # ARIMA(1,1)
            # ARIMA(order=(1, 1, 1)), #ARIMA(1,1,1)
        ]

        # volatility‐based models
        vol_models = [
            # ARCH(1), #ARCH 1
            # ARCH(2), #ARCH(2)
            # GARCH(1, 1), #GARCH(1,1)
            # GARCH(2, 1), #GARCH(2,1)
            # GARCH(1, 2) #'GARCH(1,2)
        ]

        all_models = mean_models + vol_models

        for mdl in all_models:
            sf = StatsForecast(models=[mdl], freq='C', n_jobs=-1)

            # learn on the first half
            sf.fit(df=X_train)
            result = sf.fitted_[0,0].model_
            # forecast
            fc = sf.forecast(df=X_train, h=self.prediction_horizon, fitted=True)
            model_name = list(fc.columns)[-1]
            # store
            self.model_candidates[model_name] = self.get_model_eval(X_test, fc, result, model_name)
            
            


    def test_sign_bias(self, verbose=True):
        """
        Perform the Engle-Ng Sign Bias Test on the standardized residuals.
        """
        if self.best_model.get("residuals") is None:
            raise ValueError("No residuals found in best_model.")
    
        residuals = self.best_model["residuals"]
    
        # variables
        e = np.array(residuals)
        e_lag = e[:-1]
        e_now = e[1:]
        z2_now = e_now ** 2
    
        D_neg = (e_lag < 0).astype(int)
        D_pos = (e_lag >= 0).astype(int)
    
        X = np.column_stack([
            D_neg,            # c1: sign bias
            D_neg * e_lag,    # c2: negative size bias
            D_pos * e_lag     # c3: positive size bias
        ])
        X = sm.add_constant(X)
        
        model = sm.OLS(z2_now, X).fit()
    
        if verbose:
            print(f"\n[Sign Bias Test – {self.name}]")
            print(model.summary())
    
            # Joint F-test
            print("\n[Joint F-test for c1 = c2 = c3 = 0]")
            joint_test = model.f_test("x1 = x2 = x3 = 0")
            print(joint_test)
    
        return model
        """
        === Interpretation Guide for Sign Bias Test ===
        
        This test evaluates whether your model residuals still exhibit volatility asymmetry.
        Ideally the simulation of innovation(copula for diffusion  and exponentail distribution for jump) will make the data relaistic,
        which is asymmetric, so the residual will be symmetric.
        
        We run the following regression:
            εₜ² = c₀ + c₁·I(εₜ₋₁ < 0) + c₂·I(εₜ₋₁ < 0)·εₜ₋₁ + c₃·I(εₜ₋₁ ≥ 0)·εₜ₋₁ + uₜ
        
        Variables:
        - εₜ² : Squared residual (proxy for volatility)
        - I(·) : Indicator function
        - εₜ₋₁ : Lagged residual
        
        In the OLS output:
        - `coef`: The size of the effect
        - `P>|t|`: p-value of each coefficient
            → Small p-value (< 0.05) means the effect is statistically significant
        
        Key interpretations:
        --------------------------------------------
        1. **c₁ (Sign Bias)**  
           → Significant: Negative shocks increase volatility (your jump model may be missing this asymmetry)
        
        2. **c₂ (Negative Size Bias)**  
           → Significant: Larger negative shocks increase volatility even more (η⁻ may be too low or λ underestimated)
        
        3. **c₃ (Positive Size Bias)**  
           → Significant: Positive shocks also increase volatility (check if p_up or η⁺ is too aggressive)
        
        4. **Joint F-test: c₁ = c₂ = c₃ = 0**  
           → Significant: Your model fails to fully absorb volatility asymmetry
        
        How to respond:
        --------------------------------------------
        All p-values > 0.1 → Model likely captures asymmetry well (ideal case)
        
        Only c₂ significant → Consider adjusting jump threshold or η⁻
        
        All significant → Consider recalibrating η⁺, η⁻, p_up, or jump detection rule
        
        If bias persists → Consider using volatility-adjusted residuals (e.g., from a GARCH model)
        
        """
        



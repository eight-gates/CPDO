from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
from arch import arch_model
from arch.univariate.distribution import Distribution
import statsmodels.api as sm
from simulators.GJRGARCHMarket import GJRGARCHMarket
import matplotlib.pyplot as plt
from scipy.stats import norm

from statsforecast.models import ARIMA
# import warnings
# warnings.filterwarnings("ignore")


@dataclass
class SingleCDS:
    """
    Encapsulates all logic for a single CDS: data loading, preprocessing, modeling, diagnostics, and simulation.
    """
    # Characteristics
    name: str
    path: str
    group_id: int
    market_path: str

    # Model state
    S0: float = np.nan
    recovery_rate: float = np.nan
    risk_free: pd.DataFrame | None = None
    mean_preds: pd.DataFrame | None = None
    vol_preds: np.ndarray | None = None
    mean_residuals: np.ndarray | None = None
    vol_residuals: np.ndarray | None = None
    vol_distribution: Distribution | None = field(init=False, default=None)
    vol_params: np.ndarray | None = field(init=False, default=None)
    jump_prob: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    validate_df: pd.DataFrame | None = None
    horizon: int = 0
    p_up: float = 0.5
    eta_up: float = 1.0
    eta_down: float = 1.0
    sigma_residuals: float = 0.0

    def __post_init__(self):
        self._load_and_preprocess_data()
        self._fit_mean_and_vol_models()
        self.set_jump_params()
        self.test_sign_bias(verbose=False)

    def _load_and_preprocess_data(self):
        """Load and preprocess CDS and market data."""
        df_cds = pd.read_csv(self.path, parse_dates=["DATE"]).sort_values("DATE")
        df_mkt = pd.read_csv(self.market_path, parse_dates=["DATE"]).sort_values("DATE").rename(columns={"Mid Spread": "mkt_spread"})
        df_mkt = df_mkt[["DATE", "mkt_spread"]]
        df_all = pd.merge(df_cds, df_mkt, on="DATE", how="left")
        self.trainer_df, self.validate_df = np.array_split(df_all, 2)
        self.risk_free = df_all[["DATE", "Benchmark Rate"]]
        self.recovery_rate = df_all[["Recovery Rate"]].mean().values[0]
        df_all['Mid Spread'] = np.log(df_all["Mid Spread"])
        # df_all['mkt_spread'] = np.log(df_all["mkt_spread"])
        df_all["CDS Spread ldiff"] = np.nan
        df_all["Market Spread ldiff"] = np.nan
        df_all.iloc[1:, df_all.columns.get_loc('CDS Spread ldiff')] = np.diff(df_all["Mid Spread"])
        df_all.iloc[1:, df_all.columns.get_loc('Market Spread ldiff')] = np.diff(df_all["mkt_spread"])
        df_all.dropna(subset=["CDS Spread ldiff", "Market Spread ldiff"], inplace=True)
        df_all = df_all[["DATE", "CDS Spread ldiff", "Market Spread ldiff"]]
        # df_all = df_all[["DATE", "Mid Spread", "mkt_spread"]]
        df_mkt_res = df_all[["DATE", "Market Spread ldiff"]]
        # df_mkt_res = df_all[["DATE", "mkt_spread"]]
        df_all.rename(columns={"CDS Spread ldiff": "y", "DATE": "ds", "Market Spread ldiff": "mkt_spread"}, inplace=True)
        # df_all.rename(columns={"Mid Spread": "y", "DATE": "ds"},
        #               inplace=True)
        df_mkt_res = df_mkt_res.rename(columns={"Market Spread ldiff": "y", "DATE": "ds"})
        # df_mkt_res = df_mkt_res.rename(columns={"mkt_spread": "y", "DATE": "ds"})
        df_all.dropna(subset=["y", "mkt_spread"], inplace=True)
        df_mkt_res.dropna(subset=["y"], inplace=True)
        train_df, test_df = np.array_split(df_all, 2)
        train_mkt_df, test_mkt_df = np.array_split(df_mkt_res, 2)

        self.S0 = self.validate_df.iloc[0]["Mid Spread"]
        self.horizon = len(test_df)
        self._train_df = train_df
        self._test_df = test_df

        self._train_mkt_df = train_mkt_df
        self._test_mkt_df = test_mkt_df

    def _fit_mean_and_vol_models(self):
        def evaluate_models(y_true, model_preds):
            T = len(y_true)
            results = {}
            for name, preds in model_preds.items():
                mu = np.array(preds['mu'])
                var = np.array(preds['sigma2'])
                sigma = np.sqrt(var)
                e = y_true - mu
                mse = np.mean(e**2)
                mae = np.mean(np.abs(e))
                hit_rate = np.mean(np.sign(y_true) == np.sign(mu))
                var_mse = np.mean((var - e**2)**2)
                qlike = np.mean(np.log(var) + e**2/var)
                log_score = np.mean(norm.logpdf(y_true, loc=mu, scale=sigma))
                z = (y_true - mu) / sigma
                crps = np.mean(sigma * (1/np.sqrt(np.pi) - 2*norm.pdf(z) - z * (2*norm.cdf(z) - 1)))
                results[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'HitRate': hit_rate,
                    'Var_MSE': var_mse,
                    'QLIKE': qlike,
                    'LogScore': log_score,
                    'CRPS': crps
                }

            scores_df = pd.DataFrame(results).T
            losses = pd.DataFrame(index=scores_df.index)
            losses['MSE'] = scores_df['MSE']
            losses['MAE'] = scores_df['MAE']
            losses['HitRate'] = 1 - scores_df['HitRate']
            losses['Var_MSE'] = scores_df['Var_MSE']
            losses['QLIKE'] = scores_df['QLIKE']
            losses['LogScore'] = -scores_df['LogScore']
            losses['CRPS'] = scores_df['CRPS']
            norm_losses = (losses - losses.min()) / (losses.max() - losses.min())
            overall_scores = norm_losses.mean(axis=1)
            return scores_df, overall_scores

        # arima_orders = [(2,0,0), (2,0,1), (2,0,2), (4,0,0), (4,0,1), (4,0,2),
        #                 (5,0,0), (5,0,1), (5,0,2), (6,0,0), (6,0,1), (6,0,2)]
        # mean_types = ['Constant', 'Zero', 'LS', 'AR', 'ARX', 'HAR', 'HARX']
        # vol_types = ['GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'HARCH']
        # dist_types = ['t', 'studentst', 'skewstudent', 'skewt']

        arima_orders = [(5, 0, 2), (6, 0, 2)]
        mean_types = ['Zero']
        vol_types = ['GARCH']
        dist_types = ['t', 'studentst', 'skewstudent', 'skewt']

        # Prepare data
        y_train = self._train_df['y'].values
        y_test = self._test_df['y'].values
        test_len = len(y_test)
        model_preds = {}
        model_meta = {}

        for arima_order in arima_orders:
            try:
                sf = StatsForecast(models=[ARIMA(order=arima_order, include_constant=False)], freq="C", n_jobs=1)
                sf_mkt = StatsForecast(models=[ARIMA(order=arima_order, include_constant=False)], freq="C", n_jobs=1)
                
                sf.fit(df=self._train_df.assign(unique_id=1))
                sf_mkt.fit(df=self._train_mkt_df.assign(unique_id=1))
                
                mean_result = sf.fitted_[0, 0].model_
                mean_resid = mean_result.get("residuals")

                mean_mkt_result = sf_mkt.fitted_[0, 0].model_
                mean_mkt_resid = mean_mkt_result.get("residuals")
                vol_proc = GJRGARCHMarket(mean_mkt_resid)
                # Forecast for test set
                mean_fc = sf.forecast(df=self._train_df.assign(unique_id=1), h=test_len, fitted=True, X_df=self._test_df.assign(unique_id=1).drop(columns=["y"]))
                mean_pred = mean_fc[list(mean_fc.columns)[-1]].values
            except Exception:
                continue
            for mean_type in mean_types:
                for vol_type in vol_types:
                    for dist_type in dist_types:
                        try:
                            if vol_type == 'GARCH':
                                # Fit GARCH model
                                am = arch_model(mean_resid, mean=mean_type, vol=vol_type, p=1, q=1, dist=dist_type)
                                am.volatility = vol_proc
                            else:
                                am = arch_model(mean_resid, mean=mean_type, vol=vol_type, dist=dist_type)
                            res = am.fit(disp="off")
                            vf = res.forecast(horizon=test_len, reindex=False)
                            vol_pred = np.sqrt(np.maximum(1e-12, vf.variance.iloc[0].values))
                            model_name = f"ARIMA{arima_order}|{mean_type}|{vol_type}|{dist_type}"
                            model_preds[model_name] = {'mu': mean_pred, 'sigma2': vol_pred}
                            model_meta[model_name] = {
                                'arima_order': arima_order,
                                'mean_type': mean_type,
                                'vol_type': vol_type,
                                'dist_type': dist_type,
                                'mean_pred': mean_pred,
                                'vol_pred': vol_pred,
                                'mean_resid': mean_resid,
                                'vol_resid': res.std_resid,
                                'vol_distribution': res.model.distribution,
                                'vol_params': res.params[am.volatility.num_params:]
                            }
                        except Exception:
                            continue
        if not model_preds:
            raise RuntimeError("No valid model combination found.")
        scores_df, overall_scores = evaluate_models(y_test, model_preds)

        # Plot heatmap of overall scores
        fig, ax = plt.subplots(figsize=(8, max(4, len(overall_scores)//4)))
        im = ax.imshow(overall_scores.values.reshape(-1, 1), aspect='auto', cmap='viridis')
        ax.set_yticks(range(len(overall_scores)))
        ax.set_yticklabels(overall_scores.index)
        ax.set_xticks([0])
        ax.set_xticklabels(['Overall Score'])
        plt.colorbar(im, ax=ax, orientation='vertical')
        plt.tight_layout()
        plt.show()
        # Identify best model
        best_model = overall_scores.idxmin()
        print(f"The best model is {best_model} with overall score {overall_scores[best_model]:.3f}")
        # Save best model's predictions and residuals
        best = model_meta[best_model]
        self.mean_preds = best['mean_pred']
        self.vol_preds = best['vol_pred']
        self.mean_residuals = best['mean_resid']
        self.vol_residuals = best['vol_resid']
        self.vol_distribution = best['vol_distribution']
        self.vol_params = best['vol_params']

    def set_jump_params(self):
        """Estimate jump process parameters from volatility residuals."""
        self.sigma_residuals = np.std(self.vol_preds, ddof=1)
        thresh = 3 * self.vol_residuals.mean()
        jump_resid = self.vol_preds[np.abs(self.vol_preds) > thresh]
        n = len(self.vol_preds)
        if len(jump_resid) > 0:
            jump_up = jump_resid[jump_resid > 0]
            jump_down = -jump_resid[jump_resid < 0]
            self.jump_prob = len(jump_resid) / n
            self.p_up = len(jump_up) / len(jump_resid) if len(jump_resid) > 0 else 0.5
            self.eta_up = 1.0 / np.mean(jump_up) if len(jump_up) > 0 else 1.0
            self.eta_down = 1.0 / np.mean(jump_down) if len(jump_down) > 0 else 1.0

    def test_sign_bias(self, verbose=True):
        """
        Perform the Engle-Ng Sign Bias Test on the standardized residuals.
        """
        if self.vol_residuals is None:
            raise ValueError("No residuals found in best_model.")
        e = np.array(self.vol_residuals)
        e_lag = e[:-1]
        e_now = e[1:]
        z2_now = e_now ** 2
        D_neg = (e_lag < 0).astype(int)
        D_pos = (e_lag >= 0).astype(int)
        X = np.column_stack([
            D_neg,  # c1: sign bias
            D_neg * e_lag,  # c2: negative size bias
            D_pos * e_lag  # c3: positive size bias
        ])
        X = sm.add_constant(X)
        model = sm.OLS(z2_now, X).fit()
        if verbose:
            print(f"\n[Sign Bias Test – {self.name}]")
            print(model.summary())
            print("\n[Joint F-test for c1 = c2 = c3 = 0]")
            joint_test = model.f_test("x1 = x2 = x3 = 0")
            print(joint_test)
        return model

    def sample_jump(self):
        """Sample a jump from the estimated double-exponential jump process."""
        if np.random.rand() < self.p_up:
            return np.random.exponential(scale=1 / self.eta_up)
        else:
            return -np.random.exponential(scale=1 / self.eta_down)

    def get_PITs(self) -> np.ndarray:
        """
        Probability Integral Transforms (PITs) for GARCH standardized residuals (eta_it)
        using the fitted parametric conditional distribution F_it.
        Returns historical U_it from the training period.
        """
        if self.vol_distribution is None or len(self.vol_residuals) == 0:
            raise RuntimeError(f"GARCH model not properly fitted for {self.name}. Cannot compute PITs.")
        valid_residuals = np.nan_to_num(self.vol_residuals)
        u_values = self.vol_distribution.cdf(valid_residuals, parameters=self.vol_params)
        return np.clip(u_values, 1e-7, 1.0 - 1e-7)

    def ppf_standardized_residuals(self, u_array: np.ndarray) -> np.ndarray:
        """
        Transforms uniform draws U_it to standardized residuals eta_it = F_it^{-1}(U_it)
        using the PPF of the fitted GARCH conditional distribution.
        """
        if self.vol_distribution is None:
            raise RuntimeError(f"GARCH model not properly fitted for {self.name}. Cannot compute PPF.")
        clipped_u_array = np.clip(u_array, 1e-7, 1.0 - 1e-7)
        eta_values = self.vol_distribution.ppf(clipped_u_array, parameters=self.vol_params)
        return np.nan_to_num(eta_values)

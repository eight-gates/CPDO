from typing import Dict, List, Tuple

import pandas as pd
from matplotlib import pyplot as plt

from simulators.cds_wrapper import SingleCDS
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import zscore

from statsmodels.distributions.empirical_distribution import ECDF

from sklarpy.copulas import skewed_t_copula, student_t_copula, gh_copula

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def to_pseudo_obs(x: np.ndarray) -> np.ndarray:
    """ECDF‑based PITs computed column‑wise using statsmodels' ECDF."""
    return np.column_stack([ECDF(col)(col) for col in x.T])

class CDSSimulator:
    def __init__(self, mapping: Dict[str, Tuple[str, int, str]], nu, copula_testing=False):
        """mapping: name → (csv_path, group_id ∈ 1..10)"""
        self.names = list(mapping.keys())
        self.dim = len(self.names)
        self.nu = nu
        self.paths = {nm: path for nm, (path, _, _) in mapping.items()}
        self.group_ids = [gid for _, (_, gid, _) in mapping.items()]  # zero‑index
        self.mkt_paths = {nm: mkt_path for nm, (_, _, mkt_path) in mapping.items()}
        self.horizon = np.inf

        mdists_dict = {}
        # ---------- fit marginal models ----------
        self.models: Dict[str, SingleCDS] = {}
        self.min_len = np.inf

        for nm, (csv, gid, market_csv) in tqdm(mapping.items(), desc="Fitting marginals"):
            mdl = SingleCDS(nm, csv, gid, market_csv)  # zero‑index gid
            self.models[nm] = mdl

            self.min_len = min(self.min_len, len(mdl.vol_residuals))
            self.horizon = min(self.horizon, len(mdl.vol_preds))

        for i, (name, model) in enumerate(self.models.items()):
            mdists_dict[name] = model._train_df['y'].values[-int(self.min_len):]

        mdists_df: pd.DataFrame = pd.DataFrame(mdists_dict, dtype=float)
        
        # resid_mat = np.vstack([zscore(m._train_df[["y"]][-int(self.min_len):]) for m in self.models.values()])
        # self.z = resid_mat
        # self.corr_matrix = self._estimate_correlation()
        # self.L = np.linalg.cholesky(self.corr_matrix)

        student_copula = student_t_copula.fit(data=mdists_df,
                                           method='itau',
                                           univariate_fitter_options={
                                               'significant': False
                                           },
                                           show_progress=True)
        if copula_testing:
            skewt_copula = skewed_t_copula.fit(data=mdists_df,
                                               method='mle',
                                               univariate_fitter_options={
                                                   'significant': False,
                                                   'max_iter': 60
                                               },
                                               show_progress=True)
            gaussian_copula = gh_copula.fit(data=mdists_df,
                                               method='mle',
                                               univariate_fitter_options={
                                                   'significant': False,
                                                   'max_iter': 60
                                               },
                                               show_progress=True)


            if (student_copula.aic()+student_copula.bic()) < (skewt_copula.aic()+skewt_copula.bic()):
                print("Student t copula is better than skewt copula")
                self.copula = student_t_copula
            else:
                print("Skewt copula is better than student t copula")
                self.copula = skewt_copula

            print(self.copula.summary)
            print("Gaussian copula statistics in comparison:")
            print(gaussian_copula.summary)

        print(student_copula.summary)
        self.copula = student_copula


        # self._plot_copula_pdfs()
        # gaussian_copula.copula_pdf_plot()

    def _plot_copula_pdfs(self):
        """
        Plot the copula PDF and copula density for diagnostic purposes.
        This method is separated for flexibility and reusability.
        """
        self.copula.pdf_plot(show=False, axes_names=self.names)
        self.copula.copula_pdf_plot(show=False, axes_names=self.names)
        plt.show()

    def _estimate_correlation(self):
        return np.corrcoef(self.z)

    def _sample_t_copula_shock(self):
        z = np.random.standard_t(self.nu, size=self.dim)
        correlated_normal = self.L.dot(z)
        # Chi-Quadrat for t-DoF
        chi2 = np.random.chisquare(self.nu)
        # Multivariate t-vector
        shock = correlated_normal * np.sqrt(self.nu / chi2)
        return shock

    def simulate_paths_v2(self, n_paths: int = 1000):
        level_paths = np.zeros((n_paths, self.horizon, len(self.names)))

        shock_grid = []
        for _ in range(n_paths):
            # shocks = np.vstack([self._sample_t_copula_shock() for _ in range(self.horizon)]).T
            shocks = np.vstack(self.copula.rvs(self.horizon)).T
            shock_grid.append(shocks)

        for j, nm in tqdm(enumerate(self.names), total=len(self.names), desc="Simulating series"):
            mdl = self.models[nm]
            mu = mdl.mean_preds[-self.horizon:]  # length horizon
            sigma = mdl.vol_preds[-self.horizon:]
            logS0 = np.log(mdl.S0)
            for p in range(n_paths):
                diff_path = mu + sigma * shock_grid[p][j, :]
                cum_path = logS0 + np.cumsum(diff_path)
                # cum_path = mdl.S0 + np.cumsum(diff_path)
                level_paths[p, :, j] = np.exp(cum_path)
                # level_paths[p, :, j] = cum_path
                level_paths[p, :, j] = np.maximum(0.0, level_paths[p, :, j])
                # level_paths[p, :, j] = np.maximum(0.0, cum_path)

        days = np.arange(self.horizon)

        mean_paths = np.zeros((len(self.names), self.horizon))
        for j, nm in enumerate(self.names):
            hist = self.models[nm].validate_df['Mid Spread'].values[:self.horizon]
            errs = np.abs(level_paths[:, :, j] - hist).mean(axis=1)
            meanie, worst = errs.argmin(), level_paths[:, :, j].mean(axis=0)
            mean_paths[j, :] = level_paths[meanie, :, j]
            plt.figure(figsize=(10, 5))
            plt.plot(days, hist, label='Historical', linewidth=2)
            # plt.plot(days, worst, label='Monte Carlo Avg', linewidth=2)
            plt.plot(days, level_paths[meanie, :, j], label='Monte Carlo Average', linestyle='--')
            plt.title(f"{nm}: Realized Historical Spread vs Monte Carlo Average")
            plt.xlabel('Day')
            plt.ylabel('Spread (bps)')
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        mean_best_path = mean_paths.mean(axis=0)
        best_path_matrix = np.tile(mean_best_path, (n_paths, 1))


        idx = pd.MultiIndex.from_product([range(n_paths), range(self.horizon)], names=["Path", "Day"])
        df = pd.DataFrame(
            level_paths.reshape(n_paths * (self.horizon), len(self.names)), index=idx, columns=self.names
        )
        return level_paths, df, best_path_matrix




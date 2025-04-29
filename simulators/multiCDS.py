'''
Open questions:
1. We apply the copula to have an innovation that is derived from the correlation of other CDS instruments in the basket.
But I don't believe the story ends there, the dependencies from the broader market should probably also be factored into the simulations:
a) CDS indicies
b) Value of the "Floater" that defined the spread for each CDS
'''
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from matplotlib import pyplot as plt
from simulators.jumpdiffusion import MOUJumpDiffusion
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

class MultiCDSSimulator:
    def __init__(self, name_to_series_map, nu=2):
        self.cds_basket = {name: MOUJumpDiffusion(name, series)
                       for name, series in name_to_series_map.items()}
        self.names = list(self.cds_basket.keys())
        self.dim = len(self.names)
        self.nu = nu  # degrees of freedom for t-copula
        self.multicds_horizon = 0
        self.corr_matrix = self._estimate_correlation()
        # Cholesky-Decomposition matrix
        self.L = np.linalg.cholesky(self.corr_matrix)

    def _estimate_correlation(self):
        resid_data = []
        number_of_predictions = []
        for cds in self.cds_basket.values():
            resid_data.append(cds.best_model["residuals"])
            number_of_predictions.append(len(cds.best_model["predictions"]))

        min_len_residuals = min(len(r) for r in resid_data)
        self.multicds_horizon = min(number_of_predictions)
        resid_data = [r[-min_len_residuals:] for r in resid_data]
        resid_matrix = np.vstack(resid_data)
        return np.corrcoef(resid_matrix)

    def report_mc_error(self,paths: np.ndarray, cds_names: list[str]):
        err = self.compute_mc_error(paths)
        se = err['se_mean']  # (T,n)
        avg_se = se.mean(axis=0)  # per CDS
        ci_low = err['ci_lower']; ci_high = err['ci_upper']

        # Text summary
        print("Monte‑Carlo sampling error(average SE over horizon):")
        for nm, s in zip(cds_names, avg_se):
            print(f"  {nm:<25s}: ±{s:8.3f} bps (1‑σ)")

        # Plot: bar chart of avg SE per CDS
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(cds_names, avg_se)
        ax.set_ylabel('Average Std‑Error (bps)')
        ax.set_title('Monte‑Carlo Standard Error by CDS')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout();
        plt.show()

        # Plot: SE over time for worst CDS (highest avg SE)
        worst_idx = np.argmax(avg_se)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(se[:, worst_idx])
        ax.set_title(f'Standard Error over time – {cds_names[worst_idx]}')
        ax.set_xlabel('Day');
        ax.set_ylabel('Std‑Error (bps)')
        plt.tight_layout();
        plt.show()

        return err

    def compute_mc_error(self, paths, alpha=0.05, n_resamples=2000):
        n, T, d = paths.shape
        mean_path = paths.mean(axis=0)
        se_mean = paths.std(axis=0, ddof=1) / np.sqrt(n)

        ci_low = np.zeros((T, d))
        ci_high = np.zeros((T, d))
        for t in range(T):
            for j in range(d):
                res = bootstrap(
                    (paths[:, t, j],), np.mean,
                    confidence_level=1 - alpha,
                    n_resamples=n_resamples,
                    method='percentile'
                )
                ci_low[t, j], ci_high[t, j] = res.confidence_interval
        return {'se_mean': se_mean, 'ci_lower': ci_low, 'ci_upper': ci_high}

    def _sample_t_copula_shock(self):
        z = np.random.normal(size=self.dim)
        correlated_normal = self.L.dot(z)
        # Chi-Quadrat for t-DoF
        chi2 = np.random.chisquare(self.nu)
        # Multivariate t-vector
        shock = correlated_normal * np.sqrt(self.nu / chi2)
        return shock

    def simulate_paths(self, n_paths=250):
        n = len(self.names)
        T = self.multicds_horizon
        # base = np.vstack([self.cds_basket[name].best_model['predictions'][:T]
                          # for name in self.names]).T
        all_paths = np.empty((n_paths, T, n))

        for j, name in enumerate(self.names):
            all_paths[:, :, j] = self.cds_basket[name].best_model["predictions"][:self.multicds_horizon]

        # Simulation
        for i in range(n_paths):
            for t in range(self.multicds_horizon):
                shocks = self._sample_t_copula_shock()
                for j, name in enumerate(self.names):
                    model = self.cds_basket[name]
                    innovation = model.sigma_residuals * shocks[j]
                    jump = 0
                    if model.jump_prob and np.random.rand() < model.jump_prob:
                        jump = np.random.normal(loc=model.jump_mean, scale=model.jump_std)
                    innovation += jump
                    all_paths[i, t, j] += innovation
                    all_paths[i, t, j] = max(0.0, all_paths[i, t, j]) #flooring

        days = np.arange(T)
        for j, nm in enumerate(self.names):
            hist = self.cds_basket[nm].history_test['Mid Spread'].values[:T]
            errs = np.abs(all_paths[:, :, j] - hist).mean(axis=1)
            best, worst = errs.argmin(), errs.argmax()
            plt.figure(figsize=(10, 5))
            plt.plot(days, hist, label='Historical', linewidth=2)
            plt.plot(days, all_paths[best, :, j], label='Best Fit', linestyle='--')
            plt.plot(days, all_paths[worst, :, j], label='Worst Fit', linestyle=':')
            plt.title(f"{nm}: Best/Worst Simulation vs Historical")
            plt.xlabel('Day');
            plt.ylabel('Spread (bps)')
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
            plt.grid(True);
            plt.tight_layout();
            plt.show()


        idx = pd.MultiIndex.from_product([range(n_paths), range(self.multicds_horizon)], names=["Path", "Day"])
        df = pd.DataFrame(
            all_paths.reshape(n_paths * (self.multicds_horizon), n), index=idx, columns=self.names
        )
        self.report_mc_error(all_paths, self.names)
        return all_paths, df

    def simulate_paths_multithreaded(self, n_paths: int = 250, max_workers: int = None):
        n_names = len(self.names)
        T = self.multicds_horizon

        base_paths = np.stack([
            self.cds_basket[name].best_model['predictions'][:T]
            for name in self.names
        ], axis=1)

        all_paths = np.empty((n_paths, T, n_names), dtype=float)

        def _simulate_one(seed_and_idx):
            seed, idx = seed_and_idx
            rng = np.random.default_rng(seed)
            path = base_paths.copy()

            for t in range(T):
                z = rng.standard_normal(n_names)
                corr_norm = self.L.dot(z)
                chi2 = rng.chisquare(self.nu)
                shocks = corr_norm * np.sqrt(self.nu / chi2)

                for j, name in enumerate(self.names):
                    mdl = self.cds_basket[name]
                    incr = mdl.sigma_residuals * shocks[j]
                    if mdl.jump_prob > 0 and rng.random() < mdl.jump_prob:
                        incr += rng.normal(mdl.jump_mean, mdl.jump_std)
                    # floor at zero
                    path[t, j] = max(0.0, path[t, j] + incr)

            return idx, path

        seeds_and_indices = [(int(1e6 + i), i) for i in range(n_paths)]
        with ThreadPoolExecutor(max_workers=max_workers) as exec:
            for idx, sim in exec.map(_simulate_one, seeds_and_indices):
                all_paths[idx] = sim

        idx = pd.MultiIndex.from_product(
            [range(n_paths), range(T)],
            names=["Path", "Day"]
        )
        df = pd.DataFrame(
            all_paths.reshape(n_paths * T, n_names),
            index=idx, columns=self.names
        )
        return all_paths, df


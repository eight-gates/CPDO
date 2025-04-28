'''
Open questions:
1. We apply the copula to have an innovation that is derived from the correlation of other CDS instruments in the basket.
But I don't believe the story ends there, the dependencies from the broader market should probably also be factored into the simulations:
a) CDS indicies
b) Value of the "Floater" that defined the spread for each CDS
'''
import MOUJumpDiffusion
import numpy as np
import pandas as pd

class MultiCDSSimulator:
    def __init__(self, name_to_series_map, nu=2):
        self.cds_basket = {name: MOUJumpDiffusion(name, series)
                       for name, series in name_to_series_map.items()}
        self.names = list(self.models.keys())
        self.dim = len(self.names)
        self.nu = nu  # degrees of freedom for t-copula
        self.corr_matrix = self._estimate_correlation()
        self.multicds_horizon = 0
        # Cholesky-Decomposition matrix
        self.L = np.linalg.cholesky(self.corr_matrix)

    def _estimate_correlation(self):
        resid_data = []
        number_of_predictions = []
        for cds in self.cds_basket.values():
            resid_data.append(cds["residuals"])
            number_of_predictions.append(len(cds["predictions"]))
        min_len = min(len(r) for r in resid_data)
        self.multicds_horizon = min(number_of_predictions)
        resid_data = [r[-min_len:] for r in resid_data]
        resid_matrix = np.vstack(resid_data)
        return np.corrcoef(resid_matrix)

    def _sample_t_copula_shock(self):
        z = np.random.normal(size=self.dim)
        correlated_normal = self.L.dot(z)
        # Chi-Quadrat for t-DoF
        chi2 = np.random.chisquare(self.nu)
        # Multivariate t-vector
        shock = correlated_normal * np.sqrt(self.nu / chi2)
        return shock

    def simulate_paths(self, n_paths=250):
        n_names = len(self.names)
        all_paths = np.zeros((n_paths, self.multicds_horizon, n_names))
        for j, name in enumerate(self.names):
            all_paths[:, :, j] = self.cds_basket[name]["predictions"]
        # Simulation
        for i in range(n_paths):
            for t in range(self.multicds_horizon):
                shocks = self._sample_t_copula_shock()
                for j, name in enumerate(self.names):
                    model = self.models[name]
                    innovation = model.sigma_residuals * shocks[j]
                    jump = 0
                    if model.jump_prob and np.random.rand() < model.jump_prob:
                        jump = np.random.normal(loc=model.jump_mean, scale=model.jump_std)
                        innovation += jump
                    all_paths[i, t, j] += innovation
                    all_paths[i, t, j] = max(0.0, all_paths[i, t, j]) #flooring

        idx = pd.MultiIndex.from_product([range(n_paths), range(self.multicds_horizon)], names=["Path", "Day"])
        df = pd.DataFrame(
            all_paths.reshape(n_paths * (self.multicds_horizon), n_names), index=idx, columns=self.names
        )
        return all_paths, df


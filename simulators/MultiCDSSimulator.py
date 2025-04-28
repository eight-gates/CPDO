'''
Open questions:
1. We apply the copula to have an innovation that is derived from the correlation of other CDS instruments in the basket.
But I don't believe the story ends there, the dependencies from the broader market should probably also be factored into the simulations:
a) CDS indicies
b) Value of the "Floater" that defined the spread for each CDS
'''
class MultiCDSSimulator:
    """
    Simulator für mehrere CDS-Spreads mit tail-abhängiger t-Copula-Korrelation.
    """
    def __init__(self, name_to_series_map, nu=2):
        """
        name_to_series_map: Dict von CDS-Namen zu historischen Spread-Pandas-Series.
        nu: Freiheitsgrade der t-Copula (Standard 5 für fat tails).
        """
        # Erzeuge ein OUJumpDiffusionModel für jeden Namen
        self.models = {name: OUJumpDiffusionModel(name, series)
                       for name, series in name_to_series_map.items()}
        self.names = list(self.models.keys())
        self.dim = len(self.names)
        self.nu = nu  # degrees of freedom for t-copula
        # Korrelation der Innovations schätzen (auf Basis der Residuen der Modelle)
        self.corr_matrix = self._estimate_correlation()
        # Cholesky-Dekomposition für schnelle Simulation
        self.L = np.linalg.cholesky(self.corr_matrix)

    def _estimate_correlation(self):
        """
        Schätzt die Korrelationsmatrix der Innovations der einzelnen CDS.
        Hier nutzen wir die Korrelation der AR(1)-Residuen aus den Single-Name-Modellen.
        """
        resid_data = []
        for model in self.models.values():
            # Residuen erneut berechnen (oder aus calibrate zwischenspeichern)
            X = model.history.values
            X_lag = X[:-1];
            X_next = X[1:]
            preds = model.alpha + model.phi * X_lag
            resid = X_next - preds
            resid_data.append(resid)
        # Gleiche Länge erzwingen (truncate auf Minimum, um Alignment zu gewährleisten)
        min_len = min(len(r) for r in resid_data)
        resid_data = [r[-min_len:] for r in resid_data]
        resid_matrix = np.vstack(resid_data)
        # Korrelationsmatrix berechnen (Pearson)
        return np.corrcoef(resid_matrix)

    def _sample_t_copula_shock(self):
        """
        Zieht einen einzelnen Zufallsvektor (dim = Anzahl CDS) aus der t-Copula.
        """
        # Ziehe d-dim Normal(0,1) Vektor mit Korrelation corr_matrix (via Cholesky)
        z = np.random.normal(size=self.dim)
        correlated_normal = self.L.dot(z)
        # Ziehe Chi-Quadrat für t-DoF
        chi2 = np.random.chisquare(self.nu)
        # Multivariater t Vektor
        shock = correlated_normal * np.sqrt(self.nu / chi2)
        # Jeder Eintrag in shock ist nun t-verteilt mit dof=self.nu und Korrelation wie corr_matrix
        return shock

    def simulate_paths(self, n_paths=250, T=252):
        """
        Simuliert n_paths Pfade über T Tage für alle CDS.
        Rückgabe: 3D-NumPy-Array mit Dimension (n_paths, T+1, n_names)
                  oder alternativ ein DataFrame pro Pfad.
        """
        n_names = len(self.names)
        # Array für alle Pfade
        all_paths = np.zeros((n_paths, T + 1, n_names))
        # Anfangswerte setzen (gemeinsam für alle Pfade)
        for j, name in enumerate(self.names):
            all_paths[:, 0, j] = self.models[name].x0
        # Simulation
        for i in range(n_paths):
            # Für jeden Pfad separat simulieren
            for t in range(1, T + 1):
                # Korrelierte Zufallsschocks für alle Namen an Tag t
                shocks = self._sample_t_copula_shock()
                for j, name in enumerate(self.names):
                    model = self.models[name]
                    x_prev = all_paths[i, t - 1, j]
                    # OU-Update (diskret): X_next = alpha + phi * X_prev + sigma_eps * shock + optional jump
                    drift = model.alpha + model.phi * x_prev
                    x_t = drift + model.sigma_eps * shocks[j]
                    if model.jump_prob and np.random.rand() < model.jump_prob:
                        jump = np.random.normal(loc=model.jump_mean, scale=model.jump_std)
                        x_t += jump
                    x_t = max(0.0, x_t)  # ⬅️ Add this line to floor the spread
                    all_paths[i, t, j] = x_t

        # Optional: Ausgabe in DataFrame-Struktur pro Pfad
        # Beispiel: Ein DataFrame mit MultiIndex (pfad, tag) und Spalten = CDS-Namen:
        idx = pd.MultiIndex.from_product([range(n_paths), range(T + 1)], names=["Pfad", "Tag"])
        df = pd.DataFrame(
            all_paths.reshape(n_paths * (T + 1), n_names), index=idx, columns=self.names
        )
        return all_paths, df


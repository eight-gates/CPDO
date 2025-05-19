import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import rankdata, kstest

from cdo import CPDO
from basket import CDSSimulator

# ===================== CONFIGURATION ===================== #
script_dir = os.path.dirname(os.path.abspath(__file__))

HISTORICAL_DATA = {
    # "Deutsche Bank": (os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/DEUTSCHE_BANK_DE_2015_2025.csv"), 1, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Commerzbank": (os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/COMMERZBANK_DE_2015_2025.csv"), 1, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Banca Intesa Sanpaolo": (os.path.join(script_dir, "data/banks/baa1 below/mod-mod rec 2014/INTESA_SANPAOLO_SPA_ITA_2015_2025.csv"), 1, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Bayer AG": (os.path.join(script_dir, "data/pharma/mod-mod rec 2014/BAYER_AG_DE_2015_2025.csv"), 2, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Marks and Spencers": (os.path.join(script_dir, "data/retail/baa1 below/mod-mod rec 2014/MARKS_AND_SPENCERS_GB_2015_2025.csv"), 3, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Lufthansa": (os.path.join(script_dir, "data/airlines/mod-mod rec 2014/LUFTHANSA_AIR_DE_2015_2025.csv"), 4, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Volkswagen": (os.path.join(script_dir, "data/automobile/baa1 below/mod-mod rec 2014/VOLKSWAGEN_DE_2015_2025.csv"), 5, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Siemens": (os.path.join(script_dir, "data/tech/mod-mod rec 2014/SIEMENS_DE_2015_2025.csv"), 6, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Stora Enso OYJ": (os.path.join(script_dir, "data/tech/mod-mod rec 2014/STORA_ENSO_FI_2015_2025.csv"), 6, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    "Hammerson": (os.path.join(script_dir, "data/real estate/baa1 above/HAMMERSON_GB_2015_2025.csv"), 7, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Deutschland": (os.path.join(script_dir, "data/sovereign/cum restrut 2014/DEUTSCHLAND_2015_2025.csv"), 8, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Prudential": (os.path.join(script_dir, "data/life insurance/baa1 above/PRUDENTIAL_GB_2015_2025.csv"), 9, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Aegon LTD" : (os.path.join(script_dir, "data/life insurance/baa1 below/AEGON_LTD_NL_2015_2025.csv"), 9, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Goldman Sachs" : (os.path.join(script_dir, "data/misc/GOLDMAN_SACHS_EX-RESTRUCT_US_2015_2025.csv"), 10, "/Users/kartik/PycharmProjects/CPDO/data/index/CDX_IG_5Y_US_2015_2025.csv"),
    # "Morgan Stanley" : (os.path.join(script_dir, "data/misc/MORGAN_STANLEY_EX-RESTRUCT_US_2015_2025.csv"), 10, "/Users/kartik/PycharmProjects/CPDO/data/index/CDX_IG_5Y_US_2015_2025.csv"),
    # "GlaxoSmithKline" : (os.path.join(script_dir, "data/pharma/mod-mod rec 2014/GSK_GB_2015_2025.csv"), 2, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Sanofi" : (os.path.join(script_dir, "data/pharma/mod-mod rec 2014/SANOFI_FR_2015_2025.csv"), 2, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    "Astrazeneca" : (os.path.join(script_dir, "data/pharma/mod-mod rec 2014/ASTRAZENECA_GB_2015_2025.csv"), 2, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Unibail Rodamco Westfield": (os.path.join(script_dir, "data/real estate/baa1 below/UNIBAIL_RODAMCO_WESTFIELD_FR_2015_2025.csv"), 7, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "GAP": (os.path.join(script_dir, "data/retail/baa1 above/GAP_EX-RESTRUCT_US_2015_2025.csv"), 3, "/Users/kartik/PycharmProjects/CPDO/data/index/CDX_IG_5Y_US_2015_2025.csv"),
    # "Koninklijke Ahold Delhaize": (os.path.join(script_dir, "data/retail/baa1 above/mod-mod rec 2014/KONINKLIJKE_AHOLD_DELHAIZE_NL_2015_2025.csv"), 3, "/Users/kartik/PycharmProjects/CPDO/data/index/CDX_IG_5Y_US_2015_2025.csv"),
    # "J Sainsbury" : (os.path.join(script_dir, "data/retail/baa1 below/mod-mod rec 2014/J_SAINSBURY_GB_2015_2025.csv"), 3, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Walmart" : (os.path.join(script_dir, "data/retail/WALMART_EX-RESTRUCT_CH_2015_2025.csv"), 3, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Brazil" : (os.path.join(script_dir, "data/sovereign/cum restrut 2014/BRAZIL_2015_2025.csv"), 8, "/Users/kartik/PycharmProjects/CPDO/data/index/CDX_IG_5Y_US_2015_2025.csv"),
    # "USA" : (os.path.join(script_dir, "data/sovereign/cum restrut 2014/US_SOV5YR_US_2015_2025.csv"), 8, "/Users/kartik/PycharmProjects/CPDO/data/index/CDX_IG_5Y_US_2015_2025.csv"),
    # "Leonardo SPA" : (os.path.join(script_dir, "data/tech/mod-mod rec 2014/LEONARDO_SPA_ITA_2015_2025.csv"), 6, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Alstom": (os.path.join(script_dir, "data/tech/mod-mod rec 2014/ALSTOM_FRA_2015_2025.csv"), 6, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Apple" : (os.path.join(script_dir, "data/tech/APPLE_EX-RESTRUCT_US_2015_2025.csv"), 6, "/Users/kartik/PycharmProjects/CPDO/data/index/CDX_IG_5Y_US_2015_2025.csv"),
    # "Mediobanca" : (os.path.join(script_dir, "data/banks/baa1 below/mod-mod rec 2014/MEDIOBANCA_BANCA_SPA_ITA_2015_2025.csv"), 1, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "UBS": (os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/UBS_CH_2015_2025.csv"), 1, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Credit Agricole" : (os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/CREDIT_AGRICOLE_FR_2015_2025.csv"), 1, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Banco Bilbao" : (os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/BANCO_BILBAO_ES_2015_2025.csv"), 1, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Barclays" : (os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/BARCLAYS_GB_2015_2025.csv"), 1, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Mercedes Benz": (os.path.join(script_dir, "data/automobile/baa1 above/mod-mod rec 2014/MERCEDES_BENZ_DE_2015_2025.csv"), 5, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Greece": (os.path.join(script_dir, "data/sovereign/cum restrut 2014/GREECE_2015_2025.csv"), 8, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv"),
    # "Turkey": (os.path.join(script_dir, "data/sovereign/cum restrut 2014/TURKEY_2015_2025.csv"), 8, "/Users/kartik/PycharmProjects/CPDO/data/index/ITRAXX_EUROPE_IG43_5Y_EU_2015_2025.csv")
}

# ===================== UTILITY FUNCTIONS ===================== #
def compute_pseudo_observations(data):
    ranks = np.array([rankdata(col, method='average') for col in data.T]).T
    pseudo_obs = ranks / (data.shape[0] + 1)
    return pseudo_obs

def plot_heatmap(matrix, title, cmap, cbk):
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(n * 0.3 + 4, n * 0.3 + 4))
    annot_kws = {'size': max(6, int(200 / (n * n)))}
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, square=True,
                annot_kws=annot_kws, cbar_kws=cbk)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ===================== EVALUATION CLASS ===================== #
class CopulaDependenceEvaluator:
    def __init__(self, original_corr_matrix, dof=5):
        self.original_corr_matrix = original_corr_matrix
        self.dof = dof

    def evaluate(self, paths_array, show_plots=True):
        n_paths, T, n_names = paths_array.shape
        paths_reshaped = paths_array[:, 1:, :].reshape(-1, n_names)
        returns = np.diff(paths_array, axis=1).reshape(-1, n_names)

        simulated_corr_matrix = np.corrcoef(returns.T)
        frob_error = np.linalg.norm(simulated_corr_matrix - self.original_corr_matrix)

        thresholds = np.percentile(paths_reshaped, 95, axis=0)
        joint_extremes = np.all(paths_reshaped > thresholds, axis=1)
        joint_extreme_count = np.sum(joint_extremes)
        joint_extreme_frequency = joint_extreme_count / len(paths_reshaped)

        pseudo_obs = compute_pseudo_observations(returns)
        ks_stats = [kstest(pseudo_obs[:, i], 'uniform') for i in range(n_names)]
        ks_results = {f"CDS_{i+1}": {'statistic': stat.statistic, 'p_value': stat.pvalue} for i, stat in enumerate(ks_stats)}

        if show_plots:
            plot_heatmap(self.original_corr_matrix, "Empirical Correlation Matrix (Historical)", cmap='Blues', cbk={'label': {'label': 'Empirical Corr'}})
            plot_heatmap(simulated_corr_matrix, "Simulated Correlation Matrix (From Copula Spreads)", cmap='Reds', cbk={'label': 'Simulated Corr'})
            fig, axs = plt.subplots(1, n_names, figsize=(4 * n_names, 4))
            for i in range(n_names):
                ecdf = ECDF(pseudo_obs[:, i])
                uniform_q = np.linspace(0, 1, 100)
                axs[i].plot(uniform_q, ecdf(uniform_q), label='Empirical CDF')
                axs[i].plot(uniform_q, uniform_q, 'r--', label='Uniform CDF')
                axs[i].set_title(f'Pseudo-Obs CDF (CDS {i+1})')
                axs[i].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
            plt.suptitle("Empirical vs Uniform CDFs of Pseudo-Observations")
            plt.tight_layout()
            plt.show()

        return {
            "frob_error": frob_error,
            "simulated_corr_matrix": simulated_corr_matrix,
            "joint_extreme_frequency": joint_extreme_frequency,
            "joint_extreme_count": joint_extreme_count,
            "ks_test_uniformity": ks_results,
        }

# ===================== SIMULATION CLASS ===================== #
class MultiCDSSimulation:
    def __init__(self, historical_data, num_paths=10000):
        self.historical_data = historical_data
        self.num_paths = num_paths
        self.simulator = CDSSimulator(mapping=historical_data, nu=len(historical_data.items()))
        self.cds_names = self.simulator.names

    def run(self):
        paths_array, paths_df, mean_path = self.simulator.simulate_paths_v2(self.num_paths)
        return paths_array, paths_df, mean_path

    def plot_scenarios(self, paths_df, scenarios=5):
        for pfad in range(scenarios):
            plt.figure(figsize=(12, 6))
            for name in self.cds_names:
                series = paths_df.loc[pfad][name].values
                plt.plot(series, label=name)
            plt.title(f'Scenario {pfad + 1}: CDS Spread Simulations')
            plt.xlabel('Days')
            plt.ylabel('CDS Spread (bps)')
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_average_spreads(self, paths_array):
        avg_spreads = paths_array.mean(axis=0)
        days = np.arange(avg_spreads.shape[0])
        plt.figure(figsize=(12, 6))
        for i, name in enumerate(self.cds_names):
            plt.plot(days, avg_spreads[:, i], label=name)
        plt.title("Average spread per day for each CDS")
        plt.xlabel("Days")
        plt.ylabel("CDS Spread (bps)")
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.tight_layout()
        plt.show()

# ===================== MAIN EXECUTION ===================== #
def main():
    # Simulation
    simulation = MultiCDSSimulation(HISTORICAL_DATA, num_paths=1000)

    paths_array, paths_df, mean_path = simulation.run()

    # Visualization
    simulation.plot_scenarios(paths_df, scenarios=5)
    simulation.plot_average_spreads(paths_array)

    # Evaluation
    # evaluator = CopulaDependenceEvaluator(original_corr_matrix=simulation.simulator.corr_matrix, dof=5)
    # results = evaluator.evaluate(paths_array, show_plots=True)
    # print(results)

    # CPDO Calculation
    # cpdo = CPDO(spreads=mean_path, coupon=0.045, r=0.03)
    # print("Probability of Default:", cpdo.PD)
    # print("Expected Loss:", cpdo.EL)
    # print("99% VaR of loss:", cpdo.VaR95)
    # print("Implied Rating:", cpdo.rating)

if __name__ == "__main__":
    main()


import warnings
warnings.filterwarnings("ignore")

from cdo import CPDO
import pandas as pd

from simulators.multiCDS import MultiCDSSimulator
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import t, norm, rankdata, kstest
import seaborn as sns

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

historical_data = {
    # "Deutsche Bank": os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/DEUTSCHE_BANK_DE_2015_2025.csv"),
    # "Commerzbank": os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/COMMERZBANK_DE_2015_2025.csv"),
    # "Banca Intesa Sanpaolo": os.path.join(script_dir, "data/banks/baa1 below/mod-mod rec 2014/INTESA_SANPAOLO_SPA_ITA_2015_2025.csv"),
    # "Bayer AG": os.path.join(script_dir, "data/pharma/mod-mod rec 2014/BAYER_AG_DE_2015_2025.csv"),
    # "Marks and Spencers": os.path.join(script_dir, "data/retail/baa1 below/mod-mod rec 2014/MARKS_AND_SPENCERS_GB_2015_2025.csv"),
    "Lufthansa": os.path.join(script_dir, "data/airlines/mod-mod rec 2014/LUFTHANSA_AIR_DE_2015_2025.csv"),
    "Volkswagen": os.path.join(script_dir, "data/automobile/baa1 below/mod-mod rec 2014/VOLKSWAGEN_DE_2015_2025.csv"),
    # "Siemens": os.path.join(script_dir, "data/tech/mod-mod rec 2014/SIEMENS_DE_2015_2025.csv"),
    # "Stora Enso OYJ": os.path.join(script_dir, "data/tech/mod-mod rec 2014/STORA_ENSO_FI_2015_2025.csv"),
    # "Hammerson": os.path.join(script_dir, "data/real estate/baa1 above/HAMMERSON_GB_2015_2025.csv"),
    # "Deutschland": os.path.join(script_dir, "data/sovereign/cum restrut 2014/DEUTSCHLAND_2015_2025.csv"),
    # "Prudential": os.path.join(script_dir, "data/life insurance/baa1 above/PRUDENTIAL_GB_2015_2025.csv"),
    "Aegon LTD" : os.path.join(script_dir, "data/life insurance/baa1 below/AEGON_LTD_NL_2015_2025.csv"),
    # "Goldman Sachs" : os.path.join(script_dir, "data/misc/GOLDMAN_SACHS_EX-RESTRUCT_US_2015_2025.csv"),
    # "Morgan Stanley" : os.path.join(script_dir, "data/misc/MORGAN_STANLEY_EX-RESTRUCT_US_2015_2025.csv"),
    # "GlaxoSmithKline" : os.path.join(script_dir, "data/pharma/mod-mod rec 2014/GSK_GB_2015_2025.csv"),
    # "Sanofi" : os.path.join(script_dir, "data/pharma/mod-mod rec 2014/SANOFI_FR_2015_2025.csv"),
    # "Astrazeneca" : os.path.join(script_dir, "data/pharma/mod-mod rec 2014/ASTRAZENECA_GB_2015_2025.csv"),
    # "Unibail Rodamco Westfield": os.path.join(script_dir, "data/real estate/baa1 below/UNIBAIL_RODAMCO_WESTFIELD_FR_2015_2025.csv"),
    # "GAP": os.path.join(script_dir, "data/retail/baa1 above/GAP_EX-RESTRUCT_US_2015_2025.csv"),
    # "Koninklijke Ahold Delhaize": os.path.join(script_dir, "data/retail/baa1 above/mod-mod rec 2014/KONINKLIJKE_AHOLD_DELHAIZE_NL_2015_2025.csv"),
    "J Sainsbury" : os.path.join(script_dir, "data/retail/baa1 below/mod-mod rec 2014/J_SAINSBURY_GB_2015_2025.csv"),
    # "Walmart" : os.path.join(script_dir, "data/retail/WALMART_EX-RESTRUCT_CH_2015_2025.csv"),
    # "Brazil" : os.path.join(script_dir, "data/sovereign/cum restrut 2014/BRAZIL_2015_2025.csv"),
    # "USA" : os.path.join(script_dir, "data/sovereign/cum restrut 2014/US_SOV5YR_US_2015_2025.csv"),
    # "Leonardo SPA" : os.path.join(script_dir, "data/tech/mod-mod rec 2014/LEONARDO_SPA_ITA_2015_2025.csv"),
    # "Alstom": os.path.join(script_dir, "data/tech/mod-mod rec 2014/ALSTOM_FRA_2015_2025.csv"),
    # "Apple" : os.path.join(script_dir, "data/tech/APPLE_EX-RESTRUCT_US_2015_2025.csv"),
    "Mediobanca" : os.path.join(script_dir, "data/banks/baa1 below/mod-mod rec 2014/MEDIOBANCA_BANCA_SPA_ITA_2015_2025.csv"),
    # "UBS": os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/UBS_CH_2015_2025.csv"),
    # "Credit Agricole" : os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/CREDIT_AGRICOLE_FR_2015_2025.csv"),
    # "Banco Bilbao" : os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/BANCO_BILBAO_ES_2015_2025.csv"),
    # "Barclays" : os.path.join(script_dir, "data/banks/baa1 above/mod-mod rec 2014/BARCLAYS_GB_2015_2025.csv"),
    # "Mercedes Benz": os.path.join(script_dir, "data/automobile/baa1 above/mod-mod rec 2014/MERCEDES_BENZ_DE_2015_2025.csv"),
    # "Greece": os.path.join(script_dir, "data/sovereign/cum restrut 2014/GREECE_2015_2025.csv"),
    # "Turkey": os.path.join(script_dir, "data/sovereign/cum restrut 2014/TURKEY_2015_2025.csv")

}

def compute_pseudo_observations(data):
    """
    Converts raw data to pseudo-observations (ranks normalized to (0,1)),
    needed for copula GOF tests.
    """
    ranks = np.array([rankdata(col, method='average') for col in data.T]).T
    pseudo_obs = ranks / (data.shape[0] + 1)
    return pseudo_obs

def plot_heatmap(matrix,title,cmap, cbk):
    n=matrix.shape[0]
    fig,ax=plt.subplots(figsize=(n*0.3+4,n*0.3+4))
    annot_kws={'size':max(6,int(200/(n*n)))}
    sns.heatmap(matrix,annot=True,fmt='.2f',cmap=cmap,square=True,
                annot_kws=annot_kws,cbar_kws=cbk)
    plt.title(title); plt.xticks(rotation=45,ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()

def evaluate_copula_dependence_extended(paths_array, original_corr_matrix, dof=5, show_plots=True):
    """
    Extended copula evaluation: compares dependence structure, validates tail events,
    adds GOF tests and comparison to Gaussian copula.
    """
    n_paths, T, n_names = paths_array.shape
    paths_reshaped = paths_array[:, 1:, :].reshape(-1, n_names)
    returns = np.diff(paths_array, axis=1).reshape(-1, n_names)

    # === CORRELATION STRUCTURE ===
    simulated_corr_matrix = np.corrcoef(returns.T)
    frob_error = np.linalg.norm(simulated_corr_matrix - original_corr_matrix)

    # === TAIL DEPENDENCE: JOINT EXTREMES ===
    thresholds = np.percentile(paths_reshaped, 95, axis=0)
    joint_extremes = np.all(paths_reshaped > thresholds, axis=1)
    joint_extreme_count = np.sum(joint_extremes)
    joint_extreme_frequency = joint_extreme_count / len(paths_reshaped)

    # TODO: Not working right now
    # === Q-Q PLOTS of pseudo-observations vs t/Gaussian copula ===
    # pseudo_obs = compute_pseudo_observations(returns)

    # === GOF: KS test for uniformity of marginals ===
    # ks_stats = [kstest(pseudo_obs[:, i], 'uniform') for i in range(n_names)]
    # ks_results = {f"CDS_{i+1}": {'statistic': stat.statistic, 'p_value': stat.pvalue} for i, stat in enumerate(ks_stats)}

    # === Visuals ===
    if show_plots:
        plot_heatmap(original_corr_matrix, "Empirical Correlation Matrix (Historical)", cmap='Blues', cbk={'label': {'label': 'Empirical Corr'}})
        plot_heatmap(simulated_corr_matrix, "Simulated Correlation Matrix (From Copula Spreads)", cmap='Reds',
                     cbk={'label': 'Simulated Corr'})

        # TODO: Not working right now
        # # Q-Q Plot for pseudo observations
        # fig, axs = plt.subplots(1, n_names, figsize=(4 * n_names, 4))
        # for i in range(n_names):
        #     ecdf = ECDF(pseudo_obs[:, i])
        #     uniform_q = np.linspace(0, 1, 100)
        #     axs[i].plot(uniform_q, ecdf(uniform_q), label='Empirical CDF')
        #     axs[i].plot(uniform_q, uniform_q, 'r--', label='Uniform CDF')
        #     axs[i].set_title(f'Pseudo-Obs CDF (CDS {i+1})')
        #     axs[i].legend(loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
        # plt.suptitle("Empirical vs Uniform CDFs of Pseudo-Observations")
        # plt.tight_layout()
        # plt.show()

    return {
        "frob_error": frob_error,
        "simulated_corr_matrix": simulated_corr_matrix,
        "joint_extreme_frequency": joint_extreme_frequency,
        "joint_extreme_count": joint_extreme_count,
        # "ks_test_uniformity": ks_results,
    }

# ===================== EXECUTE MULTI-CDS SPREAD SIMULATION ===================== #
# Initialize the multi-CDS simulator
simulator = MultiCDSSimulator(name_to_series_map=historical_data, nu=len(historical_data))
# paths_array, paths_df = simulator.simulate_paths(n_paths=100)
paths_array, paths_df = simulator.simulate_paths_multithreaded(n_paths=100)

print(f'Shape of Monte Carlo matrix: {paths_array.shape}')
# ========== Plot 5: A few scenarios ==========
cds_names = simulator.names
scenarios = 5

for pfad in range(scenarios):
    plt.figure(figsize=(12, 6))
    for name in cds_names:
        series = paths_df.loc[pfad][name].values  # Zeitreihe dieses Namens für Szenario `pfad`
        plt.plot(series, label=name)
    plt.title(f'Scenario {pfad + 1}: CDS Spread Simulations')
    plt.xlabel('Days')
    plt.ylabel('CDS Spread (bps)')
    plt.legend(loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== Plot: Average spread per day for each CDS ==========

# Mittelwert über alle Pfade (Achse 0), bleibt (T+1, n_CDS)
avg_spreads = paths_array.mean(axis=0)  # Shape: (253, 5)

# x-Achse: Zeitpunkte (0 bis 252)
days = np.arange(avg_spreads.shape[0])

plt.figure(figsize=(12, 6))
for i, name in enumerate(simulator.names):
    plt.plot(days, avg_spreads[:, i], label=name)

plt.title("Average spread per day for each CDS")
plt.xlabel("Days")
plt.ylabel("CDS Spread (bps)")
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1.02,1),borderaxespad=0)
plt.tight_layout()
plt.show()

results = evaluate_copula_dependence_extended(
    paths_array=paths_array,
    original_corr_matrix=simulator.corr_matrix,
    dof=5,
    show_plots=True
)
print(results)

# CDS spreads: numpy array of shape (n_scenarios, T, 5) with simulated daily spreads for 5 names
cpdo = CPDO(initial_nav=100, coupon_rate=0.05, cushion_rate=0.01, recovery_rate=0.4,
            max_leverage=15, cash_out_frac=0.05, time_horizon_years=10, coupon_freq=2)
results = cpdo.simulate(spread_paths=paths_array)
print("Probability of Default:", results["PD"])
print("Expected Loss:", results["EL"])
print("99% VaR of loss:", results["VaR99"])
print("Implied Rating:", results["rating"])

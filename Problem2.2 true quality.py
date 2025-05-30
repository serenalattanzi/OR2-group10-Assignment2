#Required libraries
import numpy as np  # Numerical operations and array handling
import pandas as pd  # DataFrame handling and reading Excel files
from scipy.stats import truncnorm  # To generate truncated normal distributions
from tqdm import tqdm  # Progress bar for loops
import matplotlib.pyplot as plt  # Plotting and visualization

#DATA INPUT

path = "C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/means.xlsx"
#path = "C:/Users/alilo\OneDrive - University of Twente/1 ANNO/quartile 4/means.xlsx"
means_df = pd.read_excel(path)
means_df.head() # Visualize first rows
print(means_df.head(10)) 

# Given parameters
δ = 5                   # δ: maximum energy flow per time step [kWh]
RC = 50                 # RC: battery capacity [kWh]
T = 97                  # T: number of time steps per day (15-minute intervals)

# Means
μ_L = means_df["load"].to_numpy()        # μ_L[t] = average load at time t
μ_E = means_df["generation"].to_numpy()  # μ_E[t] = average generation at time t
μ_P = means_df["price"].to_numpy()       # μ_P[t] = average market price at time t

#Random Number Generator
MasterRNG = np.random.default_rng(seed=1)  #For reproducible stream of seeds

# Decision Rule Thresholds
# Definition of the 27 alternatives
alts = [(γ1, γ2) for γ1 in range(22, 27) for γ2 in range(25, 31) if γ1 < γ2]

#γ1 = 23                 # γ₁: price threshold to start buying (from market or storing excess generation)
#γ2 = 26                 # γ₂: price threshold to start selling (from battery to market)

#Samples creation
def samples_matrix(N, μ_L, μ_E, μ_P, δ, RC, T=97):
    rng = np.random.default_rng(seed=1)

    #Generation
    μ_E_mat = np.tile(μ_E, (N, 1))
    E_std = np.sqrt(0.0625)
    E_low = μ_E_mat - 0.5
    E_high = μ_E_mat + 0.5
    E_t_all = truncnorm.rvs((E_low - μ_E_mat) / E_std, (E_high - μ_E_mat) / E_std,
                            loc=μ_E_mat, scale=E_std, size=(N, T), random_state=rng)
    #Load
    μ_L_mat = np.tile(μ_L, (N, 1))
    L_std = np.sqrt(0.625)
    L_low = μ_L_mat - 3.75
    L_high = μ_L_mat + 3.75
    L_t_all = truncnorm.rvs((L_low - μ_L_mat) / L_std, (L_high - μ_L_mat) / L_std,
                            loc=μ_L_mat, scale=L_std, size=(N, T), random_state=rng)

    #Price
    μ_P_mat = np.tile(μ_P, (N, 1))
    mix = rng.uniform(size=(N, T)) < 0.1
    P_std = np.where(mix, np.sqrt(10000), np.sqrt(10))
    P_low = np.where(mix, μ_P_mat - 25, μ_P_mat - 50)
    P_high = np.where(mix, μ_P_mat + 200, μ_P_mat + 50)

    P_t_all = truncnorm.rvs(
        (P_low - μ_P_mat) / P_std,
        (P_high - μ_P_mat) / P_std,
        loc=μ_P_mat, scale=P_std,
        size=(N, T), random_state=rng
    )
    return E_t_all, L_t_all, P_t_all

#Simulate one day
def simulate_one_day(E_t, L_t, P_t, γ1, γ2, δ, RC):
    R = 25
    C = 0
    T = E_t.shape[0]

    for t in range(T):
        E_raw = E_t[t]
        L = L_t[t]
        P = P_t[t]

        E_plus = max(0, E_raw)
        E_minus = -min(0, E_raw)
        dont_charge = R >= (96 - t) * δ 

        xEL = min(E_plus, L + E_minus)
        xRL = min(L + E_minus - xEL, R, δ) if P >= γ2 else 0
        xML = L + E_minus - xEL - xRL
        xER = min(E_plus - xEL, δ, RC - R)

        if P <= γ1 and not dont_charge:
            xMR = min(RC - R - xER, δ - xER)
        elif dont_charge and P < 0:
            xMR = min(RC - R - xER, δ - xER)
        else:
            xMR = 0

        if dont_charge and P > 0:
            xRM = min(δ - xRL, R - xRL)
        elif P >= γ2:
            xRM = min(δ - xRL, R - xRL)
        else:
            xRM = 0
        R = R + xER + xMR - xRL - xRM
        R = min(RC, max(0, R))  

        c = P * (xRM + L) - P * (xML + xMR)
        C += c

    return C

#Simulate many days
def simulate_many_days(E_all, L_all, P_all, γ1, γ2, δ=5, RC=50):
    N = E_all.shape[0]
    profits = []

    for i in tqdm(range(N), desc=f"Simulating for γ=({γ1},{γ2})", unit="day", dynamic_ncols=True): #Show progress bar
        profit = simulate_one_day(E_all[i], L_all[i], P_all[i], γ1, γ2, δ, RC)
        profits.append(profit)

    mean_profit = np.mean(profits)
    variance_profit = np.var(profits, ddof=1)
    return mean_profit, variance_profit


def evaluate_all_alternatives(E_all, L_all, P_all, δ=5, RC=50):
    
    results = []
    alternatives = [(γ1, γ2) for γ1 in range(22, 27) for γ2 in range(25, 31) if γ1 < γ2]

    for γ1, γ2 in tqdm(alternatives, desc="Evaluating all alternatives"):
        mean_profit, variance_profit = simulate_many_days(E_all, L_all, P_all, γ1, γ2, δ, RC)
        results.append((γ1, γ2, mean_profit, variance_profit))

    df = pd.DataFrame(results, columns=["γ1", "γ2", "mean_profit", "variance_profit"])
    return df


#Running
N = 1000000  

E_all, L_all, P_all = samples_matrix(N, μ_L, μ_E, μ_P, δ, RC, T)

results_df = evaluate_all_alternatives(E_all, L_all, P_all, δ, RC)

# Save results to CSV
results_df.to_csv("true_quality_table.csv", index=False)

# Show top results
print(results_df.sort_values("mean_profit", ascending=False).head())

#Required libraries
import numpy as np  # Numerical operations and array handling
import pandas as pd  # DataFrame handling and reading Excel files
import openpyxl  # Excel file engine used by pandas for reading .xlsx files
from scipy.stats import truncnorm  # To generate truncated normal distributions
from tqdm import tqdm  # Progress bar for loops (e.g., simulations)
import matplotlib.pyplot as plt  # Plotting and visualization (optional but useful)

# Insert file to the excel file here
# Serena
path = "C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/means.xlsx"
# Alice
#path = "C:/Users/alilo\OneDrive - University of Twente/1 ANNO/quartile 4/means.xlsx"

# Upload data
means_df = pd.read_excel(path)

print ("loading code")

# Visualize first rows
means_df.head()
print(means_df.head(10)) 
# Fixed problem parameters
# Decision Rule Thresholds
γ1 = 23                 # γ₁: price threshold to start buying (from market or storing excess generation)
γ2 = 26                 # γ₂: price threshold to start selling (from battery to market)

#FLOW VARIABLES (computed during simulation)

# R     → battery level at time t [MWh]

# P_t   → price at time t [€/MWh]
# L_t   → load (demand) at time t [MWh]
# E_t   → solar generation at time t [MWh]

# xEL   → solar to load         (E → L)
# xRL   → battery to load       (R → L)
# xML   → market to load        (M → L)
# xER   → solar to battery      (E → R)
# xMR   → market to battery     (M → R)
# xRM   → battery to market     (R → M)

# c_t   → profit at time t [€]
# C     → total daily profit [€]

# Given parameters
δ = 5                   # δ: maximum energy flow per time step [kWh]
RC = 50                 # RC: battery capacity [kWh]
T = 97                  # T: number of time steps per day (15-minute intervals)

# Means
μ_L = means_df["load"].to_numpy()        # μ_L[t] = average load at time t
μ_E = means_df["generation"].to_numpy()  # μ_E[t] = average generation at time t
μ_P = means_df["price"].to_numpy()       # μ_P[t] = average market price at time t

#Random Number Generator
MasterRNG = np.random.default_rng(seed=1)  # For reproducible stream of seeds

#Truncated Normal Sampler
def sample_truncated_normal(mean, std, lower, upper, rng):
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng)


def simulate_many_days_vectorized(N, γ1, γ2, μ_L, μ_E, μ_P, δ=5, RC=50, T=97):
    rng = np.random.default_rng(seed=1)

    # Generation
    μ_E_mat = np.tile(μ_E, (N, 1))
    E_std = np.sqrt(0.0625)
    E_low = μ_E_mat - 0.5
    E_high = μ_E_mat + 0.5
    E_t_all = truncnorm.rvs((E_low - μ_E_mat) / E_std, (E_high - μ_E_mat) / E_std,
                            loc=μ_E_mat, scale=E_std, size=(N, T), random_state=rng)
    # Load
    μ_L_mat = np.tile(μ_L, (N, 1))
    L_std = np.sqrt(0.625)
    L_low = μ_L_mat - 3.75
    L_high = μ_L_mat + 3.75
    L_t_all = truncnorm.rvs((L_low - μ_L_mat) / L_std, (L_high - μ_L_mat) / L_std,
                            loc=μ_L_mat, scale=L_std, size=(N, T), random_state=rng)

    # Price
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

# Simulate one day
def simulate_one_day(E_t, L_t, P_t, γ1, γ2, δ=5, RC=50):
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
        R = min(RC, max(0, R))  # Clamp battery to [0, RC]

        c = P * (xRM + L) - P * (xML + xMR)
        C += c

    return C


def simulate_many_days(E_all, L_all, P_all, γ1, γ2, δ=5, RC=50):
    N = E_all.shape[0]
    profits = []

    for i in tqdm(range(N), desc="Simulating", unit="day", dynamic_ncols=True):
        profit = simulate_one_day(E_all[i], L_all[i], P_all[i], γ1, γ2, δ, RC)
        profits.append(profit)

    mean_profit = np.mean(profits)
    variance_profit = np.var(profits, ddof=1)
    return mean_profit, variance_profit, profits


#Running
N = 1000000  

E_all, L_all, P_all = simulate_many_days_vectorized(N, γ1, γ2, μ_L, μ_E, μ_P)
mean_profit, variance_profit, profits = simulate_many_days(E_all, L_all, P_all, γ1, γ2)

print("Expected daily profit:", mean_profit)
print("Sample variance of profit:", variance_profit)


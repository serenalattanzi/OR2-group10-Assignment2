#Required libraries
import numpy as np  # Numerical operations and array handling
import pandas as pd  # DataFrame handling and reading Excel files
import openpyxl  # Excel file engine used by pandas for reading .xlsx files
from scipy.stats import truncnorm  # To generate truncated normal distributions
from tqdm import tqdm  # Progress bar for loops (e.g., simulations)
import matplotlib.pyplot as plt  # Plotting and visualization (optional but useful)

path = "C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/means.xlsx"

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


#Simulate One Day
def simulate_one_day(γ1, γ2, μ_L, μ_E, μ_P, rng_L, rng_E, rng_P):
    R = 25  # Battery level
    C = 0  # Total profit

    for t in range(T):

        #Generated energy
        μ_E_t = μ_E[t]
        σ_E_t = np.sqrt(0.0625)
        a_E = μ_E_t - 0.5
        b_E = μ_E_t + 0.5
        E_t_raw = sample_truncated_normal(μ_E_t, σ_E_t, a_E, b_E, rng_E)
        E_t_plus = max(0, E_t_raw)   # E_t^+ → usable solar generation
        E_t_minus = -min(0, E_t_raw) # E_t^- → consumption by solar plant

        #Load
        μ_L_t = μ_L[t]
        σ_L_t = np.sqrt(0.625)
        a_L = μ_L_t - 3.75
        b_L = μ_L_t + 3.75
        L_t = sample_truncated_normal(μ_L_t, σ_L_t, a_L, b_L, rng_L)

        #Market price
        μ_P_t = μ_P[t]
        u = rng_P.uniform()
        if u < 0.1:
            σ_P_t = np.sqrt(10000)
            a_P = μ_P_t - 25
            b_P = μ_P_t + 200
        else:
            σ_P_t = np.sqrt(10)
            a_P = μ_P_t - 50
            b_P = μ_P_t + 50
        P_t = sample_truncated_normal(μ_P_t, σ_P_t, a_P, b_P, rng_P)

        # Decision variables
        # Solar → Load
        xEL = min(E_t_plus, L_t + E_t_minus)

        # Battery → Load
        if P_t >= γ2:
            xRL = min(L_t + E_t_minus - xEL, R, δ)
        else:
            xRL = 0

        # Market → Load
        xML = L_t + E_t_minus - xEL - xRL

        # Only charge battery if price is low and we have enough time
        dont_charge = R >= (96 - t) * δ  # True if charging still useful

        # Solar → Battery
        xER = min(E_t_plus - xEL, δ, RC - R)

        # Market → Battery
        if P_t <= γ1 and not dont_charge:
                xMR = min(RC - R - xER, δ - xER)
        elif dont_charge and P_t <0:  #
                xMR = min(RC - R - xER, δ - xER)
        else:
            xMR = 0

        # Battery → Market
        if dont_charge and P_t > 0:
            xRM = min(δ - xRL, R - xRL)
        elif P_t >= γ2:
            xRM = min(δ - xRL, R - xRL)
        else:
            xRM = 0

        #  Update battery level
        R = R + xER + xMR - xRL - xRM
        R = min(RC, max(0, R))  # Clamp battery to [0, RC]

        #  Profit calculation
        c_t = P_t * (xRM + L_t) - P_t * (xML + xMR)
        C += c_t

    return C

#Simulate many days
def simulate_many_days(N, γ1, γ2, μ_L, μ_E, μ_P):
    profits = []

    for _ in range(N):
        seed_P = MasterRNG.integers(1, 1_000_000)
        seed_E = MasterRNG.integers(1, 1_000_000)
        seed_L = MasterRNG.integers(1, 1_000_000)

        rng_P = np.random.default_rng(seed_P)
        rng_E = np.random.default_rng(seed_E)
        rng_L = np.random.default_rng(seed_L)

        profit = simulate_one_day(γ1, γ2, μ_L, μ_E, μ_P, rng_L, rng_E, rng_P)
        profits.append(profit)

    mean_profit = np.mean(profits)
    variance_profit = np.var(profits, ddof=1)  # sample variance
    return mean_profit, variance_profit


#Running
N = 100  # Use 1000000 later
expected_profit, sample_variance = simulate_many_days(N, γ1, γ2, μ_L, μ_E, μ_P)
print("Expected daily profit:", expected_profit)
print("Sample variance of profit:", sample_variance)




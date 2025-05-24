#Required libraries
import numpy as np  # Numerical operations and array handling
import pandas as pd  # DataFrame handling and reading Excel files
from scipy.stats import truncnorm  # To generate truncated normal distributions
from tqdm import tqdm  # Progress bar for loops
import matplotlib.pyplot as plt  # Plotting and visualization

#DATA INPUT
# Insert file to the excel file here
# Serena
path = "C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/means.xlsx"
# Alice
#path = "C:/Users/alilo\OneDrive - University of Twente/1 ANNO/quartile 4/means.xlsx"
means_df = pd.read_excel(path)
means_df.head() # Visualize first rows
print(means_df.head(10)) 

#FLOW VARIABLES
#R   → battery level at time t [MWh]

#P   → price at time t [€/MWh]
#L   → load (demand) at time t [MWh]
#E   → solar generation at time t [MWh]

#xEL → solar to load      (E → L)
#xRL → battery to load    (R → L)
#xML → market to load     (M → L)
#xER → solar to battery   (E → R)
#xMR → market to battery  (M → R)
#xRM → battery to market  (R → M)

#c_t → profit at time t [€]
#C   → total daily profit [€]   

# Given parameters
δ = 5    # Maximum energy flow per time step [kWh]
RC = 50  # Battery capacity [kWh]
T = 97   # Number of time steps per day (15-minute intervals)
#Running
M = 100  # Number of experiments
N = 500  # Number of days to simulate

# Means
μ_L = means_df["load"].to_numpy()        # μ_L[t] = average load at time t
μ_E = means_df["generation"].to_numpy()  # μ_E[t] = average generation at time t
μ_P = means_df["price"].to_numpy()       # μ_P[t] = average market price at time t        

#Random Number Generator
MasterRNG = np.random.default_rng(seed=1)  #For reproducible stream of seeds

#Truncated Normal Sampler
def sample_truncated_normal(mean, std, lower, upper, rng):
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng)

#Parameters sample creation
def sample_parameters(M, N, μ_L, μ_E, μ_P, rng, T=97):

    #Generation
    μ_E_mat = np.tile(μ_E, (M, N, 1))
    E_std = np.sqrt(0.0625)
    E_low = μ_E_mat - 0.5
    E_high = μ_E_mat + 0.5
    E_t_all = truncnorm.rvs((E_low - μ_E_mat) / E_std, (E_high - μ_E_mat) / E_std,
                            loc=μ_E_mat, scale=E_std, size=(M, N, T), random_state=rng)
    #Load
    μ_L_mat = np.tile(μ_L, (M, N, 1))
    L_std = np.sqrt(0.625)
    L_low = μ_L_mat - 3.75
    L_high = μ_L_mat + 3.75
    L_t_all = truncnorm.rvs((L_low - μ_L_mat) / L_std, (L_high - μ_L_mat) / L_std,
                            loc=μ_L_mat, scale=L_std, size=(M, N, T), random_state=rng)

    #Price
    μ_P_mat = np.tile(μ_P, (M, N, 1))
    mix = rng.uniform(size=(M, N, T)) < 0.1
    P_std = np.where(mix, np.sqrt(10000), np.sqrt(10))
    P_low = np.where(mix, μ_P_mat - 25, μ_P_mat - 50)
    P_high = np.where(mix, μ_P_mat + 200, μ_P_mat + 50)

    P_t_all = truncnorm.rvs(
        (P_low - μ_P_mat) / P_std,
        (P_high - μ_P_mat) / P_std,
        loc=μ_P_mat, scale=P_std,
        size=(M, N, T), random_state=rng)
    
    return E_t_all, L_t_all, P_t_all

# Decision Rule Thresholds
Y = [(γ1, γ2) for γ1 in range(22, 27) for γ2 in range(25, 31) if γ1 < γ2]
K = len(Y)  # Total number of alternatives (27)

#  Load true qualities from Excel 
true_df = pd.read_excel("C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/true_qualities.xlsx")

true_df.set_index(['gamma 1', 'gamma 2'], inplace=True)

# Get ordered true quality and variance arrays matching Y
true_quality = np.array([true_df.loc[(γ1, γ2), 'mean'] for (γ1, γ2) in Y])
true_variance = np.array([true_df.loc[(γ1, γ2), 'variance'] for (γ1, γ2) in Y])

#Simulate one day
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
        R = min(RC, max(0, R))  

        c = P * (xRM + L) - P * (xML + xMR)
        C += c

    return C

def simulate_policy(policy, M, N, E_all, L_all, P_all, δ=5, RC=50):
   
    # prior bliefs
    μ_0 = np.full(K, 5500.0)      # prior mean
    var_0 = np.full(K, 1200.0**2) # prior variance
    # n_obs = np.zeros(K)           # number of observations for each alternative

    # Known variances
    σ_w = 1200 # Constant observation variance for all alternatives
    var_w = σ_w**2  

    # storage
    quality_matrix = np.zeros((M, N))
    precision_matrix = np.zeros((M, N))

    # Epsilon for epsilon-greedy policy
    ε0 = 0.95

    # Run M experiments

    for m in tqdm(range(M), desc=policy):
        μ = μ_0.copy()
        var = var_0.copy()

        for n in range(N):
            if policy == "exploration":
                choice = np.random.randint(K)
            elif policy == "exploitation":
                choice = np.argmax(μ)
            elif policy == "ε_greedy":
                ε = ε0 * (1 - n / N)
                choice = np.random.randint(K) if np.random.random() < ε else np.argmax(μ)
            elif policy == "kg":
                kg_bonus = np.sqrt(var) * (1200 / (1200 + np.sqrt(var)))
                choice = np.argmax(μ + kg_bonus)
            else:
                raise ValueError("Unknown policy")

            γ1, γ2 = Y[choice]
            profit = simulate_one_day(E_all[m, n], L_all[m, n], P_all[m, n], γ1, γ2)

            μ[choice] = (μ[choice] * var_w + profit * var[choice]) / (var[choice] + var_w)
            var[choice] = (var[choice] * var_w) / (var[choice] + var_w)

            best = np.argmax(μ)
            quality_matrix[m, n] = true_quality[best]

    return quality_matrix

# Run the simulation for all policies

E_all, L_all, P_all = sample_parameters(M, N, μ_L, μ_E, μ_P, rng=MasterRNG, T=T)

results = {}
for policy in ["exploration", "exploitation", "ε_greedy", "kg"]:
    results[policy] = simulate_policy(policy, M, N, E_all, L_all, P_all)

plt.figure(figsize=(12, 6))
for policy, matrix in results.items():
    avg_curve = matrix.mean(axis=0)
    plt.plot(avg_curve, label=policy.replace("_", " ").title())

plt.title("Offline Learning — Average True Quality of Best Belief")
plt.xlabel("Day")
plt.ylabel("True Quality")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
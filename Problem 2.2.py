#Required libraries
import numpy as np
import pandas as pd  
from scipy.stats import truncnorm, norm  
from tqdm import tqdm  
import matplotlib.pyplot as plt 

#Data Input
path = "C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/means.xlsx"
#path = "C:/Users/alilo/OneDrive - University of Twente/1 ANNO/quartile 4/means.xlsx"
means_df = pd.read_excel(path)
means_df.head() # Visualize first rows
print(means_df.head(10)) 
# Means
μ_L = means_df["load"].to_numpy()        # μ_L[t] = average load at time t
μ_E = means_df["generation"].to_numpy()  # μ_E[t] = average generation at time t
μ_P = means_df["price"].to_numpy()       # μ_P[t] = average market price at time t        

# Decision Rule Thresholds
Y = [(γ1, γ2) for γ1 in range(22, 27) for γ2 in range(25, 31) if γ1 < γ2]
K = len(Y)  # Total number of alternatives (27)

# True qualities input 
true_df = pd.read_excel("C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/true_qualities.xlsx")
#true_df = pd.read_excel("C:/Users/alilo/OneDrive - University of Twente/1 ANNO/quartile 4/true_qualities.xlsx")
true_df.set_index(['gamma 1', 'gamma 2'], inplace=True)
true_quality = np.array([true_df.loc[(γ1, γ2), 'mean'] for (γ1, γ2) in Y])

# Given parameters
δ = 5  # Maximum energy flow per time step [kWh]
RC = 50  # Battery capacity [kWh]
T = 97  # Number of time steps per day (15-minute intervals)

# Parameters for the policies
σ_w = 1200 
var_w = σ_w**2  # Constant observation variance for all alternatives
ε0 = 0.95  # Epsilon for epsilon-greedy policy

# Running Parameters
M = 100  # Number of experiments
N = 500  # Number of days to simulate

#Parameters sample creation
def sample_parameters(M, N, μ_L, μ_E, μ_P, rng, T=97):
    #Generation
    μ_E_mat = np.tile(μ_E, (M, N, 1))
    E_std = np.sqrt(0.0625)
    E_low = μ_E_mat - 0.5
    E_high = μ_E_mat + 0.5
    E_t_all = truncnorm.rvs((E_low - μ_E_mat) / E_std, (E_high - μ_E_mat) / E_std, loc=μ_E_mat, scale=E_std, size=(M, N, T), random_state=rng)
    
    #Load
    μ_L_mat = np.tile(μ_L, (M, N, 1))
    L_std = np.sqrt(0.625)
    L_low = μ_L_mat - 3.75
    L_high = μ_L_mat + 3.75
    L_t_all = truncnorm.rvs((L_low - μ_L_mat) / L_std, (L_high - μ_L_mat) / L_std, loc=μ_L_mat, scale=L_std, size=(M, N, T), random_state=rng)

    #Price
    μ_P_mat = np.tile(μ_P, (M, N, 1))
    mix = rng.uniform(size=(M, N, T)) < 0.1
    P_std = np.where(mix, np.sqrt(10000), np.sqrt(10))
    P_low = np.where(mix, μ_P_mat - 25, μ_P_mat - 50)
    P_high = np.where(mix, μ_P_mat + 200, μ_P_mat + 50)
    P_t_all = truncnorm.rvs((P_low - μ_P_mat) / P_std, (P_high - μ_P_mat) / P_std, loc=μ_P_mat, scale=P_std, size=(M, N, T), random_state=rng)
    
    return E_t_all, L_t_all, P_t_all

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

# Simulate policies offline
def simulate_policy_offline(policy, M, N, E_all, L_all, P_all, δ=5, RC=50):
    # Storage
    quality_matrix = np.zeros((M, N))

    # Run M experiments
    for m in tqdm(range(M), desc=policy):

        # Initial beliefs
        μ = np.full(K, 5500.0) # Prior mean
        var = np.full(K, 1200.0**2) # Prior variance
        rng = np.random.default_rng(seeds[m])  # New seed for each experiment
    
        for n in range(N):
            if policy == "exploration":
                choice = np.random.randint(K)
            elif policy == "exploitation":
                max_μ = np.max(μ)
                candidates = np.flatnonzero(μ == max_μ)  # Indices of alternatives with max mean
                choice = rng.choice(candidates)
            elif policy == "ε_greedy":
                c = 0.95  
                ε = c / (n + 1)  # +1 to avoid div by zero
                if rng.random() < ε:
                    choice = rng.integers(K)  # Explore
                else:
                   max_μ = np.max(μ)
                   candidates = np.flatnonzero(μ == max_μ) # Exploit
                   choice = rng.choice(candidates)
            elif policy == "kg":
                β = 1 / var
                β_w = 1 / var_w 
                var_tilde = (1 / β) - (1 / (β + β_w))
                σ_tilde = np.sqrt(var_tilde)
              
                μ_matrix = np.tile(μ, (K, 1))  # Matrix of means
                np.fill_diagonal(μ_matrix, -np.inf) # Fill diagonal with -inf to ignore self-comparison
                μ_star = μ_matrix.max(axis=1)  # vector of max for j ≠ i
               
                ζ = -np.abs((μ - μ_star) / σ_tilde) 
                f_ζ = ζ * norm.cdf(ζ) + norm.pdf(ζ)
                ν_kg = σ_tilde * f_ζ
                
                candidates = np.flatnonzero(ν_kg == np.max(ν_kg)) # Choose indices where ν_kg is maximum
                choice = rng.choice(candidates) # Randomly select if multiple candidates have the same score

            else:
                raise ValueError("Unknown policy")

            γ1, γ2 = Y[choice]
            profit = simulate_one_day(E_all[m, n], L_all[m, n], P_all[m, n], γ1, γ2)

            μ[choice] = (μ[choice] * var_w + profit * var[choice]) / (var[choice] + var_w)
            var[choice] = (var[choice] * var_w) / (var[choice] + var_w)

            best = np.argmax(μ)
            quality_matrix[m, n] = true_quality[best]

    return quality_matrix

#Random Number Generator
MasterRNG = np.random.default_rng(seed=1) 
seeds = MasterRNG.integers(0, 1e9, size=M) 

E_all, L_all, P_all = sample_parameters(M, N, μ_L, μ_E, μ_P, rng=MasterRNG, T=T)

#Run the simulation for all policies
results = {}
for policy in ["exploration", "exploitation", "ε_greedy", "kg"]:
    results[policy] = simulate_policy_offline(policy, M, N, E_all, L_all, P_all)

# Graph for the results
plt.figure(figsize=(12, 6))
for policy, matrix in results.items():
    avg_curve = matrix.mean(axis=0)
    plt.plot(avg_curve, label=policy.replace("_", " ").title())

plt.title("Offline Learning")
plt.xlabel("Day")
plt.ylabel("Quality")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
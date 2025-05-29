import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from tqdm import tqdm

# General simulation settings
M = 100     # Number of experiments
N = 500     # Number of days per experiment
T = 97      # Number of time steps per day
δ = 5       # Max charge/discharge rate
RC = 50     # Battery capacity
σ_w = 1200  # Standard deviation of observation noise
var_w = σ_w ** 2

# Set of alternatives: 27 (γ₁, γ₂) pairs
Y = [(γ1, γ2) for γ1 in range(22, 27) for γ2 in range(25, 31) if γ1 < γ2]
K = len(Y)

# Load daily average profiles
#Serena
#means_df = pd.read_excel("C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/means.xlsx")
# Alice
means_df = pd.read_excel("C:/Users/alilo/OneDrive - University of Twente/1 ANNO/quartile 4/means.xlsx")

μ_L = means_df["load"].to_numpy()
μ_E = means_df["generation"].to_numpy()
μ_P = means_df["price"].to_numpy()

# Load true qualities
#true_df = pd.read_excel("C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/true_qualities.xlsx")
true_df = pd.read_excel("C:/Users/alilo/OneDrive - University of Twente/1 ANNO/quartile 4/true_qualities.xlsx")
true_df.set_index(['gamma 1', 'gamma 2'], inplace=True)
true_quality = np.array([true_df.loc[(γ1, γ2), 'mean'] for (γ1, γ2) in Y])

def sample_parameters(M, N, μ_L, μ_E, μ_P, rng, T=97):
    μ_E_mat = np.tile(μ_E, (M, N, 1))
    E_std = np.sqrt(0.0625)
    E_low = μ_E_mat - 0.5
    E_high = μ_E_mat + 0.5
    E = truncnorm.rvs((E_low - μ_E_mat) / E_std, (E_high - μ_E_mat) / E_std,
                      loc=μ_E_mat, scale=E_std, size=(M, N, T), random_state=rng)

    μ_L_mat = np.tile(μ_L, (M, N, 1))
    L_std = np.sqrt(0.625)
    L_low = μ_L_mat - 3.75
    L_high = μ_L_mat + 3.75
    L = truncnorm.rvs((L_low - μ_L_mat) / L_std, (L_high - μ_L_mat) / L_std,
                      loc=μ_L_mat, scale=L_std, size=(M, N, T), random_state=rng)

    μ_P_mat = np.tile(μ_P, (M, N, 1))
    mix = rng.uniform(size=(M, N, T)) < 0.1
    P_std = np.where(mix, np.sqrt(10000), np.sqrt(10))
    P_low = np.where(mix, μ_P_mat - 25, μ_P_mat - 50)
    P_high = np.where(mix, μ_P_mat + 200, μ_P_mat + 50)
    P = truncnorm.rvs((P_low - μ_P_mat) / P_std, (P_high - μ_P_mat) / P_std,
                      loc=μ_P_mat, scale=P_std, size=(M, N, T), random_state=rng)

    return E, L, P

def simulate_one_day(E_t, L_t, P_t, γ1, γ2, δ=5, RC=50):
    R = 25
    C = 0
    for t in range(len(E_t)):
        E_raw, L, P = E_t[t], L_t[t], P_t[t]
        E_plus, E_minus = max(0, E_raw), -min(0, E_raw)
        dont_charge = R >= (96 - t) * δ

        xEL = min(E_plus, L + E_minus)
        xRL = min(L + E_minus - xEL, R, δ) if P >= γ2 else 0
        xML = L + E_minus - xEL - xRL
        xER = min(E_plus - xEL, δ, RC - R)

        xMR = min(RC - R - xER, δ - xER) if P <= γ1 and not dont_charge else 0
        xRM = min(δ - xRL, R - xRL) if (dont_charge and P > 0) or (P >= γ2) else 0

        R = min(RC, max(0, R + xER + xMR - xRL - xRM))
        C += P * (xRM + L) - P * (xML + xMR)
    return C

def simulate_policy_online(policy, M, N, E_all, L_all, P_all, true_quality, seeds, δ=5, RC=50):
    ε0 = 0.95
    quality_matrix = np.zeros((M, N))
    from scipy.stats import norm  # import necessario per h(z)

    for m in tqdm(range(M), desc=f"{policy}"):
        μ = np.full(K, 5500.0)
        var = np.full(K, 1200.0**2)
        rng = np.random.default_rng(seeds[m])  # new seed for each experiment

        for n in range(N):
            if policy == "exploration":
                choice = rng.integers(K)
            elif policy == "exploitation":
                choice = np.argmax(μ)
            elif policy == "ε_greedy":
                # ε = ε0 * (1 - n / N)
                # choice = rng.integers(K) if rng.random() < ε else np.argmax(μ

                c = 0.95  # pick any constant in (0, 1)
                ε = c / (n + 1)  # +1 to avoid div by zero
                if rng.random() < ε:
                    choice = rng.integers(K)  # explore
                else:
                   choice = np.argmax(μ)  # exploit
            elif policy == "kg":
                total_std = np.sqrt(var + var_w)
                kg_values = np.zeros(K)
                for i in range(K):
                    # Find the next-best mean (excluding alternative i)
                    μ_others = np.delete(μ, i)
                    μ_star = np.max(μ_others)
                    z = np.abs(μ[i] - μ_star) / total_std[i] if total_std[i] > 0 else 0.0
                    h_z = norm.pdf(z) + z * (1 - norm.cdf(z))
                    kg_values[i] = h_z * total_std[i]
                choice = np.argmax(μ + kg_values)

            else:
                raise ValueError("Unknown policy")

            γ1, γ2 = Y[choice]
            profit = simulate_one_day(E_all[m, n], L_all[m, n], P_all[m, n], γ1, γ2, δ, RC)

            μ[choice] = (μ[choice] * var_w + profit * var[choice]) / (var[choice] + var_w)
            var[choice] = (var[choice] * var_w) / (var[choice] + var_w)

            quality_matrix[m, n] = true_quality[choice]  # ✅ log chosen alternative's quality

    return quality_matrix

# Set up controlled seed stream for reproducibility
rng_global = np.random.default_rng(seed=42)
seeds = rng_global.integers(0, 1e9, size=M)

# Generate shared random scenarios
E_all, L_all, P_all = sample_parameters(M, N, μ_L, μ_E, μ_P, rng=rng_global)

# Run and collect results
results_online = {}
for policy in ["exploration", "exploitation", "ε_greedy", "kg"]:
    results_online[policy] = simulate_policy_online(policy, M, N, E_all, L_all, P_all, true_quality, seeds)

# Plot
plt.figure(figsize=(12, 6))
for policy, matrix in results_online.items():
    avg_curve = matrix.mean(axis=0)
    plt.plot(avg_curve, label=policy.replace("_", " ").title())

plt.title("Online Learning — Avg. True Quality of Sampled Alternative")
plt.xlabel("Day")
plt.ylabel("True Quality of Action Taken")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

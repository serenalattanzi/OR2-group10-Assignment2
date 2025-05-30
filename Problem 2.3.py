import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
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
means_df = pd.read_excel("C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/means.xlsx")
# Alice
#means_df = pd.read_excel("C:/Users/alilo/OneDrive - University of Twente/1 ANNO/quartile 4/means.xlsx")
μ_L = means_df["load"].to_numpy()
μ_E = means_df["generation"].to_numpy()
μ_P = means_df["price"].to_numpy()

# Load true qualities
true_df = pd.read_excel("C:/Users/seren/OneDrive/Escritorio/OR2-group10-Assignment2/true_qualities.xlsx")
#true_df = pd.read_excel("C:/Users/alilo/OneDrive - University of Twente/1 ANNO/quartile 4/true_qualities.xlsx")
true_df.set_index(['gamma 1', 'gamma 2'], inplace=True)
true_quality = np.array([true_df.loc[(γ1, γ2), 'mean'] for (γ1, γ2) in Y])

# Sampling Parameters
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

def simulate_policy_online(policy, M, N, E_all, L_all, P_all, true_quality, seeds, δ=5, RC=50):
    ε0 = 0.95
    quality_matrix = np.zeros((M, N))

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
                # Step 1: Compute tilde_sigma_i
                beta = 1 / var
                beta_w = 1 / var_w
                sigma2_tilde = (1 / beta) - (1 / (beta + beta_w))
                tilde_sigma = np.sqrt(sigma2_tilde)

                # Step 2: Compute μ_star = max_{j ≠ i} μ_j for each i
                μ_matrix = np.tile(μ, (K, 1))  # shape (K, K)
                np.fill_diagonal(μ_matrix, -np.inf)
                μ_star = μ_matrix.max(axis=1)

                # Step 3: ζ_i = -|μ_i - μ_star| / tilde_sigma_i
                zeta = -np.abs(μ - μ_star) / tilde_sigma

                # Step 4: f(ζ) = ζ Φ(ζ) + φ(ζ)
                f_zeta = zeta * norm.cdf(zeta) + norm.pdf(zeta)

                # Step 5: ν_KG = tilde_sigma * f(ζ)
                ν_kg = tilde_sigma * f_zeta

                # Step 6: OKG Score
                score = μ + (N - n) * ν_kg

                # Step 7: Choose best with tie-breaking
                max_score = np.max(score)
                best_candidates = np.flatnonzero(score == max_score)
                choice = rng.choice(best_candidates)

            else:
                raise ValueError("Unknown policy")

            γ1, γ2 = Y[choice]
            profit = simulate_one_day(E_all[m, n], L_all[m, n], P_all[m, n], γ1, γ2, δ, RC)

            μ[choice] = (μ[choice] * var_w + profit * var[choice]) / (var[choice] + var_w)
            var[choice] = (var[choice] * var_w) / (var[choice] + var_w)

            quality_matrix[m, n] = true_quality[choice]

    return quality_matrix

# Set up seeds for reproducibility
MasterRNG = np.random.default_rng(seed=1)
seeds = MasterRNG.integers(0, 1e9, size=M)

E_all, L_all, P_all = sample_parameters(M, N, μ_L, μ_E, μ_P, rng=MasterRNG)

# Run the simulation for all policies
results_online = {}
for policy in ["exploration", "exploitation", "ε_greedy", "kg"]:
    results_online[policy] = simulate_policy_online(policy, M, N, E_all, L_all, P_all, true_quality, seeds)

# Graph for the results
plt.figure(figsize=(12, 6))
for policy, matrix in results_online.items():
    avg_curve = matrix.mean(axis=0)
    plt.plot(avg_curve, label=policy.replace("_", " ").title())

plt.title("Online Learning")
plt.xlabel("Day")
plt.ylabel("Quality")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

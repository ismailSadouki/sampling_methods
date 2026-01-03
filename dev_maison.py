import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(2026)

N = 1000
population = pd.DataFrame({
    "id": np.arange(1, N+1),
    "income": np.random.normal(50000, 10000, N)
})

pop_mean = population["income"].mean()
pop_total = population["income"].sum()
print(f"True population mean: {pop_mean:.2f}")
print(f"True population total: {pop_total:.2f}\n")

n = 100
srs_sample = population.sample(n=n, random_state=123)
srs_mean = srs_sample["income"].mean()
srs_total = srs_sample["income"].sum() * (N/n)
srs_se = srs_sample["income"].std(ddof=1)/np.sqrt(n)
srs_ci = stats.t.interval(0.95, df=n-1, loc=srs_mean, scale=srs_se)

k = N // n
start = np.random.randint(0, k)
sys_indices = np.arange(start, N, k)
sys_sample = population.iloc[sys_indices]
sys_mean = sys_sample["income"].mean()
sys_total = sys_sample["income"].sum() * (N/n)
sys_se = sys_sample["income"].std(ddof=1)/np.sqrt(n)
sys_ci = stats.t.interval(0.95, df=n-1, loc=sys_mean, scale=sys_se)
population["strata"] = np.where(population["income"] < 50000, "low", "high")
strata_sizes = population["strata"].value_counts()

n_low = round(strata_sizes["low"]/N * n)
n_high = n - n_low
sample_low = population[population["strata"]=="low"].sample(n=n_low, random_state=123)
sample_high = population[population["strata"]=="high"].sample(n=n_high, random_state=123)
strat_sample = pd.concat([sample_low, sample_high])

strat_stats = strat_sample.groupby("strata")["income"].agg(['mean','var','count']).rename(columns={'count':'n_s'})
strat_stats['N_s'] = strat_stats.index.map(strata_sizes)
strat_stats['weight'] = strat_stats['N_s'] / N
strat_stats['weighted_mean'] = strat_stats['mean'] * strat_stats['weight']

strat_mean = strat_stats['weighted_mean'].sum()
strat_total = strat_mean * N

strat_stats['weighted_var'] = (strat_stats['weight']**2) * (strat_stats['var']/strat_stats['n_s'])
strat_se = np.sqrt(strat_stats['weighted_var'].sum())
strat_ci = stats.t.interval(0.95, df=n-1, loc=strat_mean, scale=strat_se)

results = pd.DataFrame({
    "Sampling": ["SRS", "Systematic", "Stratified"],
    "Mean_Estimate": [srs_mean, sys_mean, strat_mean],
    "Mean_95CI": [f"({srs_ci[0]:.2f}, {srs_ci[1]:.2f})",
                   f"({sys_ci[0]:.2f}, {sys_ci[1]:.2f})",
                   f"({strat_ci[0]:.2f}, {strat_ci[1]:.2f})"],
    "Total_Estimate": [srs_total, sys_total, strat_total]
})

print("Summary of Estimates:\n")
print(results)

print("\nDetailed Stratified Stats:")
print(strat_stats[['mean','var','n_s','N_s','weight','weighted_mean','weighted_var']])


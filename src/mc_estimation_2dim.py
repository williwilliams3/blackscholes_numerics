import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def black_scholes_call(S, K, T, mu, sigma, r):
    """
    Calculate Black-Scholes European call option price.

    Parameters:
    S (array): Current stock price
    K (float): Option's strike price
    T (float): Time to expiration (in years)
    mu (array): Drift (expected return)
    r (float): Risk-free interest rate
    sigma (array): Volatility of the stock

    Returns:
    call_price array: Call option price
    """
    d1 = (np.log(S / K) + (mu + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def simulate_gbm(T, dt, N, mu, sigma, S0):
    """
    Simulate Geometric Brownian Motion using Euler-Maruyama scheme.

    Parameters:
    T (float): Total time
    dt (float): Time step
    N (int): Number of steps
    mu (array): Drift (expected return)
    sigma (array): Volatility (standard deviation of the returns)
    S0 (array): Initial stock price

    Returns:
    numpy.ndarray: Array of simulated stock prices over time
    """

    num_assets = len(S0)
    t = np.linspace(0, T, N + 1)
    W = np.random.normal(0, np.sqrt(dt), size=(N, num_assets))
    dS = mu * dt + sigma * W
    cumulative_dS = np.cumprod(1 + dS, axis=0)
    S = S0 * cumulative_dS
    S = np.insert(S, 0, S0, axis=0)
    return t, S


def plot_trayectories(n_paths=5):
    # Plot simulated prices for visual verification
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(n_paths):  # Plot up to 10 paths
        _, S = simulate_gbm(T, dt, N, r, sigma, S0)
        axs[0].plot(_, S[:, 0], label=f"Path {i+1}", alpha=0.5)
        axs[1].plot(_, S[:, 1], label=f"Path {i+1}", alpha=0.5)

    axs[0].set_title("Simulated Stock Prices (GBM) for Asset 1")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Stock Price")
    axs[0].axhline(y=K[0], color="black", linestyle="-.", label="Strike Price")
    axs[0].legend()
    axs[1].set_title("Simulated Stock Prices (GBM) for Asset 2")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Stock Price")
    axs[1].axhline(y=K[1], color="black", linestyle="-.", label="Strike Price")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("figs/GBM_trayectories.png")


def plot_error_estimation():
    num_assets = 2
    bs_price = black_scholes_call(S0, K, T, mu, sigma, r)
    num_paths_values = [100, 1_000, 10_000, 100_000]
    mc_price_means = []
    mc_price_sds = []

    def get_mc_estimate():
        simulated_prices = np.zeros((num_paths, num_assets))
        for i in range(num_paths):
            _, S = simulate_gbm(T, dt, N, r, sigma, S0)
            simulated_prices[i] = np.maximum(S[-1] - K, 0) * np.exp(-r * T)
        return np.mean(simulated_prices, axis=0)

    for num_paths in num_paths_values:
        # Monte Carlo simulations of GBM
        mc_estimates = []
        # Repeat 5 times to get estimation of std
        for k in range(5):
            mc_estimates.append(get_mc_estimate())

        # Monte Carlo option price (average of simulated prices)
        mc_estimates = np.array(mc_estimates)
        mc_price_means.append(mc_estimates[0])
        mc_price_sds.append(np.std(mc_estimates, axis=0))
    mc_price_means = np.array(mc_price_means)
    mc_price_sds = np.array(mc_price_sds)
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for j in range(2):
        axs[j].errorbar(
            num_paths_values,
            mc_price_means[:, j],
            yerr=mc_price_sds[:, j],
            fmt="o",
            capsize=5,
            label="Monte Carlo",
        )
        axs[j].axhline(
            y=bs_price[j], color="black", linestyle="-.", label="Black-Scholes Price"
        )
        axs[j].set_xscale("log")
        axs[j].set_xlabel("Number of Samples")
        axs[j].set_ylabel("Option Price")
        axs[j].set_title(f"MC Error Estimation Asset {j} ")
        axs[j].legend()

    plt.tight_layout()
    plt.savefig("figs/GBM_MC_estimation.png")


if __name__ == "__main__":
    # Parameters
    num_assets = 2
    S0 = np.array([100.0, 100.0])  # Initial stock price
    K = np.array([105.0, 110.0])  # Option's strike price
    T = 1.0  # Time to expiration (in years)
    r = 0.05  # Risk-free interest rate
    mu = np.array([0.05, 0.05])  # Expected return
    sigma = np.array([0.2, 0.5])  # Volatility of the stock
    # Simulation parameters
    dt = 1 / 252  # Time step (daily intervals, for example)
    N = int(T / dt)  # Number of steps
    create_directory_if_not_exists("figs/")
    plot_trayectories()
    plot_error_estimation()

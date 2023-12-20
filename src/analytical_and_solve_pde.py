import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from FMNM.Parameters import Option_param
from FMNM.Processes import Diffusion_process
from FMNM.BS_pricer import BS_pricer
import os


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes European call option price.

    Parameters:
    S (float): Current stock price
    K (float): Option's strike price
    T (float): Time to expiration (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility of the stock

    Returns:
    float: Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def plot_value_price(axis=[50, 140, 0, 50]):
    K = 105.0  # Option's strike price
    T = 1.0  # Time to expiration (in years)
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility of the stock
    S_max = 6 * float(K)
    S_min = float(K) / 6
    S_vec = np.linspace(S_min, S_max, 1000)
    price_vec = black_scholes_call(S_vec, K, T, r, sigma)
    plt.plot(
        S_vec,
        np.maximum(S_vec - K, 0),
        color="blue",
        label="Payoff Asset 1",
        linestyle="--",
    )
    plt.plot(S_vec, price_vec, color="blue", label="BS curve Asset 1")
    if type(axis) == list:
        plt.axis(axis)

    K = 110.0  # Option's strike price
    T = 1.0  # Time to expiration (in years)
    r = 0.05  # Risk-free interest rate
    sigma = 0.5  # Volatility of the stock
    price_vec = black_scholes_call(S_vec, K, T, r, sigma)
    plt.plot(
        S_vec,
        np.maximum(S_vec - K, 0),
        color="red",
        label="Payoff Asset 2",
        linestyle="--",
    )
    plt.plot(S_vec, price_vec, color="red", label="BS curve Asset 2")

    plt.xlabel("Initial value")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Analytical Black-Scholes price")
    plt.savefig(f"figs/BS_price.png")


def plot_value_price_pde(axis=[50, 140, 0, 50]):
    opt_param = Option_param(S0=100, K=105, T=1, exercise="European", payoff="call")
    diff_param = Diffusion_process(r=0.05, sig=0.2)
    BS = BS_pricer(opt_param, diff_param)
    if type(BS.S_vec) != np.ndarray or type(BS.price_vec) != np.ndarray:
        BS.PDE_price((700, 500))
    plt.plot(
        BS.S_vec,
        BS.payoff_f(BS.S_vec),
        color="blue",
        label="Payoff Asset 1",
        linestyle="--",
    )
    plt.plot(BS.S_vec, BS.price_vec, color="blue", label="BS curve Asset 1")

    opt_param = Option_param(S0=100, K=110, T=1, exercise="European", payoff="call")
    diff_param = Diffusion_process(r=0.05, sig=0.5)
    BS = BS_pricer(opt_param, diff_param)
    if type(BS.S_vec) != np.ndarray or type(BS.price_vec) != np.ndarray:
        BS.PDE_price((7000, 5000))
    plt.plot(
        BS.S_vec,
        BS.payoff_f(BS.S_vec),
        color="red",
        label="Payoff Asset 2",
        linestyle="--",
    )
    plt.plot(BS.S_vec, BS.price_vec, color="red", label="BS curve Asset 2")

    if type(axis) == list:
        plt.axis(axis)
    plt.xlabel("Initial value")
    plt.ylabel("Price")
    plt.title(f"Numerical Black-Scholes price")
    plt.legend()
    plt.savefig(f"figs/BS_price_pde.png")


if __name__ == "__main__":
    # Parameters
    create_directory_if_not_exists("figs/")
    plot_value_price()
    plot_value_price_pde()

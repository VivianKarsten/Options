"""Script to calculate option prices using the Binomial Option Pricing Model (BOPM)"""

import math
import numpy as np
from scipy.stats import binom


def get_payoff_option(S, K, option_right="call"):
    """calculates the payoff for a long position in a call or a put option at a certain time"""
    if option_right == "call":
        return max(S - K, 0)
    elif option_right == "put":
        return max(K - S, 0)
    else:
        print("Option right can either be call or put.")


def get_tree_variables(sigma, dt, r, q=0):
    """
    Calculate variables u, d and p of binomial tree
    p is the probability of an up movement in a risk-neutral world
    S can move up by factor u or down by factor d
    u >= 1
    0 < d <= 1
    d = 1/u
    Assumption is log normal behaviour of underlying stock
    """
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    return u, d, p


def get_price_european_option(T, S0, K, r, sigma, n, q=0, option_right="call"):
    """
    T       time to maturity in years
    S0      price underlying at t=0
    K       Strike price of the option
    r       risk free rate per year
    sigma   volatility
    n       number of time steps of the binomial tree
    q       dividend rate per year
    """
    dt = T / n
    u, d, p = get_tree_variables(sigma, dt, r, q)

    # Calculate stock price and option price at end of tree:
    ST = np.zeros([n + 1, 1])
    payoffs_end = np.zeros([n + 1, 1])
    chance_payoff = np.zeros([n + 1, 1])

    for nr_d in range(n + 1):
        # nr_d is the number of times of a down movement.
        ST[nr_d] = S0 * u ** (n - nr_d) * d ** (nr_d)
        payoffs_end[nr_d] = get_payoff_option(ST[nr_d], K, option_right)
        chance_payoff[nr_d] = binom.pmf(n - nr_d, n, p)

    # Calculate option price at t=0, f0:
    weighted_payoff_end = chance_payoff * payoffs_end
    expected_payoff_end = weighted_payoff_end.sum()
    f0 = math.exp(-n * (r - q) * dt) * expected_payoff_end

    return f0


def get_price_american_option(T, S0, K, r, sigma, n, q=0, option_right="call"):
    dt = T / n
    u, d, p = get_tree_variables(sigma, dt, r, q)

    # Calculate tree stock prices, S:

    S = np.zeros([n + 1, n + 1])
    for t_step in range(0, n + 1):
        for nr_d in range(t_step + 1):
            S[nr_d, t_step] = S0 * u ** (t_step - nr_d) * d ** (nr_d)

    # Calculate tree option price, f:

    # calculate option payoff at end
    f = np.zeros([n + 1, n + 1])
    for nr_d in range(n + 1):
        f[nr_d, -1] = get_payoff_option(S[nr_d, n], K, option_right)

    # move backwards through the tree
    exercise = np.zeros([n, n])
    for t_step in reversed(range(n)):
        for nr_d in range(t_step + 1):
            # Binomial value:
            f[nr_d, t_step] = math.exp(-(r - q) * dt) * (
                p * f[nr_d, t_step + 1] + (1 - p) * f[nr_d + 1, t_step + 1]
            )
            # exercise value
            exercise[nr_d, t_step] = get_payoff_option(S[nr_d, t_step], K, option_right)
            # get max of binom and exercise value:
            if f[nr_d, t_step] < exercise[nr_d, t_step]:
                f[nr_d, t_step] = exercise[nr_d, t_step]

    return f[0, 0]


price_option = get_price_american_option(
    T=2, S0=50, K=52, r=0.05, sigma=0.3, n=2, q=0, option_right="put"
)
print(f"The price of the option is {round(price_option,2)} dollars.")

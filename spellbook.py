from scipy import stats
import numpy as np


class Distribution:
    def __init__(self, pdf: callable, cdf: callable, ppf: callable):
        self.pdf = pdf
        self.cdf = cdf
        self.ppf = ppf


class QuantileTransformer:
    def __init__(self, dst_from: Distribution, dst_to: Distribution):
        self.f1 = dst_from.cdf
        self.f2_inv = dst_to.ppf

    def __call__(self, x: np.ndarray):
        x_trans = self.f2_inv(self.f1(x))
        return x_trans


uniform_0_1 = Distribution(stats.uniform.pdf, stats.uniform.cdf, stats.uniform.ppf)
normal_standard = Distribution(stats.norm.pdf, stats.norm.cdf, stats.norm.ppf)


def g_tilde_pdf(x: np.ndarray):
    return np.abs(x)*np.exp(-x**2)


def g_tilde_cdf(x: np.ndarray):
    res = np.zeros_like(x, np.float64)
    int_1 = x <= 0
    res[int_1] = np.exp(-x[int_1]**2)/2
    int_2 = x > 0
    res[int_2] = 1 - np.exp(-x[int_2]**2)/2
    return res


def g_tilde_ppf(x: np.ndarray):
    res = np.zeros_like(x, np.float64)
    int_1 = np.logical_and(x >= 0, x <= 1/2)
    res[int_1] = -(-np.log(2*x[int_1]))**0.5
    int_2 = np.logical_and(x >= 1/2, x <= 1)
    res[int_2] = (-np.log(2-2*x[int_2]))**0.5
    return res


g_tilde = Distribution(g_tilde_pdf, g_tilde_cdf, g_tilde_ppf)


g_polynomial = Distribution(lambda x: (5/2)*x**(3/2), lambda x: x**(5/2), lambda x: x**(2/5))


def sample_from_distribution(dst: Distribution, n_samples: int):
    selection = stats.uniform.rvs(0, 1, n_samples).astype(np.float64)
    if dst == uniform_0_1:
        return selection
    else:
        qt = QuantileTransformer(uniform_0_1, dst)
        res = qt(selection)
        return res


def run_simulation(h: callable, g: Distribution, pi: Distribution, n_samples: int):
    selection = sample_from_distribution(g, n_samples)
    eps = 1e-2
    estimated = h(selection)*(pi.pdf(selection)+eps)/(g.pdf(selection)+eps)

    estimated_values = [estimated[:n].mean() for n in range(1, n_samples)]

    return np.array(estimated_values), estimated

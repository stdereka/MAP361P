from scipy import stats
import numpy as np


EPS = 1e-2


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
    estimated = h(selection)*(pi.pdf(selection)+EPS)/(g.pdf(selection)+EPS)

    estimated_values = [estimated[:n].mean() for n in range(1, n_samples)]

    return np.array(estimated_values), estimated


def sample_labels(alphas: np.ndarray, n_samples: int):
    selection = stats.uniform.rvs(0, 1, n_samples).astype(np.float64)
    labels = np.zeros_like(selection, np.int)
    for i in range(1, len(alphas)):
        labels[selection >= alphas[:i].sum()] = i
    return labels


class AdaptiveSampling:
    def __init__(self, h: callable, dists: list, pi: Distribution):
        self.d = len(dists)
        self.alphas = np.ones(self.d)/self.d
        self.dists = dists
        self.h = h
        self.pi = pi

    def _compose_dists(self, x: np.ndarray):
        res = np.zeros_like(x, np.float64)
        for n, dist in enumerate(self.dists):
            res += self.alphas[n] * dist.pdf(x)
        return res

    def fit(self, n_samples=10000, max_iter=1000, tolerance=1e-6):
        alphas_log = []
        approx_log = []
        variance_log = []

        for i in range(max_iter):
            alphas_log.append(self.alphas)
            labels = sample_labels(self.alphas, n_samples)
            selection = np.zeros(n_samples, np.float64)
            for label, count in zip(*np.unique(labels, return_counts=True)):
                if count != 0:
                    selection[labels == label] = sample_from_distribution(self.dists[label], count)

            estimated = self.h(selection)*(self.pi.pdf(selection)+EPS)/(self._compose_dists(selection)+EPS)
            approx = np.mean(estimated)
            approx_log.append(approx)
            variance_log.append(estimated.var())

            denominator = np.sum((self.h(selection)*(self.pi.pdf(selection)+EPS)/(self._compose_dists(selection)+EPS))**2)

            new_alphas = np.zeros_like(self.alphas, np.float64)

            for label in np.unique(labels):
                subsel = selection[labels == label]
                nominator = np.sum((self.h(subsel)*(self.pi.pdf(subsel)+EPS)/(self._compose_dists(subsel)+EPS))**2)
                new_alphas[label] = nominator/denominator

            delta = np.abs(new_alphas - self.alphas).mean()
            self.alphas = new_alphas

            if delta <= tolerance:
                print(f"Algorithm converged on iteration {i}")
                break
        else:
            print(f"Algorithm did not converge")

        return alphas_log, approx_log, variance_log

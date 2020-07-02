from scipy import stats
import numpy as np

EPS = 1e-6


# Class for storing distributions and sampling from them
class Distribution:
    def __init__(self, pdf: callable, cdf: callable, ppf: callable):
        """
        :param pdf: Probability density function.
        :param cdf: Cumulative distribution function.
        :param ppf: Quantile function.
        """
        self.pdf = pdf
        self.cdf = cdf
        self.ppf = ppf

    def sample(self, n_samples: int):
        """
        Sample from distribution.
        :param n_samples: Number of samples.
        :return: res - np.array with samples.
        """
        selection = stats.uniform.rvs(0, 1, n_samples).astype(np.float64)
        qt = QuantileTransformer(uniform_0_1, self)
        res = qt(selection)
        return res


# Performs quantile transformation on given sample
class QuantileTransformer:
    def __init__(self, dst_from: Distribution, dst_to: Distribution):
        """
        :param dst_from: Initial distribution.
        :param dst_to: Distribution to be transformed into.
        """
        self.f1 = dst_from.cdf
        self.f2_inv = dst_to.ppf

    def __call__(self, x: np.ndarray):
        """
        Apply transformation on given sample.
        :param x:
        :return:
        """
        x_trans = self.f2_inv(self.f1(x))
        return x_trans


# Some basic distributions
uniform_0_1 = Distribution(stats.uniform.pdf, stats.uniform.cdf, stats.uniform.ppf)
normal_standard = Distribution(stats.norm.pdf, stats.norm.cdf, stats.norm.ppf)
laplace = Distribution(stats.laplace.pdf, stats.laplace.cdf, stats.laplace.ppf)


# Optimal distribution for S3
def g_tilde_pdf(x: np.ndarray):
    return np.abs(x)*np.exp(-x**2/2)/2


def g_tilde_cdf(x: np.ndarray):
    res = np.zeros_like(x, np.float64)
    int_1 = x <= 0
    res[int_1] = np.exp(-x[int_1]**2/2)/2
    int_2 = x > 0
    res[int_2] = 1 - np.exp(-x[int_2]**2/2)/2
    return res


def g_tilde_ppf(x: np.ndarray):
    res = np.zeros_like(x, np.float64)
    int_1 = np.logical_and(x >= 0, x <= 1/2)
    res[int_1] = -(-2*np.log(2*x[int_1]))**0.5
    int_2 = np.logical_and(x >= 1/2, x <= 1)
    res[int_2] = (-2*np.log(2-2*x[int_2]))**0.5
    return res


g_tilde = Distribution(g_tilde_pdf, g_tilde_cdf, g_tilde_ppf)


# Some custom distributions
g_polynomial = Distribution(lambda x: (5/2)*x**(3/2), lambda x: x**(5/2), lambda x: x**(2/5))
g_hyperbolic = Distribution(lambda x: (1/np.log(2))/(x+1), lambda x: np.log(x+1), lambda x: np.exp(x) - 1)


def run_simple_simulation(h: callable, g: Distribution, pi: Distribution, n_samples: int):
    """
    Runs the simulation initially proposed in S1/S2.
    :param h:
    :param g:
    :param pi:
    :param n_samples:
    :return: estimated_values - Monte-Carlo integral estimation with [1, n_samples] sample size.
    """
    selection = g.sample(n_samples)
    estimated = h(selection)*(pi.pdf(selection)+EPS)/(g.pdf(selection)+EPS)

    # TODO Can be done faster!
    estimated_values = [estimated[:n].mean() for n in range(1, n_samples)]

    return np.array(estimated_values), estimated


# Class for adaptive sampling algorithm
class AdaptiveSampling:
    def __init__(self, h: callable, dists: list, pi: Distribution):
        """
        :param h: Function $h(x)$ from problem statement.
        :param dists: List with distributions for adaptive sampling.
        :param pi: Function $\pi(x)$ from problem statement.
        """
        self.d = len(dists)
        # Init alphas with equal values
        self.alphas = np.ones(self.d)/self.d
        self.dists = dists
        self.h = h
        self.pi = pi

    def _compose_dists(self, x: np.ndarray):
        """
        Computes $\sum_{i=1}^{d} \alpha_{i} g_{i}(x)$ expression.
        :param x:
        :return:
        """
        res = np.zeros_like(x, np.float64)
        for n, dist in enumerate(self.dists):
            res += self.alphas[n] * dist.pdf(x)
        return res

    def _sample_labels(self, n_samples: int):
        """
        Generates n_samples indexes corresponding to the distributions in self.dists with probabilities in self.alphas.
        :param n_samples:
        :return: labels - np.array with labels.
        """
        selection = stats.uniform.rvs(0, 1, n_samples).astype(np.float64)
        labels = np.zeros_like(selection, np.int)
        for i in range(1, len(self.alphas)):
            labels[selection >= self.alphas[:i].sum()] = i
        return labels

    def sample(self, n_samples=10000):
        """
        Samples n_samples values from distributions in self.dists.
        ith distribution is chosen with probability self.alphas[i].
        :param n_samples:
        :return: selection - generated sample, labels - np.array with distribution numbers.
        """
        labels = self._sample_labels(n_samples)
        selection = np.zeros(n_samples, np.float64)
        for label, count in zip(*np.unique(labels, return_counts=True)):
            if count != 0:
                selection[labels == label] = self.dists[label].sample(count)
        return selection, labels

    def estimate(self, n_samples=10000):
        """
        Generates n_samples values of $\frac{\pi(x) h(x)}{\sum_{i=1}^{d} \alpha_{i} g_{i}(x)}$.
        :param n_samples:
        :return:
        """
        selection, _ = self.sample(n_samples)
        estimated = self.h(selection)*self.pi.pdf(selection)/self._compose_dists(selection)
        return estimated

    def fit(self, n_samples=10000, max_iter=1000, tolerance=1e-6, debug=False):
        """
        Training loop. Performs all the steps described in the problem statement.
        :param n_samples: Number of samples to estimate alphas on each step.
        :param max_iter: Maximal number of iterations.
        :param tolerance: MAE threshold distance between current and updated self.alphas to stop fitting.
        :param debug: Store estimated Monte-Carlo integral values in history["global_estimation_log"] on each step.
        :return:
        """
        history = {
            "alphas_log": [],
            "approx_log": [],
            "variance_log": [],
            "global_estimation_log": []
        }

        for i in range(max_iter):
            history["alphas_log"].append(self.alphas)

            selection, labels = self.sample(n_samples)

            estimated = self.h(selection)*self.pi.pdf(selection)/self._compose_dists(selection)
            approx = np.mean(estimated)
            history["approx_log"].append(approx)
            history["variance_log"].append(estimated.var())

            if debug:
                m = 100
                ests = np.array([self.estimate(m).mean()*m**0.5 for _ in range(10000)])
                history["global_estimation_log"].append(ests)

            denominator = np.sum((self.h(selection)*self.pi.pdf(selection)/self._compose_dists(selection))**2)

            new_alphas = np.zeros_like(self.alphas, np.float64)

            for label in np.unique(labels):
                subsel = selection[labels == label]
                numerator = np.sum((self.h(subsel)*self.pi.pdf(subsel)/self._compose_dists(subsel))**2)
                new_alphas[label] = numerator/denominator

            delta = np.abs(new_alphas - self.alphas).mean()
            self.alphas = new_alphas

            if delta <= tolerance:
                print(f"Algorithm converged on iteration {i}")
                break
        else:
            print(f"Maximal iteration ({max_iter}) is reached")

        return history

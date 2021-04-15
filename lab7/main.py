from scipy.stats import norm, laplace, uniform, chi2
import numpy as np


def hi2Normal(dist, mu, sigma, k, N):
    theoryQuantiles = np.linspace(mu - 3 * sigma, mu +  3 * sigma, num=k - 1)
    cumulantes = norm(loc=mu, scale=sigma).cdf(theoryQuantiles)

    probs = [cumulantes[0]]
    rates = [len(list(filter(lambda x: x <= theoryQuantiles[0], dist)))]
    for i in range(k - 2):
        probs.append(cumulantes[i + 1] - cumulantes[i])
        rates.append(len(list(filter(
            lambda x: theoryQuantiles[i + 1] >= x >= theoryQuantiles[i], dist))))
    probs.append(1 - cumulantes[-1])
    rates.append(len(list(filter(lambda x: x >= theoryQuantiles[-1], dist))))

    result = 0
    for i in range(0, k - 1):
        result += ((rates[i] - N * probs[i]) ** 2) / (probs[i] * N)

    print("quantiles")
    print(theoryQuantiles)

    print("n_i")
    print(rates)

    print("p_i")
    print(np.around(probs, decimals=4))

    Npi = [N * x for x in probs]
    print("n*p_i")
    print(np.around(Npi, decimals=4))

    absDelts = [rates[i] - Npi[i] for i in range(len(probs))]
    print("n_i-n*p_i")
    print(np.around(absDelts, decimals=4))

    delts = [absDelts[i] / Npi[i] for i in range(len(probs))]
    print("n_i-n*p_i /n*p_i")
    print(np.around(delts, decimals=4))

    print("result")
    print(result)

    return np.around(result, decimals=2)


if __name__ == "__main__":
    alpha = 0.05
    k = 6
    sizeNormal = 100
    sizeAdds = 20

    normal = np.sort(norm.rvs(loc=0, scale=1, size=sizeNormal))
    estMean, estVar = np.mean(normal), np.sqrt(np.var(normal))
    print(estMean)
    print(estVar)
    laplace = np.sort(laplace.rvs(size=sizeAdds, scale=estVar / np.sqrt(2), loc=estMean))
    uni = np.sort(uniform.rvs(size=sizeAdds, loc=estMean - 3*estVar, scale=6 * estVar))

    value = chi2.ppf(1 - alpha, k - 1)
    print(value)
    hiNorm = hi2Normal(normal, estMean, estVar, k, sizeNormal)
    hiUniform = hi2Normal(uni, estMean, estVar, k, sizeAdds)
    hiLaplace = hi2Normal(laplace, estMean, estVar, k, sizeAdds)

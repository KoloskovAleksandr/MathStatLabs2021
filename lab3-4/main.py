import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, cauchy, laplace, uniform, gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF


def z_quantile(distr, np):
    return distr[int(len(distr) * np)]


def variates(sizes):
    rvs = [[], [], [], [], []]
    for size in sizes:
        rvs[0].append(numpy.sort(norm.rvs(loc=0, scale=1, size=size)))
        rvs[1].append(numpy.sort(laplace.rvs(size=size, scale=1 / numpy.sqrt(2), loc=0)))
        rvs[2].append(numpy.sort(poisson.rvs(10, size=size)))
        rvs[3].append(numpy.sort(cauchy.rvs(size=size)))
        rvs[4].append(numpy.sort(uniform.rvs(size=size, loc=-numpy.sqrt(3), scale=2 * numpy.sqrt(3))))
    return rvs


def distributions():
    return [norm(loc=0, scale=1), laplace(scale=1 / numpy.sqrt(2), loc=0),
            poisson(10), cauchy(),
            uniform(loc=-numpy.sqrt(3), scale=2 * numpy.sqrt(3))]


def boxplots(sizes):
    names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]
    rvs = variates(sizes)
    sizes_str = [str(size) for size in sizes]
    red_square = dict(markerfacecolor='r', marker='s')
    for i in range(len(names)):
        fig, ax = plt.subplots()
        ax.set_title(names[i])
        ax.boxplot(rvs[i], vert=False, flierprops=red_square)
        ax.set_yticklabels(sizes_str, fontsize=8)
        plt.show()
        fig.savefig(names[i] + '.png')


def empirical_outliers(sizes, count):
    names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]
    p_outlays = []

    for size in sizes:
        p_outlay = [0, 0, 0, 0, 0]
        for i in range(count):
            rvs = variates([size])
            for j in range(len(rvs)):
                q_25, q_75 = z_quantile(rvs[j][0], 0.25), z_quantile(rvs[j][0], 0.75)
                inf, sup = q_25 - 1.5 * (q_75 - q_25), q_75 + 1.5 * (q_75 - q_25)
                outliers = [x for x in rvs[j][0] if x < inf or x > sup]
                p_outlay[j] = p_outlay[j] + len(outliers) / len(rvs[j][0])
        p_outlays.append([outlay / count for outlay in p_outlay])

    with open('EOutliers.tex', 'w') as file:
        file.write(r" Sample & Outliers\\ ")
        file.write("\hline ")
        for k in range(len(names)):
            for j in range(len(sizes)):
                file.write(names[k] + " $n$ =" + str(sizes[j]) + " & ")
                file.write(str(round(p_outlays[j][k], 3)) + r"\\")
                file.write(" \hline ")


def theoretical_outliers(sizes):
    names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]
    p_outlay, Q_25, Q_75, Inf, Sup = [[0, 0, 0, 0, 0] for j in range(5)]
    distrs = distributions()
    for i in range(len(distrs)):
        Q_25[i], Q_75[i] = distrs[i].ppf(0.25), distrs[i].ppf(0.75)
        Inf[i], Sup[i] = Q_25[i] - 1.5 * (Q_75[i] - Q_25[i]), Q_75[i] + 1.5 * (Q_75[i] - Q_25[i])
        p_outlay[i] = distrs[i].cdf(Inf[i]) + (1 - distrs[i].cdf(Sup[i]))


    with open('TOutliers.tex', 'w') as file:
        file.write(
            r"Distibution&$Q^{\text{T}}_1$&$Q^{\text{T}}_3$&$X^{\text{T}}_1$&$X^{\text{T}}_2$&$P^{\text{T}}$\\)")
        file.write("\hline ")
        for k in range(len(names)):
                file.write(names[k] + " & ")
                file.write(str(round(Q_25[k], 3)) + " & ")
                file.write(str(round(Q_75[k], 3)) + " & ")
                file.write(str(round(Inf[k], 3)) + " & ")
                file.write(str(round(Sup[k], 3)) + " & ")
                file.write(str(round(p_outlay[k], 3)) + r"\\")
                file.write(" \hline ")


def empirical_distributions(sizes):
    names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]

    x = numpy.linspace(-4, 4, 1000)
    rvs = variates(sizes)
    distrs = distributions()

    for j in range(len(sizes)):
        for i in range(len(names)):
            y = ECDF(rvs[i][j])(x)
            z = distrs[i].cdf(x)
            fig, ax = plt.subplots()
            ax.set_title(names[i] + str(sizes[j]))
            ax.plot(x, z)
            ax.plot(x, y)
            plt.show()
            fig.savefig(names[i] + str(sizes[j]) + '.png')


def kernel_distributions(sizes):
    names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]
    scale = [0.5, 1, 2]
    x_r = numpy.linspace(-4, 4, 1000)
    x_p = numpy.linspace(6, 14, 9)
    rvs = variates(sizes)
    distrs = distributions()

    for k in range(len(scale)):
        for j in range(len(sizes)):
            for i in range(len(names)):
                if names[i] == "Poisson":
                    x = x_p
                    y = gaussian_kde(rvs[i][j], bw_method=scale[k] * 1.06 * sizes[j] ** (-1 / 5))(x)
                    z = distrs[i].pmf(x)
                else:
                    x = x_r
                    y = gaussian_kde(rvs[i][j], bw_method=scale[k] * 1.06 * sizes[j] ** (-1 / 5))(x)
                    z = distrs[i].pdf(x)
                fig, ax = plt.subplots()
                ax.set_title(names[i] + str(sizes[j]) + ' ' + str(scale[k]))
                ax.plot(x, y)
                ax.plot(x, z)
                plt.show()
                fig.savefig(names[i] + str(sizes[j]) + ' ' + str(scale[k]) + '.png')


if __name__ == "__main__":
    sizes3 = [20, 100]
    sizes4 = [20, 60, 100]
    count = 1000
    boxplots(sizes3)
    empirical_outliers(sizes3, count)
    theoretical_outliers(sizes3)
    empirical_distributions(sizes4)
    kernel_distributions(sizes4)

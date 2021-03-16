import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal, pearsonr, spearmanr, multinomial


def Covariance(variances, correlations):
    covariance = [[] for i in range(len(variances))]
    for i in range(len(variances)):
        for k in range(len(variances)):
            covariance[i].append((variances[i] * variances[k]) * correlations[i][k])
    return covariance


def QuadrantCorrelation(x, y):
    centerRVS = [x - np.mean(x), y - np.mean(y)]
    centerRVS = [[centerRVS[0][i], centerRVS[1][i]] for i in range(len(centerRVS[0]))]
    n1 = len([elem for elem in centerRVS if elem[0] >= 0 and elem[1] >= 0])
    n2 = len([elem for elem in centerRVS if elem[0] <= 0 and elem[1] >= 0])
    n3 = len([elem for elem in centerRVS if elem[0] <= 0 and elem[1] <= 0])
    n4 = len([elem for elem in centerRVS if elem[0] >= 0 and elem[1] <= 0])
    return (n1 + n3 - n2 - n4) / float(len(x))


def mean_2(x):
    return np.mean([elem * elem for elem in x])


def MixedNormalDistribution(weights, means, covariances, size):
    cases = multinomial(1, weights).rvs(size)
    rvs = []
    for case in cases:
        k = case.tolist().index(1)
        rvs.append(multivariate_normal.rvs(means[k], covariances[k]))
    rvs = np.asarray(rvs)
    return [rvs[:, k] for k in range(len(rvs[0]))]


def confidence_ellipse(x, y, ax, n_std=3.0, edgecolor='red', linestyle='-'):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    radius_x = np.sqrt(1 + pearson)
    radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=radius_x * 2, height=radius_y * 2, facecolor='none',
                      edgecolor=edgecolor, linestyle=linestyle)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    sizes = [20, 60, 100]
    count = 1000
    means = [0, 0]
    variances_1, variances_2 = [1, 1], [10, 10]
    correlations_1, correlations_2 = [0, 0.5, 0.9], [0.9, -0.9]
    measuares = [['$E(z)$', np.mean], ['$E(z^2)$', mean_2], ['$D(z)$', np.var]]
    empiricalCorrelations = ['r', 'r_S', 'r_Q']
    variateCorrelations = [[[[0 for l in range(len(empiricalCorrelations))]
                             for k in range(len(measuares))]
                            for j in range(len(sizes))]
                           for i in range(len(correlations_1))]
    mixedVariateCorrelations = [[[0 for l in range(len(empiricalCorrelations))]
                                 for k in range(len(measuares))]
                                for j in range(len(sizes))]

    for i in range(len(correlations_1)):
        covariance = Covariance(variances_1,
                                [[1, correlations_1[i]], [correlations_1[i], 1]])
        for j in range(len(sizes)):
            P, S, Q = [], [], []
            for m in range(count):
                rvs = multivariate_normal.rvs(means, covariance, sizes[i])
                x, y = rvs[:, 0], rvs[:, 1]
                p, tmp = pearsonr(x, y)
                s, tmp = spearmanr(x, y)
                P.append(p)
                S.append(s)
                Q.append(QuadrantCorrelation(x, y))
            empCors = [P, S, Q]
            for k in range(len(measuares)):
                for l in range(len(empCors)):
                    variateCorrelations[i][j][k][l] = measuares[k][1](empCors[l])
                    
    mixedCovariances = [Covariance(variances_1, [[1, correlations_2[0]], [correlations_2[0], 1]]),
                        Covariance(variances_2, [[1, correlations_2[1]], [correlations_2[1], 1]])]
    for j in range(len(sizes)):
        P, S, Q = [], [], []
        for m in range(count):
            rvs = MixedNormalDistribution([0.9, 0.1], [means, means], mixedCovariances, sizes[j])
            p, tmp = pearsonr(rvs[0], rvs[1])
            s, tmp = spearmanr(rvs[0], rvs[1])
            P.append(p)
            S.append(s)
            Q.append(QuadrantCorrelation(rvs[0], rvs[1]))
        empCors = [P, S, Q]
        for k in range(len(measuares)):
            for l in range(len(empCors)):
                mixedVariateCorrelations[j][k][l] = measuares[k][1](empCors[l])

    for i in range(len(correlations_1)):
        with open('pho = ' + str(correlations_1[i]) + ".tex", 'w') as file:
            for j in range(len(sizes)):
                file.write(" $n$=" + str(sizes[j]) + "&$r$" + "&$r_S$" + r"&$r_Q$\\")
                file.write("\hline")
                for k in range(len(measuares)):
                    file.write(measuares[k][0])
                    for m in range(len(empiricalCorrelations)):
                        file.write("&" + str(round(variateCorrelations[i][j][k][m], 4)))
                    file.write(r"\\")
                    file.write("\hline")
                file.write(r" & & & \\")
                file.write("\hline ")

    with open('Mixed.tex', 'w') as file:
        for j in range(len(sizes)):
            file.write(" $n$=" + str(sizes[j]) + "&$r$" + "&$r_S$" + r"&$r_Q$\\")
            file.write("\hline")
            for k in range(len(measuares)):
                file.write(measuares[k][0])
                for m in range(len(empiricalCorrelations)):
                    file.write("&" + str(round(mixedVariateCorrelations[j][k][m], 4)))
                file.write(r"\\")
                file.write("\hline")
            file.write(r" & & & \\")
            file.write("\hline ")

    for i in range(len(correlations_1)):
        covariance = Covariance(variances_1,
                                [[1, correlations_1[i]], [correlations_1[i], 1]])
        for j in range(len(sizes)):
            rvs = multivariate_normal.rvs(means, covariance, sizes[j])
            x, y = rvs[:, 0], rvs[:, 1]
            fig, ax_nstd = plt.subplots()
            ax_nstd.axvline(c='grey', lw=1)
            ax_nstd.axhline(c='grey', lw=1)
            ax_nstd.scatter(x, y, s=0.5)
            confidence_ellipse(x, y, ax_nstd)
            ax_nstd.scatter(0, 0, c='red', s=3)
            ax_nstd.set_xlabel('x')
            ax_nstd.set_ylabel('y')
            ax_nstd.set_title('n = ' + str(sizes[j]) + r', $\rho$ =' + str(correlations_1[i]))
            plt.show()
            fig.savefig('Ellipse ' + str(correlations_1[i]) + ' ' + str(sizes[j]) + '.png')


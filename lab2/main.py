import numpy
from scipy.stats import norm, poisson, cauchy, laplace, uniform


def z_np(distr, np):
    if np.is_integer():
        return distr[int(np)]
    else:
        return distr[int(np) + 1]


def z_q(distr, size):
    z_1 = z_np(distr, size / 4)
    z_2 = z_np(distr, 3 * size / 4)
    return (z_1 + z_2) / 2


def z_r(distr, size):
    return (distr[0] + distr[size - 1]) / 2


def z_tr(distr, size):
    r = int(size / 4)
    sum = 0
    for i in range(r + 1, size - r + 1):
        sum += distr[i]
    return sum / (size - 2 * r)


def mean(distr, size):
    return numpy.mean(distr)


def median(distr, size):
    return numpy.median(distr)


if __name__ == "__main__":

    sizes = [10, 100, 1000]
    count = 1001
    names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]

    characteristics = [[[], [], [], [], []],
                       [[], [], [], [], []],
                       [[], [], [], [], []]]
    expectedChrctrs = [[[], [], [], [], []],
                        [[], [], [], [], []],
                        [[], [], [], [], []]]
    varianceChrctrs = [[[], [], [], [], []],
                        [[], [], [], [], []],
                        [[], [], [], [], []]]

    functions = [mean, median, z_r, z_q, z_tr]

    for i in range(len(sizes)):
        size = sizes[i]
        for k in range(count):
            n = numpy.sort(norm.rvs(loc=0, scale=1, size=size))
            l = numpy.sort(laplace.rvs(size=size, scale=1 / numpy.sqrt(2), loc=0))
            p = numpy.sort(poisson.rvs(10, size=size))
            c = numpy.sort(cauchy.rvs(size=size))
            u = numpy.sort(uniform.rvs(size=size, loc=-numpy.sqrt(3), scale=2 * numpy.sqrt(3)))
            distributions = [n, l, p, c, u]
            for num in range(len(distributions)):
                d = distributions[num]
                for s in range(len(functions)):
                    if k == 0:
                        characteristics[i][num].append([])
                    value = functions[s](d, size)
                    characteristics[i][num][s].append(value)

        for j in range(len(distributions)):
            for k in range(len(functions)):
                distribution = characteristics[i][j][k]
                expectedChrctrs[i][j].append(numpy.mean(distribution))
                varianceChrctrs[i][j].append(numpy.std(distribution) ** 2)

    for i in range(len(names)):
        with open(names[i] + ".tex", 'w') as file:
            for j in range(len(sizes)):
                file.write(names[i] + " $n$=" + str(sizes[j]) + " & & & & & " + r"\\")
                file.write("\hline")
                file.write(" &\overline{x}" + "&$med x$" + "&$z_R$" + "&$z_Q$" + r"&$z_{tr}$\\")
                file.write("\hline")
                file.write("$E(z)$")
                for k in range(len(functions)):
                    file.write("&" + str(round(expectedChrctrs[j][i][k], 5)))
                file.write(r"\\")
                file.write("\hline")
                file.write("$D(z)$")
                for k in range(len(functions)):
                    file.write("&" + str(round(varianceChrctrs[j][i][k], 5)))
                file.write(r"\\")
                file.write("\hline")
                file.write(" & & & & & " + r"\\" + "\hline ")
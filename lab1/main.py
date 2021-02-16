from matplotlib import pyplot
from scipy.stats import norm, laplace, poisson, cauchy, uniform
import numpy

if __name__ == "__main__":

    sizes = [10, 50, 1000]
    densities = [norm(loc=0, scale=1), laplace(scale=1 / numpy.sqrt(2), loc=0),
                 poisson(10), cauchy(),
                 uniform(loc=-numpy.sqrt(3), scale=2 * numpy.sqrt(3))]
    names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]

    for size in sizes:
        n = norm.rvs(loc=0, scale=1, size=size)
        l = laplace.rvs(scale=1 / numpy.sqrt(2), loc=0, size=size)
        p = poisson.rvs(10, size=size)
        c = cauchy.rvs(size=size)
        u = uniform.rvs(loc=-numpy.sqrt(3), scale=2 * numpy.sqrt(3), size=size)
        distributions = [n, l, p, c, u]
        build = list(zip(distributions, densities, names))

        for histogram, density, name in build:
            fig, ax = pyplot.subplots(1, 1)
            ax.hist(histogram, density=True, histtype='stepfilled', alpha=0.6, color="red", bins=20)
            if histogram is p:
                x = numpy.arange(-5, 25)
                ax.plot(x, poisson.pmf(x, 10), 'ok', lw=2)
            else:
                x = numpy.linspace(density.ppf(0.01), density.ppf(0.99), 1000)
                ax.plot(x, density.pdf(x), 'k-', lw=2)
            ax.set_ylabel("Density")
            ax.set_title(name + " distribution. Size: " + str(size))
            pyplot.savefig(name + str(size) + ".png")



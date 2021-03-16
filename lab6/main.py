import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, linregress

def LAD(x, y):
    k_q = 1.491
    size = len(x)
    l = int(size / 4)
    j = size - l + 1
    med_x, med_y = np.median(x), np.median(y)
    r_Q = 0
    for i in range(size):
        r_Q += np.sign((x[i] - med_x) * (y[i] - med_y))
    r_Q = r_Q / size
    q_y, q_x = (y[j] - y[l]) / k_q, (x[j] - x[l]) / k_q

    a = r_Q * q_y / q_x
    return a, med_y - a * med_x

if __name__ == "__main__":
    x = np.linspace(-1.8, 2, 20)
    e = norm.rvs(0, 1, size=20)
    y0 = 2 * x + 2
    y1 = 2 * x + 2 + e
    e = norm.rvs(0, 1, size=20)
    y2 = 2 * x + 2 + e
    y2[0], y2[19] = y2[0] + 10, y2[19] - 10

    result1 = linregress(x, y1)
    a1, b1 = result1.slope, result1.intercept
    a3, b3 = LAD(x, y1)
    plt.plot(x, y1, 'o', label='original data')
    plt.plot(x, y0, 'b', label='$2x + 2$')
    plt.plot(x, b1 + a1 * x, 'r', label='LS')
    plt.plot(x, b3 + a3 * x, 'g', label='LAD')
    plt.legend()
    plt.show()
    plt.savefig('No perturbation')

    result2 = linregress(x, y2)
    a2, b2 = result2.slope, result2.intercept
    a4, b4 = LAD(x, y2)
    plt.plot(x, y2, 'o', label='original data')
    plt.plot(x, y0, 'b', label='$2x + 2$')
    plt.plot(x, b2 + a2 * x, 'r', label='LS')
    plt.plot(x, b4 + a4 * x, 'g', label='LAD')
    plt.legend()
    plt.show()
    plt.savefig('With perturbation')

    print(round(a1, 4), round(b1, 4))
    print(round(a2, 4), round(b2, 4))
    print(round(a3, 4), round(b3, 4))
    print(round(a4, 4), round(b4, 4))
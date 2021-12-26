import sys
import numpy as np
import matplotlib.pyplot as plt


def perturb_DP(y, alpha=2):
    '''
    To perturb a continuous-valued numpy vector with additive Laplace noise, evaluated by DP,
    where the noise level is connected with the privacy leakage parameter alpha

    Input
        y: numpy vector
        alpha: a float value in (0, infty). smaller means more private, more noisy

    Output
        y_noise: numpy vector of the same shape as y
    '''
    # truncate y into the bounded range of [a, b], where T is determined by 2.5% quantile
    a, b = np.quantile(y, 0.025), np.quantile(y, 0.975)
    y_noise = np.copy(y)
    y_noise[y < a] = a
    y_noise[y > b] = b
    y_noise += np.random.laplace(scale=(b - a) / alpha, size=y.shape[0])

    return y_noise


def perturb_IP(y, numThresh=1):
    '''
    To perturb a continuous-valued numpy vector with additive Laplace noise, evaluated by IP,
    where the privacy leakage parameter is the average interval width

    Input
        y: numpy vector
        alpha: a float value in (0, infty). smaller means more private, more noisy

    Output
        y_noise: numpy vector of the same shape as y
    '''
    # truncate y into the bounded range of [a, b], where T is determined by 2.5% quantile
    a, b = np.quantile(y, 0.025), np.quantile(y, 0.975)
    n = y.shape[0]
    y_noise = np.zeros(n)
    intervals = np.zeros((n, 2))
    for j in range(n):
        intervals[j, 0], intervals[j, 1] = a, b
        for i in range(numThresh):
            t = np.random.uniform(low=a, high=b)
            if y[j] < t:
                intervals[j, 1] = np.minimum(t, intervals[j, 1])
                y_noise[j] += (2 * t - b) / numThresh
            else:
                intervals[j, 0] = np.maximum(t, intervals[j, 0])
                y_noise[j] += (2 * t - a) / numThresh

    # estimate the privacy leakage
    leak = np.zeros(n)
    for j in range(n):
        leak[j] = np.logical_and(y >= intervals[j, 0], y < intervals[j, 1]).mean()
    leak_avg = leak.mean()

    print(f'IP privacy leakage (or average interval width) is {leak_avg}')

    return intervals, leak_avg, y_noise


# test
if __name__ == '__main__':
    y = np.random.normal(size=100)

    plt.figure(1)
    plt.title('Y versus DP noisy Y')
    y_noise = perturb_DP(y, alpha=1)
    plt.plot(y, y_noise, '.')
    plt.show()

    plt.figure(2, figsize=(8, 4))
    intervals, leak_avg, y_noise = perturb_IP(y, numThresh=1)
    plt.subplot(2, 1, 1)
    plt.title('Y and DP noisy Y')
    plt.plot(y, y_noise, '.')

    plt.subplot(2, 1, 2)
    plt.title('Y and IP intervals')
    plt.plot(y, '-')
    plt.plot(intervals[:, 0], 'bx-')
    plt.plot(intervals[:, 1], 'bx-')

    plt.tight_layout()
    plt.show()

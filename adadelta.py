import sys
import numpy

from matplotlib import pyplot

n_steps = 100000
gradients = [1e-4, 1e-3, 1e-2, 1e-1, 1e2]
decay_rate = float(sys.argv[1])
epsilon = float(sys.argv[2])

for gradient in gradients:
    mean_gradient2 = 0
    mean_step2 = 0

    steps = numpy.zeros(n_steps)
    for i in range(n_steps):
        mean_gradient2 = decay_rate * mean_gradient2 + (1 - decay_rate) * gradient ** 2
        steps[i] = ((mean_step2 + epsilon) / (mean_gradient2 + epsilon)) ** 0.5 * gradient
        mean_step2 = decay_rate * mean_step2 + (1 - decay_rate) * steps[i] ** 2

    pyplot.plot(steps)
    pyplot.ylabel('Absolute step')
    pyplot.xlabel('Iteration number')
pyplot.legend(gradients, loc='best')
pyplot.savefig('plot.pdf')
pyplot.show()

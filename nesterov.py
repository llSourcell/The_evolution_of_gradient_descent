import numpy as np
import math

def calc_numerical_gradient(func, x, delta_x):
    """Function for computing gradient numerically."""
    val_at_x = func(x)
    val_at_next = func(x + delta_x)
    return (val_at_next - val_at_x) / delta_x


def nesterov_descent(func, L, dimension, init_x=None, numerical_gradient=True, delta_x=0.0005, gradient_func=None,epsilon=None):
    
    assert delta_x > 0, "Step must be positive."

    if (init_x is None):
        x = np.zeros(dimension)  # todo проверка что функция определена в 0.
    else:
        x = init_x

    if (epsilon is None):
        epsilon = 0.05

    lambda_prev = 0
    lambda_curr = 1
    gamma = 1
    y_prev = x
    alpha = 0.05 / (2 * L)

    if numerical_gradient:
        gradient = calc_numerical_gradient(func, x, delta_x)
    else:
        gradient = gradient_func(x)

    while np.linalg.norm(gradient) >= epsilon:
        y_curr = x - alpha * gradient
        x = (1 - gamma) * y_curr + gamma * y_prev
        y_prev = y_curr

        lambda_tmp = lambda_curr
        lambda_curr = (1 + math.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
        lambda_prev = lambda_tmp

        gamma = (1 - lambda_prev) / lambda_curr

        if numerical_gradient:
            gradient = calc_numerical_gradient(func, x, delta_x)
        else:
            gradient = gradient_func(x)

    return x

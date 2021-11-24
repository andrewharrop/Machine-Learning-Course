import numpy as np
import scipy.optimize as so
# How is this function flawed?


def linearModelPredict(coefficients, features):
    return np.dot(coefficients, features.T)


features = np.array([[1, 0], [1, -1], [1, 2]])
thetas = np.array([0.1, 0.3])

predictions = linearModelPredict(thetas, features)

print(f"Predictions: {predictions}")

"""
    Linear Model predict function is flawed.
    Think about the linear algebra. If thetas is a 2x2 matrix, the function will still work,
    but the result will be invalid according to the linear regression equation.
    We can add a check to the function to make sure thetas is a viable matrix.
"""


# This is better.
def linearModelPredict(coefficients, features):
    if coefficients.shape[0] != features.shape[1]:
        raise ValueError(
            "The number of coefficients must equal the number of features.")
    return features@coefficients


features = np.array([[1, 0], [1, -1], [1, 2]])
thetas = np.array([0.1, 0.3])

predictions = linearModelPredict(thetas, features)

print(f"Verified predictions: {predictions}")


# Now we apply a loss function to the predictions.
def linearModelLossRSS(coefficients, features, observations):
    predictions = linearModelPredict(coefficients, features)
    residuals = observations - predictions
    gradient = -2*np.dot(residuals, features)
    rss = np.sum(residuals**2)
    return rss, gradient


features = np.array([[1, 0], [1, -1], [1, 2]])
thetas = np.array([0.1, 0.3])
observations = np.array([0, 0.4, 2])

(rss, gradient) = linearModelLossRSS(thetas, features, observations)

print(f"RSS: {rss}")
print(f"Gradient: {gradient}")

# Now minimize it the coefficients.
coefs = so.minimize(linearModelLossRSS, [0, 0], args=(
    features, observations), jac=True)

print(f"Minimized coefficients: {coefs.x}")

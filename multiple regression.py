import numpy as np
import math

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X: list[list], Y: list[float]) -> dict:
        X = np.array(X)
        Y = np.array(Y)

        # bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # solution
        self.coefficients = np.linalg.pinv(X_b) @ Y

        return {
            f"b{i}": coef for i, coef in enumerate(self.coefficients)
        }

    def predict(self, X: list) -> list:
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b @ self.coefficients


#example, where we consider the price of the flat and how it related to area, number of bedrooms, age
x_base = [[60, 2, 20], [75, 3, 15], [90, 3, 10], [120, 4, 8], [150, 4, 5], [80, 2, 25], [110, 3, 12], [130, 4, 7], [95, 3, 18], [160, 5, 3]]
y_base = [150, 200, 250, 320, 400, 180, 290, 350, 240, 450]

model = MultipleLinearRegression()
coef = model.fit(x_base, y_base)
rounded_coef = {k: round(v, 2) for k, v in coef.items()}
print(f'Coef: {rounded_coef}')

x_new = [100, 3, 10]
pred = model.predict(x_new)
print(f'Prediction: {pred}')
class SimpleLinearRegression:
    def __init__(self):
        self.x_mean = None
        self.y_mean = None

        self.b0 = None  # Intercept
        self.b1 = None  # Slope

        self.X = None
        self.Y = None

        self.error = None

    def fit(self, X: list, Y: list) -> dict:
        self.x_mean = sum(X)/len(X)
        self.y_mean = sum(Y)/len(Y)

        n = 0
        d = 0
        for i in range(len(X)):
            n += ((X[i] - self.x_mean) * (Y[i] - self.y_mean))
            d += (X[i] - self.x_mean)**2

        self.b1 = n/d

        self.b0 = self.y_mean - self.b1 * self.x_mean

        return {
            'b1' : self.b1,
            'b0' : self.b0
        }

    def predict(self, x: float) -> float:
        self.Y = self.b0 + self.b1 * x
        return self.Y



x_entry = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_entry = [8, 11, 13, 18, 20, 24, 27, 29, 33, 36]

model = SimpleLinearRegression()
parameter_data = model.fit(X=x_entry, Y=y_entry)
print(parameter_data)
new_x = 11
results = model.predict(x=new_x)
print(f"When x is {new_x} -> y will be {results}")
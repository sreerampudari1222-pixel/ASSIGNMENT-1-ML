**PUDARI SREE RAM CHARAN TEJA  700762701** MACHINE LEARNING ASSIGNMENT - 1

Linear Regression: Closed-form vs Gradient Descent 

This script demonstrates:

Generating synthetic linear data with Gaussian noise.
Solving linear regression using the closed-form solution (Normal Equation).
Solving linear regression using Gradient Descent.
Comparing both solutions with plots and loss curve. """
----------------------------
1. Import Libraries
----------------------------
import numpy as np import matplotlib.pyplot as plt

----------------------------
2. Generate Synthetic Dataset
----------------------------
np.random.seed(42) # for reproducibility n_samples = 200 X = np.random.uniform(0, 5, n_samples) epsilon = np.random.normal(0, 1, n_samples) # Gaussian noise y = 3 + 4 * X + epsilon # y = 3 + 4x + noise

Plot raw data
plt.scatter(X, y, color='blue', label='Raw data') plt.xlabel('X') plt.ylabel('y') plt.title('Synthetic Data') plt.legend() plt.show()

----------------------------
3. Closed-Form Solution (Normal Equation)
----------------------------
X_b = np.c_[np.ones((n_samples, 1)), X] # add bias column theta_closed = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) intercept_closed, slope_closed = theta_closed

print(f"Closed-form solution:\nIntercept: {intercept_closed:.4f}, Slope: {slope_closed:.4f}")

Plot closed-form fitted line
plt.scatter(X, y, color='blue', label='Raw data') plt.plot(X, X_b.dot(theta_closed), color='red', label='Closed-form fit') plt.xlabel('X') plt.ylabel('y') plt.title('Closed-form Linear Regression') plt.legend() plt.show()

----------------------------
4. Gradient Descent Implementation
----------------------------
theta_gd = np.array([0.0, 0.0]) learning_rate = 0.05 iterations = 1000 m = n_samples loss_history = []

for i in range(iterations): gradients = (2/m) * X_b.T.dot(X_b.dot(theta_gd) - y) theta_gd = theta_gd - learning_rate * gradients mse = np.mean((X_b.dot(theta_gd) - y) ** 2) loss_history.append(mse)

intercept_gd, slope_gd = theta_gd print(f"Gradient Descent solution:\nIntercept: {intercept_gd:.4f}, Slope: {slope_gd:.4f}")

Plot loss curve
plt.plot(range(iterations), loss_history, color='green') plt.xlabel('Iterations') plt.ylabel('MSE') plt.title('Loss Curve (Gradient Descent)') plt.show()

----------------------------
5. Comparison Plot
----------------------------
plt.scatter(X, y, color='blue', label='Raw data') plt.plot(X, X_b.dot(theta_closed), color='red', label='Closed-form fit') plt.plot(X, X_b.dot(theta_gd), color='orange', linestyle='--', label='Gradient Descent fit') plt.xlabel('X') plt.ylabel('y') plt.title('Linear Regression Comparison') plt.legend() plt.show()

----------------------------
6. Short Explanation
----------------------------
print("Comparison Summary:") print(f"- Closed-form: Intercept ≈ {intercept_closed:.2f}, Slope ≈ {slope_closed:.2f}") print(f"- Gradient Descent: Intercept ≈ {intercept_gd:.2f}, Slope ≈ {slope_gd:.2f}") print("Observation: Gradient Descent converges closely to the closed-form solution, as seen in the fitted lines and MSE loss curve.")

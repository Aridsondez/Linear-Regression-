# CAP 4630 HW1 — Linear Regression with Gradient Descent (Problem 1)

This README is your step-by-step guide to complete **Problem 1** and earn full credit. It explains what to implement, what to document, how to validate your results, and exactly what to include in your submission.

---

## 1. Goal of Problem 1

Given `D3.csv` with 3 input features \\(x_1, x_2, x_3\\) and target \\(y\\), you will:

1. Train a linear regression model using **gradient descent**.
2. Report the learned parameters \\(\\theta_0, \\theta_1, \\theta_2, \\theta_3\\) on the original feature scale.
3. Plot and include a **loss curve** that shows convergence.
4. Predict \\(y\\) for three new inputs: `(1, 1, 1)`, `(2, 0, 4)`, `(3, 2, 1)`.
5. Explain your method, derivations, and preprocessing choices.

---

## 2. Deliverables Checklist

Include the following in your final submission.

- Code
  - `problem1_gradient_descent.py` that:
    - Loads `D3.csv`
    - Standardizes features, adds bias
    - Runs gradient descent
    - Converts learned parameters back to the original scale
    - Saves `loss_curve.png`
    - Prints final parameters and the three predictions
- Report
  - Your LaTeX report (`report_scaffold.tex` or your own) compiled to PDF that contains:
    - Cost function \\(J(\\theta)\\) and gradient update derivation
    - Explanation of feature scaling and hyperparameters (learning rate, iterations)
    - The loss curve figure
    - Final model with \\(\\theta_0, \\theta_1, \\theta_2, \\theta_3\\)
    - Predictions for `(1, 1, 1)`, `(2, 0, 4)`, `(3, 2, 1)`
- Optional verification
  - Show that a normal equation solution roughly matches your gradient descent results

---

## 3. File Structure

```
HW1/
├── D3.csv
├── problem1_gradient_descent.py
├── loss_curve.png                 # created after running the script
├── report_scaffold.tex            # or your own LaTeX
└── README.md
```

---

## 4. Theory You Must Explain

### 4.1 Hypothesis
Model with bias term:
\\[
h_\\theta(x) = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + \\theta_3 x_3
\\]

### 4.2 Cost Function
Mean Squared Error (with a helpful factor):
\\[
J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} \\big(h_\\theta(x^{(i)}) - y^{(i)}\\big)^2
\\]

### 4.3 Gradient Descent Update
For each parameter \\(\\theta_j\\):
\\[
\\theta_j := \\theta_j - \\alpha \\cdot \\frac{1}{m} \\sum_{i=1}^{m} \\big(h_\\theta(x^{(i)}) - y^{(i)}) x^{(i)}_j
\\]
Notes:
- For the bias term, use \\(x^{(i)}_0 = 1\\).
- \\(\\alpha\\) is the learning rate. Choose a value such as 0.01 to 0.05. If loss increases, your \\(\\alpha\\) is too high.

### 4.4 Feature Scaling
Standardize features so all features have similar scale. For each feature \\(x_j\\):
\\[
x'_j = \\frac{x_j - \\mu_j}{\\sigma_j}
\\]
- Compute \\(\\mu\\) and \\(\\sigma\\) on the training features.
- Keep \\(\\mu\\) and \\(\\sigma\\) to convert back to the original units after training.

### 4.5 Converting Coefficients Back to Original Scale
If the model was trained on standardized features, you must map the final parameters back to the original feature units. If the standardized model is:
\\[
y = \\theta_0^{std} + \\sum_j \\theta_j^{std} \\cdot \\frac{x_j - \\mu_j}{\\sigma_j}
\\]
then the original scale coefficients are
\\[
\\theta_j = \\frac{\\theta_j^{std}}{\\sigma_j}, \\quad
\\theta_0 = \\theta_0^{std} - \\sum_j \\theta_j \\mu_j
\\]

---

## 5. Step-by-Step Instructions

1. **Load data**
   - Load `D3.csv` without headers, split into `X` (first 3 columns) and `y` (last column).

2. **Standardize features**
   - Compute `mu` and `sigma` across the rows for each of the 3 columns.
   - Create `Xs = (X - mu) / sigma`.

3. **Add bias column**
   - Create `Xb = [1, Xs]` by concatenating a column of ones onto `Xs`.

4. **Initialize parameters**
   - Start with `theta_std = zeros(4)`.

5. **Run gradient descent**
   - Hyperparameters
     - `alpha = 0.03` (start here, adjust if needed)
     - `iters = 20000`
   - On each iteration:
     - `grad = (Xb.T @ (Xb @ theta_std - y)) / m`
     - `theta_std -= alpha * grad`
     - Optionally log `J(theta_std)` every 100 steps

6. **Track convergence**
   - Compute and store `J(theta_std)` over time.
   - Plot iterations vs `J` to `loss_curve.png`.

7. **Convert parameters to original scale**
   - Use the formulas in section 4.5.
   - Report \\(\\theta_0, \\theta_1, \\theta_2, \\theta_3\\) on original scale.

8. **Make predictions**
   - Use the original scale parameters to predict for `(1,1,1)`, `(2,0,4)`, `(3,2,1)`:
     - `y_hat = theta_0 + theta_1*x1 + theta_2*x2 + theta_3*x3`

9. **Explain your choices**
   - Why you standardized
   - What learning rate and iterations you used
   - How you confirmed convergence
   - How you converted back to original units

---

## 6. How to Run

From the folder that contains `D3.csv`:

```bash
python problem1_gradient_descent.py D3.csv
```

This will print the learned parameters and the three predictions. It will also create `loss_curve.png`.

---

## 7. Common Pitfalls and Fixes

- **Loss goes up or oscillates**
  - Decrease learning rate. Try 0.02, 0.01, or 0.005.

- **Loss decreases at first then plateaus high**
  - Increase iterations. Try 30k to 50k.
  - Verify that you standardized features.

- **Huge or NaN values**
  - Check for bad CSV formatting, missing values, or zero standard deviation.

- **Predictions look off by a constant**
  - Verify that you converted coefficients back to original scale correctly.

---

## 8. Validation Ideas

- Compute the **normal equation** on the same data to verify that your \\(\\theta\\) is close.
- Split data into train and test to sanity check generalization.
- Shuffle rows and rerun to confirm stability.

---

## 9. What to Write in Your Report

- **Derivation**
  - \\(J(\\theta)\\) and gradient update rule. Short and clear.
- **Preprocessing**
  - Feature scaling method and formulas. Show your \\(\\mu\\) and \\(\\sigma\\) vectors.
- **Hyperparameters**
  - List learning rate and iteration count. State why.
- **Evidence of convergence**
  - Paste `loss_curve.png`. Mention the iteration where it stabilized.
- **Final model**
  - Write the full equation with numbers: \\(\\hat{y} = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + \\theta_3 x_3\\).
- **Predictions**
  - Show intermediate substitution for each of the three required inputs.

---

## 10. Quick Start Using Provided Script

You already have `problem1_gradient_descent.py` from this workspace. It implements all the steps above, including standardization, gradient descent, conversion back to original scale, and plotting the loss curve. Place `D3.csv` next to it and run it. Then copy the printed results and the plot into your LaTeX report.

Good luck. Fill in the report with your derivations and choices, and you are set for full credit.

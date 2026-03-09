import numpy as np

def rbf_design_matrix(x, centers, sigma):
    # Builds Phi[k, i] = exp( -(x_k - mu_i)^2 / (2 sigma^2))
    x = np.asarray(x)[:, None]          # (N, 1)
    c = np.asarray(centers)[None, :]    # (1, M)
    Phi = np.exp(-0.5*((x-c)/sigma)**2) # (N, M)
    return Phi 

def fit_rbf_least_squares(x_train, y_train, centers, sigma):
    # Solves for w to minimize ||phi @ w - f||^2
    phi = rbf_design_matrix(x_train, centers, sigma)
    w, *_ = np.linalg.lstsq(phi, y_train, rcond=None) # (M, )
    return w # This w is the minimizer of the total error 

def predict_rbf(x, centers, sigma, w):
    # calculates y = Phi @ w
    phi = rbf_design_matrix(x, centers, sigma)
    return phi @ w


if __name__ == "__main__":
    # Data generation for sin(2x)
    x_train = np.arange(0.0, 2*np.pi + 1e-9, 0.1)
    x_test = np.arange(0.05, 2*np.pi + 1e-9, 0.1)

    y_train_sin = np.sin(2*x_train)
    y_test_sin = np.sin(2*x_test)
    # Data generation for square(2x), no idea why it is called square(2x). It's a function
    # that is +1 when sin(2x)>=0, and -1 otherwise
    y_train_sq = np.where(np.sin(2*x_train) >= 0, 1.0, -1.0)
    y_test_sq = np.where(np.sin(2*x_test) >= 0, 1.0, -1.0)
    # Choose RBF nodes by hand
    M_values = [2, 10, 25, 50, 63, 75, 100]
    for M in M_values:
        centers = np.linspace(0, 2*np.pi, M)
        delta = centers[1] - centers[0]
        sigma = delta / np.sqrt(2)  # This is a heuristic to have a nice overlap between Gaussians.
        # train + test sin(2x)
        w_sin = fit_rbf_least_squares(x_train, y_train_sin, centers, sigma)
        pred_sin = predict_rbf(x_test, centers, sigma, w_sin)
        mae_sin = np.mean(np.abs(pred_sin - y_test_sin))
        # train + test square(2x)
        w_sq = fit_rbf_least_squares(x_train, y_train_sq, centers, sigma)
        pred_sq = predict_rbf(x_test, centers, sigma, w_sq)

        # y_hat = predict_rbf(x_test, centers, sigma, w_sq)   # raw output (real)
        # pred_sq = np.where(y_hat >= 0, 1.0, -1.0)           # transformed output (±1)
        # uncomment lines above for the answer to the second question in 3.1
        mae_sq = np.mean(np.abs(pred_sq - y_test_sq))

        print(f"M: {M} | MAE sin:", mae_sin)
        print(f"M: {M} | MAE square:", mae_sq)
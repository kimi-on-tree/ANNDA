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

def fit_rbf_delta_rule(x_train, y_train, centers, sigma, eta=0.01, epochs=50, shuffle=True):
    Phi = rbf_design_matrix(x_train, centers, sigma)   # (N, M)
    y = np.asarray(y_train).reshape(-1)                # (N,)
    N, M = Phi.shape

    w = np.zeros(M)                                    # init
    idx = np.arange(N)

    for _ in range(epochs):
        if shuffle:
            np.random.shuffle(idx)

        for k in idx:
            phi_k = Phi[k]                             # (M,)
            y_hat = phi_k @ w
            e = y[k] - y_hat
            w += eta * e * phi_k

    return w
def sigma_from_centers(centers, mult=1.0):
    centers = np.asarray(centers)
    if len(centers) < 2:
        base = 2*np.pi
    else:
        spacings = np.diff(np.sort(centers))
        base = np.median(spacings) / np.sqrt(2)
    return base * mult

def run_one_setting(x_train, x_test,
                    y_train_noisy, y_test_noisy,
                    y_test_clean,
                    centers, sigma,
                    eta=0.01, epochs=200):

    # LS
    w_ls = fit_rbf_least_squares(x_train, y_train_noisy, centers, sigma)
    pred_ls = predict_rbf(x_test, centers, sigma, w_ls)
    ls_noisy = mse(pred_ls, y_test_noisy)
    ls_clean = mse(pred_ls, y_test_clean)

    # Delta rule
    w_dl = fit_rbf_delta_rule(x_train, y_train_noisy, centers, sigma, eta=eta, epochs=epochs, shuffle=True)
    pred_dl = predict_rbf(x_test, centers, sigma, w_dl)
    dl_noisy = mse(pred_dl, y_test_noisy)
    dl_clean = mse(pred_dl, y_test_clean)

    return ls_noisy, ls_clean, dl_noisy, dl_clean

# ---- comparison experiment ----
def compare_center_placement(M_values, sigma_mults, repeats=20, eta=0.01, epochs=200):
    print("Comparing EVEN vs RANDOM centers (averaged over random repeats)")
    print("Metric: test MSE (noisy) and test MSE (clean)")

    for M in M_values:
        even_centers = np.linspace(0, 2*np.pi, M)

        for mult in sigma_mults:
            sigma_even = sigma_from_centers(even_centers, mult=mult)

            # EVEN (single deterministic run)
            even_sin = run_one_setting(
                x_train, x_test,
                y_train_sin_n, y_test_sin_n,
                y_test_sin,
                even_centers, sigma_even,
                eta=eta, epochs=epochs
            )

            # RANDOM (average over repeats)
            rand_ls_noisy = []
            rand_ls_clean = []
            rand_dl_noisy = []
            rand_dl_clean = []

            for _ in range(repeats):
                rand_centers = np.sort(np.random.uniform(0, 2*np.pi, size=M))
                sigma_rand = sigma_from_centers(rand_centers, mult=mult)

                ls_noisy, ls_clean, dl_noisy, dl_clean = run_one_setting(
                    x_train, x_test,
                    y_train_sin_n, y_test_sin_n,
                    y_test_sin,
                    rand_centers, sigma_rand,
                    eta=eta, epochs=epochs
                )

                rand_ls_noisy.append(ls_noisy)
                rand_ls_clean.append(ls_clean)
                rand_dl_noisy.append(dl_noisy)
                rand_dl_clean.append(dl_clean)

            print(f"\nSIN | M={M:3d} mult={mult:>4}")
            print(f"  EVEN   | LS noisy={even_sin[0]:.4g} clean={even_sin[1]:.4g} | DL noisy={even_sin[2]:.4g} clean={even_sin[3]:.4g}")
            print(f"  RANDOM | LS noisy={np.mean(rand_ls_noisy):.4g}±{np.std(rand_ls_noisy):.2g} "
                  f"clean={np.mean(rand_ls_clean):.4g}±{np.std(rand_ls_clean):.2g} | "
                  f"DL noisy={np.mean(rand_dl_noisy):.4g}±{np.std(rand_dl_noisy):.2g} "
                  f"clean={np.mean(rand_dl_clean):.4g}±{np.std(rand_dl_clean):.2g}")
            
def predict_rbf(x, centers, sigma, w):
    # calculates y = Phi @ w
    phi = rbf_design_matrix(x, centers, sigma)
    return phi @ w

def mse(a, b):
    return np.mean((a - b)**2)

if __name__ == "__main__":
    np.random.seed(0)
    noise_std = np.sqrt(0.1)
    # Data generation for sin(2x)
    x_train = np.arange(0.0, 2*np.pi + 1e-9, 0.1)
    x_test = np.arange(0.05, 2*np.pi + 1e-9, 0.1)

    y_train_sin = np.sin(2*x_train)
    y_test_sin = np.sin(2*x_test)
    y_train_sin_n = y_train_sin + np.random.normal(0, noise_std, size=y_train_sin.shape)
    y_test_sin_n  = y_test_sin  + np.random.normal(0, noise_std, size=y_test_sin.shape)
    # Data generation for square(2x), no idea why it is called square(2x). It's a function
    # that is +1 when sin(2x)>=0, and -1 otherwise
    y_train_sq = np.where(np.sin(2*x_train) >= 0, 1.0, -1.0)
    y_test_sq = np.where(np.sin(2*x_test) >= 0, 1.0, -1.0)
    y_train_sq_n = y_train_sq + np.random.normal(0, noise_std, size=y_train_sq.shape)
    y_test_sq_n  = y_test_sq  + np.random.normal(0, noise_std, size=y_test_sq.shape)
    # Choose RBF nodes by hand
    # Uncomment code below for answer to question 4
    # M_values = [10, 25, 50]
    # sigma_mults = [0.75, 1.0, 1.5]
    # compare_center_placement(M_values, sigma_mults, repeats=30, eta=0.01, epochs=200)
    M_values = [10, 25, 50, 63, 75, 100]
    sigma_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
    for M in M_values:
        centers = np.linspace(0, 2*np.pi, M)
        delta = centers[1] - centers[0]
        base_sigma = delta / np.sqrt(2)

        for mult in sigma_multipliers:
            sigma = base_sigma * mult

            # --- SIN (noisy) ---
            w_ls = fit_rbf_least_squares(x_train, y_train_sin_n, centers, sigma)
            pred_ls = predict_rbf(x_test, centers, sigma, w_ls)
            err_ls_noisy = mse(pred_ls, y_test_sin_n)
            err_ls_clean = mse(pred_ls, y_test_sin)

            w_dl = fit_rbf_delta_rule(x_train, y_train_sin_n, centers, sigma, eta=0.01, epochs=200, shuffle=True)
            pred_dl = predict_rbf(x_test, centers, sigma, w_dl)
            err_dl_noisy = mse(pred_dl, y_test_sin_n)
            err_dl_clean = mse(pred_dl, y_test_sin)

            #print(f"SIN    | M={M:3d} mult={mult:>4} | "
            #    f"MSE noisy LS={err_ls_noisy:.4g} DL={err_dl_noisy:.4g} | "
            #    f"MSE clean LS={err_ls_clean:.4g} DL={err_dl_clean:.4g}")

            # --- SQUARE (noisy) ---
            w_ls = fit_rbf_least_squares(x_train, y_train_sq_n, centers, sigma)
            pred_ls = predict_rbf(x_test, centers, sigma, w_ls)
            err_ls_noisy = mse(pred_ls, y_test_sq_n)
            err_ls_clean = mse(pred_ls, y_test_sq)

            w_dl = fit_rbf_delta_rule(x_train, y_train_sq_n, centers, sigma, eta=0.01, epochs=200, shuffle=True)
            pred_dl = predict_rbf(x_test, centers, sigma, w_dl)
            err_dl_noisy = mse(pred_dl, y_test_sq_n)
            err_dl_clean = mse(pred_dl, y_test_sq)

            # print(f"SQUARE | M={M:3d} mult={mult:>4} | "
            #     f"MSE noisy LS={err_ls_noisy:.4g} DL={err_dl_noisy:.4g} | "
            #     f"MSE clean LS={err_ls_clean:.4g} DL={err_dl_clean:.4g}")


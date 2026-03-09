import numpy as np
import matplotlib.pyplot as plt

def rbf_design_matrix(x, centers, sigma):
    x = np.asarray(x)
    if x.ndim == 1: x = x[:, None]
    centers = np.asarray(centers)
    if centers.ndim == 1: centers = centers[:, None]
    diff = x[:, None, :] - centers[None, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    return np.exp(-0.5 * sq_dist / (sigma**2))

def fit_rbf_least_squares(x_train, y_train, centers, sigma):
    Phi = rbf_design_matrix(x_train, centers, sigma) 
    w, *_ = np.linalg.lstsq(Phi, y_train, rcond=None)
    return w

def predict(x, centers, sigma, w):
    return rbf_design_matrix(x, centers, sigma) @ w

def competitive_learning(data, k, eta=0.1, epochs=500):
    data = np.asarray(data)
    centers = np.random.uniform(data.min(), data.max(), (k, 1))
    
    for _ in range(epochs):
        x = data[np.random.randint(len(data))]
        winner_idx = np.argmin(np.abs(centers - x))
        centers[winner_idx] += eta * (x - centers[winner_idx])
    
    return np.sort(centers.flatten())

def run_comparison_v2():
    x_train = np.arange(0, 2*np.pi, 0.1)
    x_test = np.arange(0.05, 2*np.pi, 0.1)
    
    y_clean = np.sin(2*x_train)
    y_test_clean = np.sin(2*x_test)
    
    noise_std = np.sqrt(0.1)
    y_noisy = y_clean + np.random.normal(0, noise_std, len(x_train))
    
    # params
    M = 10
    mult = 2.0
    
    print(f"--- Comparison (M={M}, sigma_mult={mult}) ---")
    #---Clean Data---
    # A. Manual (Uniform)
    c_man = np.linspace(0, 2*np.pi, M)
    s_man = (c_man[1] - c_man[0]) * mult
    
    # B. Random (No Learning)
    c_rnd = np.sort(np.random.uniform(0, 2*np.pi, M)) 
    s_rnd = np.mean(np.diff(c_rnd)) * mult
    
    # C. CL (Learned from Random)
    c_cl = competitive_learning(x_train, M)
    s_cl = np.mean(np.diff(c_cl)) * mult

    print("\n--- Clean Data Results ---")

    w_man = fit_rbf_least_squares(x_train, y_clean, c_man, s_man)
    mse_man = np.mean((predict(x_test, c_man, s_man, w_man) - y_test_clean)**2)

    w_rnd = fit_rbf_least_squares(x_train, y_clean, c_rnd, s_rnd)
    mse_rnd = np.mean((predict(x_test, c_rnd, s_rnd, w_rnd) - y_test_clean)**2)

    w_cl = fit_rbf_least_squares(x_train, y_clean, c_cl, s_cl)
    mse_cl = np.mean((predict(x_test, c_cl, s_cl, w_cl) - y_test_clean)**2)
    
    print(f"Manual (Ideal) MSE: {mse_man:.4e}")
    print(f"Random (Init)  MSE: {mse_rnd:.4e}")
    print(f"CL (Learned)   MSE: {mse_cl:.4e}")

    # ---Noisy Data---
    print("\n--- Noisy Data Results ---")
    w_man_n = fit_rbf_least_squares(x_train, y_noisy, c_man, s_man)
    mse_man_n = np.mean((predict(x_test, c_man, s_man, w_man_n) - y_test_clean)**2)
    
    w_rnd_n = fit_rbf_least_squares(x_train, y_noisy, c_rnd, s_rnd)
    mse_rnd_n = np.mean((predict(x_test, c_rnd, s_rnd, w_rnd_n) - y_test_clean)**2)
    
    w_cl_n = fit_rbf_least_squares(x_train, y_noisy, c_cl, s_cl)
    mse_cl_n = np.mean((predict(x_test, c_cl, s_cl, w_cl_n) - y_test_clean)**2)
    
    print(f"Manual (Ideal) MSE: {mse_man_n:.4e}")
    print(f"Random (Init)  MSE: {mse_rnd_n:.4e}")
    print(f"CL (Learned)   MSE: {mse_cl_n:.4e}")

    # ---Plotting---
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Clean Approximation")
    plt.plot(x_test, y_test_clean, 'k--', label='True', alpha=0.6)
    plt.plot(x_test, predict(x_test, c_man, s_man, w_man), 'b', alpha=0.8, label=f'Manual (Unif)')
    plt.plot(x_test, predict(x_test, c_rnd, s_rnd, w_rnd), 'g', alpha=0.8, label=f'Random')
    plt.plot(x_test, predict(x_test, c_cl, s_cl, w_cl), 'r', alpha=0.8, label=f'CL')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.title("Clean Centers Position")
    plt.plot(c_man, np.full_like(c_man, 1.0), 'bo', markersize=8, label='Manual (Ideal)')
    plt.plot(c_rnd, np.full_like(c_rnd, 0.0), 'g^', markersize=8, label='Random (Init)')
    plt.plot(c_cl,  np.full_like(c_cl, -1.0), 'rx', markersize=8, markeredgewidth=2, label='CL (Learned)')
    plt.ylim(-2, 2)
    plt.yticks([1, 0, -1], ["Manual", "Random", "CL"])
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Noisy Approximation")
    plt.plot(x_test, y_test_clean, 'k--', label='True', alpha=0.6)
    plt.plot(x_test, predict(x_test, c_man, s_man, w_man_n), 'b', alpha=0.8, label='Manual')
    plt.plot(x_test, predict(x_test, c_rnd, s_rnd, w_rnd_n), 'g', alpha=0.8, label='Random')
    plt.plot(x_test, predict(x_test, c_cl, s_cl, w_cl_n), 'r', alpha=0.8, label='CL')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Noisy Centers Position")
    plt.plot(c_man, np.full_like(c_man, 1.0), 'bo', markersize=8, label='Manual (Ideal)')
    plt.plot(c_rnd, np.full_like(c_rnd, 0.0), 'g^', markersize=8, label='Random (Init)')
    plt.plot(c_cl,  np.full_like(c_cl, -1.0), 'rx', markersize=8, markeredgewidth=2, label='CL (Learned)')
    plt.ylim(-2, 2)
    plt.yticks([1, 0, -1], ["Manual", "Random", "CL"])
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparison_v2()

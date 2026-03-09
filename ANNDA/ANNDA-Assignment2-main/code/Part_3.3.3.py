import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

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

# Competitive Learning with Leaky Strategy
def competitive_learning_2d(data, k, eta=0.1, epochs=3000, leaky_rate=0.005):
    data = np.asarray(data)
    N, D = data.shape  
    # Initialization
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    centers = np.random.uniform(min_val, max_val, (k, D))
    
    for _ in range(epochs):
        # Random sampling
        idx = np.random.randint(N)
        x = data[idx]      
        # Find winner
        dists = np.sum((centers - x)**2, axis=1) 
        winner_idx = np.argmin(dists)     
        diff = x - centers     
        # Update winner
        centers[winner_idx] += eta * diff[winner_idx]        
        # Leaky Update 
        if leaky_rate > 0:
            mask = np.ones(k, dtype=bool)
            mask[winner_idx] = False
            centers[mask] += (eta * leaky_rate) * diff[mask]
            
    return centers

def run_ballistic_task_local():
    print("\n--- Task 3.3 Part 3: Ballistic Function Approx (Files) ---")  
    try:
        train_data = np.loadtxt('ballist.dat')
        test_data = np.loadtxt('balltest.dat')
    except OSError as e:
        print(f"Error: Could not find data files ({e}). Please ensure ballist.dat and balltest.dat are in the current directory.")
        return
    
    # Split inputs and outputs
    X_train = train_data[:, :2] 
    Y_train = train_data[:, 2:]
    X_test = test_data[:, :2]
    Y_test = test_data[:, 2:]
    
    print(f"Data Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Number of nodes
    M = 45   
    # Position centers using CL
    print(f"Running 2D Competitive Learning (M={M})...")
    centers = competitive_learning_2d(X_train, M, eta=0.2, epochs=5000, leaky_rate=0.005)  
    # Determine Sigma 
    dists = pdist(centers)
    avg_dist = np.mean(dists)
    sigma = avg_dist * 0.8 
    print(f"Calculated Sigma: {sigma:.4f}")
    # Train weights
    w = fit_rbf_least_squares(X_train, Y_train, centers, sigma)
    
    # Test
    Y_pred = predict(X_test, centers, sigma, w)   
    # Calculate MSE
    test_mse = np.mean((Y_pred - Y_test)**2)
    print(f"Test MSE (Distance & Height): {test_mse:.4e}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"RBF Centers in Input Space (M={M})")
    plt.scatter(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.3, label='Training Data')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, linewidth=2, label='Learned Centers')
    plt.xlabel('Angle')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.title("Prediction: True vs Predicted Distance")
    plt.scatter(Y_test[:, 0], Y_pred[:, 0], c='blue', alpha=0.6)
    mn = min(Y_test[:, 0].min(), Y_pred[:, 0].min())
    mx = max(Y_test[:, 0].max(), Y_pred[:, 0].max())
    plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Perfect Fit')
    plt.xlabel('True Distance')
    plt.ylabel('Predicted Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ballistic_task_local()
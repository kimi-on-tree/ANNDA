import numpy as np
import matplotlib.pyplot as plt

def competitive_learning_demo(data, k, eta=0.1, epochs=3000, leaky_rate=0.0):  
    # Initialization: 
    # Data in [2, 4], init centers far away in [-4, -2]
    centers = np.random.uniform(-4, -2, (k, 1))
    init_centers = centers.copy()
    for _ in range(epochs):
        # Random sample
        x = data[np.random.randint(len(data))]    
        # Calculate distances
        dists = np.abs(centers - x)
        winner_idx = np.argmin(dists)
        
        # Calculate difference vector
        diff = x - centers       
        # A. Update Winner 
        centers[winner_idx] += eta * diff[winner_idx]
        
        # B. Update Losers (Leaky Learning)
        if leaky_rate > 0:
            mask = np.ones(k, dtype=bool)
            mask[winner_idx] = False
            centers[mask] += (eta * leaky_rate) * diff[mask]
            
    return init_centers.flatten(), np.sort(centers.flatten())

def run_dead_unit_demo():
    print("--- Dead Unit Demonstration ---")
       
    # Generate Data: Uniform distribution in [2, 4]
    data = np.random.uniform(2, 4, 1000)
    M = 10
    
    # Vanilla CL (leaky_rate = 0)
    print("Running Vanilla CL...")
    init_vanilla, final_vanilla = competitive_learning_demo(data, M, leaky_rate=0.0)
    
    # Strategy CL (leaky_rate = 0.005)
    print("Running Leaky CL (Strategy)...")
    init_leaky, final_leaky = competitive_learning_demo(data, M, leaky_rate=0.005)
    
    # Visualization 
    plt.figure(figsize=(12, 6))
    plt.axvspan(2, 4, color='green', alpha=0.1, label='Target Data Region [2, 4]')
    plt.axvspan(-4, -2, color='gray', alpha=0.1, label='Initialization Region [-4, -2]')
    y_baseline = 1.0

    plt.scatter(init_vanilla, np.full_like(init_vanilla, y_baseline), 
                c='gray', marker='o', s=100, alpha=0.5, label='Initial Centers')

    plt.scatter(final_vanilla, np.full_like(final_vanilla, y_baseline + 1), 
                c='blue', marker='x', s=120, linewidth=2, label='Vanilla CL Final')

    plt.scatter(final_leaky, np.full_like(final_leaky, y_baseline + 2), 
                c='red', marker='x', s=120, linewidth=2, label='Leaky CL Final (Strategy)')
 
    plt.title("Demonstration of 'Dead Units' vs Leaky Strategy")
    plt.xlabel("Input Space")
    plt.yticks([1, 2, 3], ['Start', 'Vanilla Result', 'Leaky Result'])
    plt.ylim(0.5, 3.5)
    plt.legend(loc='lower right')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_dead_unit_demo()
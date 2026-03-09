import matplotlib.pyplot as plt
import numpy as np
from SOM import SOM
from HelpersSOM import load_cities

if __name__ == "__main__":
    cities = load_cities()
    n = cities.shape[0]

    som = SOM(grid_shape=(n,), input_dim=2, seed=2)

    # A bit more training than 4.1 helps (TSP is finicky)
    som.train(
        cities,
        epochs=200,
        eta=0.2,
        radius_start=2,  # e.g. 3 if n=10
        radius_end=0,
        circular_1d=True,             # IMPORTANT: ring neighbourhood
        shuffle=True
    )

    # Each SOM node has a 2D weight -> interpret as tour points
    W = som.W  # shape (n,2)

    # --- Plot tour (closed loop) ---
    plt.figure()
    plt.scatter(cities[:, 0], cities[:, 1], label="Cities")

    # Close loop by repeating first weight at end
    plt.plot(
        np.r_[W[:, 0], W[0, 0]],
        np.r_[W[:, 1], W[0, 1]],
        label="SOM tour"
    )

    # Label the cities (0..n-1)
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, f"{i}", fontsize=10)

    plt.title("Part 4.2: Cyclic tour with 1D circular SOM")
    plt.axis("equal")
    plt.legend()
    plt.show()

    # --- Derive a city visit order (optional, but useful for report) ---
    # Assign each city to its BMU index along the ring, then sort by that index.
    bmu_idx = som.map_samples(cities)      # each city -> node index 0..n-1
    order = np.argsort(bmu_idx)

    print("Tour order (by BMU index):", order.tolist())

    # --- Tour length (optional) ---
    # Compute length following this derived order (and close the loop).
    tour = cities[order]
    tour_closed = np.vstack([tour, tour[0]])
    length = np.sum(np.linalg.norm(tour_closed[1:] - tour_closed[:-1], axis=1))
    print(f"Tour length (derived order): {length:.3f}")
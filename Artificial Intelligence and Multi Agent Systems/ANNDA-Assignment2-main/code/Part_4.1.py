import numpy as np
import matplotlib.pyplot as plt

from HelpersSOM import *
from SOM import SOM

if __name__ == "__main__":
    # Load data
    X, names = load_animals()
    print(X.shape, len(names))

    # Create 1D SOM with 100 nodes
    som = SOM(
        grid_shape=(100,),   # 1D map with 100 nodes
        input_dim=X.shape[1],
        seed=1
    )

    # Train
    som.train(
        X,
        epochs=20,
        eta=0.2,
        radius_start=50,   # lab suggestion
        radius_end=0,
        circular_1d=False,
        shuffle=True
    )

    # Get BMU position for each animal
    positions = som.map_samples(X)

    # Sort animals by position
    order = np.argsort(positions)

    print("\nAnimals ordered along SOM:\n")
    for idx in order:
        print(f"{positions[idx]:3d}  {names[idx]}")

    # Optional visualization
    plt.figure(figsize=(14, 3))
    plt.scatter(positions, np.zeros_like(positions))
    for i, name in enumerate(names):
        plt.text(positions[i], 0, name, rotation=90, fontsize=6)
    plt.yticks([])
    plt.title("Animal ordering along 1D SOM")
    plt.show()

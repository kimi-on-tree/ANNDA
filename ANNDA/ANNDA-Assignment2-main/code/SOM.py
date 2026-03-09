import numpy as np
import matplotlib.pyplot as plt


class SOM:
    """
    Simple SOM (Kohonen map) with:
      - 1D or 2D output grid
      - hard neighbourhood (radius in grid distance)
      - optional circular neighbourhood for 1D grids (TSP / cyclic tour)
    """

    def __init__(self, grid_shape, input_dim, seed=0):
        self.grid_shape = tuple(grid_shape)  # (n,) or (h,w)
        self.input_dim = int(input_dim)
        self.rng = np.random.default_rng(seed)

        self.n_nodes = int(np.prod(self.grid_shape))
        # weights in [0,1], as suggested in the lab for animals/votes
        self.W = self.rng.random((self.n_nodes, self.input_dim))

        # Precompute node coordinates in output space (grid indices)
        self.coords = self._make_coords()

    def _make_coords(self):
        if len(self.grid_shape) == 1:
            n = self.grid_shape[0]
            return np.arange(n).reshape(-1, 1)  # shape (n,1)
        elif len(self.grid_shape) == 2:
            h, w = self.grid_shape
            rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            return np.stack([rr.ravel(), cc.ravel()], axis=1)  # (h*w,2)
        else:
            raise ValueError("grid_shape must be (n,) or (h,w)")

    def winner(self, x):
        """Return index of BMU (best matching unit)."""
        # squared Euclidean distances
        d2 = np.sum((self.W - x[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    def _grid_distance(self, bmu_idx, circular_1d=False):
        """
        Distance from BMU to all nodes in output grid.
        - 1D: |i-j| (or circular min(|i-j|, n-|i-j|))
        - 2D: Manhattan distance on (row,col)
        """
        if len(self.grid_shape) == 1:
            n = self.grid_shape[0]
            i = np.arange(n)
            b = bmu_idx
            dist = np.abs(i - b)
            if circular_1d:
                dist = np.minimum(dist, n - dist)
            return dist.astype(int)

        # 2D Manhattan distance
        b = self.coords[bmu_idx]  # (2,)
        dist = np.abs(self.coords[:, 0] - b[0]) + np.abs(self.coords[:, 1] - b[1])
        return dist.astype(int)

    def train(
        self,
        X,
        epochs=20,
        eta=0.2,
        radius_start=5,
        radius_end=0,
        circular_1d=False,
        shuffle=True,
    ):
        """
        Hard-radius neighbourhood. Radius decays linearly from start -> end.
        """
        X = np.asarray(X, dtype=float)
        assert X.shape[1] == self.input_dim

        for ep in range(epochs):
            # linear radius schedule (integer radius)
            if epochs == 1:
                radius = int(round(radius_end))
            else:
                t = ep / (epochs - 1)
                radius = int(round(radius_start + t * (radius_end - radius_start)))

            idxs = np.arange(X.shape[0])
            if shuffle:
                self.rng.shuffle(idxs)

            for k in idxs:
                x = X[k]
                bmu = self.winner(x)

                dist = self._grid_distance(bmu, circular_1d=circular_1d)
                nbrs = np.where(dist <= radius)[0]

                # Update all neighbours (including winner)
                self.W[nbrs] += eta * (x[None, :] - self.W[nbrs])

    def map_samples(self, X):
        """Return BMU indices for each sample in X."""
        X = np.asarray(X, dtype=float)
        bmus = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            bmus[i] = self.winner(x)
        return bmus

    def bmu_coords(self, X):
        """Return BMU coordinates in the grid (row/col for 2D, index for 1D)."""
        bmus = self.map_samples(X)
        return self.coords[bmus]

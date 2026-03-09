import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "datasets"

def load_animals(
    animals_path=DATASET_DIR / "animals.dat",
    names_path=DATASET_DIR / "animalnames.txt",
):
    props = np.loadtxt(animals_path, delimiter=",")

    with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
        names = [ln.strip().strip("'") for ln in f if ln.strip()]

    props = props.reshape(len(names), -1)
    return props, names

def load_cities(path=DATASET_DIR / "cities.dat"):
    try:
        C = np.loadtxt(
            path,
            delimiter=",",
            comments="%",
            converters={1: lambda s: float(s.strip(";"))},
        )
        if C.ndim == 1:
            raise ValueError
    except Exception:
        C = np.loadtxt(
            path,
            comments="%",
            converters={1: lambda s: float(s.strip(";"))},
        )

    assert C.shape[1] == 2, f"Expected Nx2, got {C.shape}"
    return C

def load_votes(
    votes_path=DATASET_DIR / "votes.dat",
    party_path=DATASET_DIR / "mpparty.dat",
    sex_path=DATASET_DIR / "mpsex.dat",
    district_path=DATASET_DIR / "mpdistrict.dat",
    names_path=DATASET_DIR / "mpnames.txt",
):
    votes = np.loadtxt(votes_path, delimiter=",")
    party = np.genfromtxt(party_path, comments="%")
    sex = np.genfromtxt(sex_path, comments="%")
    district = np.genfromtxt(district_path, comments="%")

    votes = votes.reshape(party.shape[0], -1)

    with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
        names = [ln.strip() for ln in f if ln.strip()]

    return votes, party.astype(int), sex.astype(int), district.astype(int), names


def build_som_grid(mp_pos, names, labels):
    """
    Returns:
        grid[(row,col)] = list of (name, label)
    """
    grid = defaultdict(list)

    for i, (r, c) in enumerate(mp_pos.astype(int)):
        grid[(r, c)].append((names[i], labels[i]))

    return grid

def plot_map(points_rc, labels, title):
    points_rc = np.asarray(points_rc, float)

    # jitter so overlapping MPs are visible
    rng = np.random.default_rng(0)
    jitter = 0.15 * (rng.random(points_rc.shape) - 0.5)

    y = points_rc[:, 0] + jitter[:, 0]  # row
    x = points_rc[:, 1] + jitter[:, 1]  # col

    plt.figure()
    plt.scatter(x, y, c=labels, s=25)   # default colormap is fine
    plt.gca().invert_yaxis()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.title(title)
    plt.grid(True, linewidth=0.5)
    plt.show()

def plot_map_with_legend(points_rc, labels, label_dict, title):
    points_rc = np.asarray(points_rc, float)

    rng = np.random.default_rng(0)
    jitter = 0.15 * (rng.random(points_rc.shape) - 0.5)

    y = points_rc[:, 0] + jitter[:, 0]
    x = points_rc[:, 1] + jitter[:, 1]

    unique_labels = np.unique(labels)

    plt.figure()

    for lab in unique_labels:
        idx = labels == lab
        plt.scatter(
            x[idx],
            y[idx],
            s=30,
            label=label_dict.get(lab, str(lab))
        )

    plt.gca().invert_yaxis()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.grid(True, linewidth=0.5)
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_with_mp_names(points_rc, names, title):
    points_rc = np.asarray(points_rc, float)

    rng = np.random.default_rng(0)
    jitter = 0.15 * (rng.random(points_rc.shape) - 0.5)

    y = points_rc[:, 0] + jitter[:, 0]
    x = points_rc[:, 1] + jitter[:, 1]

    plt.figure(figsize=(8,8))

    for xi, yi, name in zip(x, y, names):
        plt.text(xi, yi, name, fontsize=6)

    plt.gca().invert_yaxis()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.grid(True)
    plt.title(title)
    plt.show()


def plot_som_table(mp_pos, names, labels, color_dict, title):
    grid = build_som_grid(mp_pos, names, labels)

    fig, ax = plt.subplots(figsize=(12, 12))

    for r in range(10):
        for c in range(10):

            cell = grid.get((r, c), [])

            # Determine background color
            if len(cell) == 0:
                facecolor = "white"
                text = ""
            else:
                # Use label of first MP in cell for color
                label = cell[0][1]
                facecolor = color_dict.get(label, "white")

                # All names in that square
                text = "\n".join([name for name, _ in cell])

            # Draw square
            rect = Rectangle((c, r), 1, 1,
                             facecolor=facecolor,
                             edgecolor="black")
            ax.add_patch(rect)

            # Add text
            ax.text(c + 0.5, r + 0.5, text,
                    ha='center', va='center',
                    fontsize=5)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.invert_yaxis()
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.show()

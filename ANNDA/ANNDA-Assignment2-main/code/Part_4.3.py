import numpy as np
import matplotlib.pyplot as plt

from HelpersSOM import *

from SOM import SOM


if __name__ == "__main__":
    votes, party, sex, district, mp_names = load_votes()
    print(votes.shape, party.shape, sex.shape, district.shape, len(mp_names))

    som = SOM(grid_shape=(10, 10), input_dim=votes.shape[1], seed=3)
    som.train(
        votes,
        epochs=30,
        eta=0.2,
        radius_start=30,   # reasonable for a 10x10 grid
        radius_end=0,
        circular_1d=False,
        shuffle=True,
    )

    party_names = {
        0: "No party",
        1: "m",
        2: "fp",
        3: "s",
        4: "v",
        5: "mp",
        6: "kd",
        7: "c"
    }

    party_colors = {
        0: "#dddddd",
        1: "#4C72B0",  # m
        2: "#DD8452",  # fp
        3: "#55A868",  # s
        4: "#C44E52",  # v
        5: "#8172B2",  # mp
        6: "#937860",  # kd
        7: "#DA8BC3",  # c
    }

    sex_colors = {
        0: "#4C72B0",  # male
        1: "#DD8452",  # female
    }

    sex_names = {
        0: "Male",
        1: "Female"
    }

    mp_pos = som.bmu_coords(votes)  # (349,2) row/col per MP

    plot_map_with_legend(mp_pos, party, party_names, "MP voting map colored by party code")
    # plot_map_with_legend(mp_pos, sex,   sex_names, "MP voting map colored by sex code")
    # plot_map_with_legend(mp_pos, district, {}, "MP voting map colored by district code")
    # plot_map(mp_pos, district, "MP voting map colored by district code")
    # plot_with_mp_names(mp_pos, mp_names, "MP voting map with MP names")


    # plot_som_table(
    #     mp_pos,
    #     mp_names,
    #     party,
    #     party_colors,
    #     "SOM grid colored by Party"
    # )

    # plot_som_table(
    #     mp_pos,
    #     mp_names,
    #     sex,
    #     sex_colors,
    #     "SOM grid colored by Sex"
    # )

    cell_to_names = {}
    for i, (r, c) in enumerate(mp_pos.astype(int)):
        cell_to_names.setdefault((r, c), []).append(mp_names[i])

    # Print a few busiest cells
    for cell, names in sorted(cell_to_names.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(cell, len(names), names[:5], "...")
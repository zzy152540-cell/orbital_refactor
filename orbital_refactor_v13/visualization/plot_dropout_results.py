from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_dropout_status(
    timestamps,
    active_node_history,
    dropout_windows=None,
):

    count = [
        len(nodes)
        for nodes in active_node_history
    ]

    plt.figure(figsize=(8,4))

    plt.step(
        timestamps,
        count,
        where="post",
    )

    if dropout_windows:
        for start,end in dropout_windows:
            plt.axvspan(
                start,
                end,
                alpha=0.2,
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Active nodes")
    plt.title("Active Node Number")
    plt.grid(True)

    plt.show()



def plot_ci_weights(
    timestamps,
    weight_history,
):

    node_ids = set()

    for w in weight_history:
        node_ids.update(w.keys())


    plt.figure(figsize=(8,4))


    for node in node_ids:

        values=[]

        for w in weight_history:
            values.append(
                w.get(node,0)
            )

        plt.plot(
            timestamps,
            values,
            label=node,
        )


    plt.xlabel("Time (s)")
    plt.ylabel("CI weight")
    plt.title("CI Weight Evolution")
    plt.grid(True)
    plt.legend()

    plt.show()
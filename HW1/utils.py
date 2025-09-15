import matplotlib.pyplot as plt
import numpy as np
import os

def _plot_convergence(delta_seq, err_seq, tol, gamma, outpath):
    plt.figure()
    plt.plot(range(1, len(delta_seq)+1), delta_seq, label=r"$\|V_{k}-V_{k-1}\|_\infty$")
    if err_seq is not None:
        plt.plot(range(1, len(err_seq)+1), err_seq, label=r"$\|V_{k}-V^*\|_\infty$")
    plt.axhline(y=tol, linestyle="--", label=f"tol={tol:g}")
    plt.yscale("log")
    plt.xlabel("Iteration k")
    plt.ylabel("Error (sup-norm, log-scale)")
    plt.title(f"Value Iteration Convergence (gamma={gamma})")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _plot_values_bar(states, values, title, outpath):
    plt.figure()
    x = np.arange(len(states))
    plt.bar(x, values)
    plt.xticks(x, states, rotation=30, ha="right")
    plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

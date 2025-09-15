'''
2025-2 Reinforcement Learning (IIT6051)
Homework #1 
Assignment #1. student MRP examples 

Author: Jungwoo Kim (kjungwoo@yonsei.ac.kr)
'''

import os
import sys
import argparse
import numpy as np
from utils import _plot_convergence, _plot_values_bar

def q1(states, R, P, gamma=0.85, verbose=True):
    '''
    Q1. Evaluate State Value Evaluation (V(s))
    Inputs:
        states: list of states
        R: array of rewards
        P: transition probability matrix
        gamma: discount factor
    Outputs:
        V: array of state values
    '''

    # (I - γP)V = R  -> 직접해법으로 정확해
    I = np.eye(P.shape[0])
    V_exact = np.linalg.solve(I - gamma * P, R)
    
    if verbose:
        print(f"- State Values V(s) with gamma={gamma}")
        for s, v in zip(states, V_exact):
            print(f"  - {s:8s}: {v:.6f}")
    return V_exact

def q2(states, R, P, gamma=0.85, tol=1e-10, verbose=True):
    '''
    Action Value Evaluation(Q(s, a))
    Inputs:
        states: list of states
        R: array of rewards
        P: transition probability matrix
        gamma: discount factor
    Outputs:
        Q: array of action values
    '''

    S = len(states)
    V = q1(states, R, P, gamma=gamma, verbose=False)

    # Value iteration
    V_iter = np.zeros_like(R)
    iters = 0
    while True:
        iters += 1
        V_new = R + gamma * P @ V_iter
        if np.max(np.abs(V_new - V_iter)) < tol:
            V_iter = V_new
            break
        V_iter = V_new

    # Q(s, a*)
    Q = R + gamma * P @ V
    
    if verbose:
        print(f"- [Convergence] ||V_k - V_{iters-1}||_\\infty < {tol}, iters = {iters}")
        print(f"- Max |V_exact - V_iter| = {float(np.max(np.abs(V - V_iter))):.6f}")

        print("- Action-Value Q(s,a*)")
        for s, q in zip(states, Q):
            print(f"  - {s:8s}: {q: .6f} (Error: {abs(V[states.index(s)] - q):.6f})")

        if not np.allclose(V, Q):
            print("\n[WARN] Numerically, V and Q are not exactly equal.")
            exit()

    return V, Q

def q3(states, R, P, gamma=0.85, tols=[1e-6, 1e-8, 1e-10], outdir="./plots", verbose=True):
    '''
    Analysis: How many iterations are needed?
    - Actual iterations: measured by ||V_{k+1} - V_k||_\\infty < tol criterion
    - Theoretical upper bound: sufficient condition k derived from ||V_k - V*||_\\infty <= γ^k * R_max/(1-γ)
    '''

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    S = len(states)
    V_star = q1(states, R, P, gamma=gamma, verbose=False)
    Q_star = R + gamma * (P @ V_star)

    if verbose:
        _plot_values_bar(states, V_star, "State Values V*(s)", os.path.join(outdir, "values_V.png"))
        _plot_values_bar(states, Q_star, "Action-Values Q(s,a*)", os.path.join(outdir, "values_Q.png"))

    R_max = float(np.max(np.abs(R)))
    
    def theoretical_k(eps):
        if gamma >= 1.0 or R_max == 0.0:
            return None
        k = np.log(eps * (1.0 - gamma) / R_max) / np.log(gamma)
        return int(np.ceil(max(0.0, k)))

    def value_iter_with_trace(P, R, gamma, tol=1e-10):
        V = np.zeros_like(R)
        delta_seq, err_seq = [], []
        it = 0
        while True:
            it += 1
            
            V_new = R + gamma * P @ V
            
            delta = float(np.max(np.abs(V_new - V)))
            err = float(np.max(np.abs(V - V_star)))
            
            delta_seq.append(delta)
            err_seq.append(err)
            
            if delta < tol:
                return it, V_new, delta_seq, err_seq

            V = V_new
    
    results = []
    for tol in tols:
        iters, V_end, delta_seq, err_seq = value_iter_with_trace(P, R, gamma, tol)
        max_err = float(np.max(np.abs(V_end - V_star)))
        k_upper = theoretical_k(tol)

        if verbose:
            _plot_convergence(delta_seq, err_seq, tol, gamma, os.path.join(outdir, f"convergence_tol{tol:.0e}.png"))
        
        results.append({
            "tol": tol,
            "iters": iters,
            "max_err": max_err,
            "k_upper": k_upper,
            "delta_seq": delta_seq,
            "err_seq": err_seq,
            "plot_convergence_path": os.path.join(outdir, f"convergence_tol{tol:.0e}.png")
        })
        
    if verbose:
        print(f"- gamma = {gamma}, R_max = {R_max}")
        for r in results:
            print(f"  - tol = {r['tol']}, iters = {r['iters']}, max_err = {r['max_err']}, k_upper = {r['k_upper']}")
            
    return results


    # Value iteration
    V_iter = np.zeros_like(R)
    iters = 0
    while True:
        iters += 1
        V_new = R + gamma * P @ V_iter
        if np.max(np.abs(V_new - V_iter)) < tol:
            V_iter = V_new
            break
        V_iter = V_new 


if __name__ == "__main__":

    # States
    states = ["Facebook", "Class 1", "Class 2", "Class 3", "Pub", "Pass", "Sleep"]
    idx = {s:i for i, s in enumerate(states)}

    # Rewards
    R = np.array([
        -1,     # Facebook
        -2,     # Class 1
        -2,     # Class 2
        -2,     # Class 3
        +1,     # Pub
        +1,     # Pass
        0       # Sleep
    ], dtype=float)

    # Transition probability matrix
    P = np.zeros((len(states), len(states)), dtype=float)
    P[idx['Facebook'], idx['Facebook']] = 0.9
    P[idx['Facebook'], idx['Class 1']] = 0.1

    P[idx['Class 1'], idx['Facebook']] = 0.5
    P[idx['Class 1'], idx['Class 2']] = 0.5

    P[idx['Class 2'], idx['Sleep']] = 0.2
    P[idx['Class 2'], idx['Class 3']] = 0.8

    P[idx['Class 3'], idx['Pub']] = 0.4
    P[idx['Class 3'], idx['Pass']] = 0.6

    P[idx['Pub'], idx['Class 1']] = 0.2
    P[idx['Pub'], idx['Class 2']] = 0.4
    P[idx['Pub'], idx['Class 3']] = 0.4

    P[idx["Pass"], idx["Sleep"]] = 1.0

    P[idx['Sleep'], idx['Sleep']] = 1.0

    # Q1
    print(f"-----")
    print(f"Question 1")
    V1 = q1(states, R, P)
    
    # Q2
    print(f"\n-----")
    print(f"Question 2")
    V2, Q = q2(states, R, gamma=0.85, P=P, verbose=True)

    # Q3
    print(f"\n-----")
    print(f"Question 3")
    _ = q3(states, R, P, gamma=0.85, tols=[1e-6, 1e-8, 1e-10], outdir="./plots", verbose=True)

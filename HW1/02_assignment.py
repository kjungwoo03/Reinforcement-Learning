import numpy as np

def q12(states, rewards, gamma=1.0):
    """
    Calculate state values V(s) and action values Q(s,a) for 1D random walk (50% left/right policy).

    Rewards are assumed to be 'state rewards' at each time step, and the terminal states (1, N) are
    treated as 'receive reward once upon arrival then terminate'.
      - Internal states i (2..N-1):
            V(i) = r_i + γ * [0.5 * V(i-1) + 0.5 * V(i+1)]
      - Terminal states:
            V(1) = r_1,  V(N) = r_N
      - Action values (excluding terminals):
            Q(i, Left)  = r_i + γ * V(i-1)
            Q(i, Right) = r_i + γ * V(i+1)

    Args:
        states : list[int]   State labels (e.g., [1,2,3,4,5,6])
        rewards: list[float] State rewards (e.g., [100,0,0,0,0,50])
        gamma  : float       Discount factor (1.0 in assignment conditions)

    Returns:
        V  : np.ndarray (N,)  State values
        QL : np.ndarray (N,)  Left action values (NaN for terminals)
        QR : np.ndarray (N,)  Right action values (NaN for terminals)
    """
    rewards = np.asarray(rewards, dtype=float)
    N = len(states)

    # linear system
    A = np.zeros((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    # left exit
    A[0, 0] = 1.0
    b[0] = rewards[0]

    # right exit
    A[N - 1, N - 1] = 1.0
    b[N - 1] = rewards[-1]

    # internal states
    for i in range(1, N - 1):
        A[i, i] = 1.0
        A[i, i - 1] = -0.5 * gamma
        A[i, i + 1] = -0.5 * gamma
        b[i] = rewards[i]

    # linear system solution
    V = np.linalg.solve(A, b)

    # action values (Q)
    QL = np.full(N, np.nan, dtype=float)
    QR = np.full(N, np.nan, dtype=float)
    for i in range(1, N - 1):
        QL[i] = rewards[i] + gamma * V[i - 1]
        QR[i] = rewards[i] + gamma * V[i + 1]

    return V, QL, QR


def validate_random_walk(states, rewards, gamma, V, QL, QR, atol=1e-10):
    """
    Verify the calculation results satisfy the Bellman equation and boundary conditions.
    If there is a problem, raise AssertionError, and if it passes, print max |A V - b|.
    """
    rewards = np.asarray(rewards, dtype=float)
    N = len(states)

    # boundary conditions
    assert np.isclose(V[0], rewards[0], atol=atol), "left exit boundary condition violated"
    assert np.isclose(V[-1], rewards[-1], atol=atol), "right exit boundary condition violated"

    # internal states
    for i in range(1, N - 1):
        # Bellman equation: V(i) = r_i + γ * 0.5*(V(i-1)+V(i+1))
        lhs = V[i]
        rhs = rewards[i] + gamma * 0.5 * (V[i - 1] + V[i + 1])
        assert np.isclose(lhs, rhs, atol=atol), f"[Error] Bellman equation violated at state={states[i]}"

        # action values
        assert np.isclose(QL[i], rewards[i] + gamma * V[i - 1], atol=atol), f"[Error] QL violated at state={states[i]}"
        assert np.isclose(QR[i], rewards[i] + gamma * V[i + 1], atol=atol), f"[Error] QR violated at state={states[i]}"

        # random policy
        assert np.isclose(V[i], 0.5 * (QL[i] + QR[i]), atol=atol), f"[Error] V != (QL+QR)/2 @ state={states[i]}"

    # linear system with residual
    A = np.zeros((N, N), dtype=float)
    b = np.zeros(N, dtype=float)
    A[0, 0] = 1.0; b[0] = rewards[0]
    A[N - 1, N - 1] = 1.0; b[N - 1] = rewards[-1]
    for i in range(1, N - 1):
        A[i, i] = 1.0
        A[i, i - 1] = -0.5 * gamma
        A[i, i + 1] = -0.5 * gamma
        b[i] = rewards[i]
    residual = A @ V - b
    print(f"[OK] Maximum absolute residual |A * V - b| = {float(np.max(np.abs(residual))):.3e}")


if __name__ == "__main__":
    # input
    states  = [1, 2, 3, 4, 5, 6]          # state labels (1..N)
    rewards = [100, 0, 0, 0, 0, 50]        # left/right terminal rewards, internal are 0
    gamma   = 1.0                          # discount factor

    V, QL, QR = q12(states, rewards, gamma=gamma)

    print("- V(s)")
    for s, v in zip(states, V):
        print(f" - State {s}: {v:.3f}")

    print("\n- Q(s,a)")
    for i in range(len(states)):
        print(f" - State {states[i]}: QL={QL[i]:.3f}, QR={QR[i]:.3f}")

    # validation
    validate_random_walk(states, rewards, gamma, V, QL, QR)


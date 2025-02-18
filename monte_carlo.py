import numpy as np
import gym

def monte_carlo(n_samples, ep_length, alpha, gamma):
    Qsa = {}  # Dictionary for Q-values
    total_t = 0

    # Sample n times
    while total_t < n_samples:

        # 1: Generate a full episode
        s = env.reset()[0]  # For Gymnasium compatibility
        s_ep, a_ep, r_ep = [], [], []

        for t in range(ep_length):
            a = select_action(s, Qsa)
            s_next, r, done, _, _ = env.step(a)  # Unpack all returned values

            s_ep.append(s)
            a_ep.append(a)
            r_ep.append(r)

            total_t += 1
            if done or total_t >= n_samples:
                break
            s = s_next

        # 2: Update the Q-function
        g = 0.0
        for t in reversed(range(len(a_ep))):
            s, a = s_ep[t], a_ep[t]
            g = r_ep[t] + gamma * g

            # Initialize if not in Qsa
            if (s, a) not in Qsa:
                Qsa[(s, a)] = 0.0

            # Update rule
            Qsa[(s, a)] = Qsa[(s, a)] + alpha * (g - Qsa[(s, a)])

    return Qsa

def select_action(s, Qsa):
    epsilon = 0.1

    if np.random.rand() < epsilon:
        a = np.random.randint(low=0, high=env.action_space.n)  # Fix action selection
    else:
        actions = [Qsa.get((s, a), 0) for a in range(env.action_space.n)]
        a = np.argmax(actions)
    
    return a

# Correct Gym environment creation
env = gym.make('Taxi-v3')

# Run Monte Carlo learning
Q_values = monte_carlo(n_samples=10000, ep_length=100, alpha=0.1, gamma=0.99)

# Print sample Q-values
print(list(Q_values.items())[:10])  # Show first 10 Q-values
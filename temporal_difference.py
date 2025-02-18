import numpy as np
import gym


def temporal_difference(n_samples, alpha, gamma):
    Qsa = {}
    s = env.reset()[0]

    for t in range(n_samples):
        a = select_action(s, Qsa)
        s_next, r, done, _, _ = env.step(a)

        # Initialize the Q-value for state-action pair (s, a) if not present
        if (s, a) not in Qsa:
            Qsa[(s, a)] = 0.0

        # update Q function each time step with max of action values
        next_best_action = max(Qsa.get((s_next, a), 0) for a in range(env.action_space.n))
        Qsa[(s, a)] = Qsa[(s, a)] + alpha * (r + gamma * ( next_best_action - Qsa[(s, a)]) )

        if done:
            s = env.reset()[0]
        else:
            s = s_next
    
    return Qsa


def select_action(s, Qsa):
    epsilon = 0.1

    if np.random.rand() < epsilon:
        a = np.random.randint(low=0, high=env.action_space.n)  # Fix action selection
    else:
        actions = [Qsa.get((s, a), 0) for a in range(env.action_space.n)]  # Get Q-values for all actions
        a = np.argmax(actions)  # Select action with max Q-value
    
    
    return a


env = gym.make('Taxi-v3')

Q_values = temporal_difference(n_samples=10000, alpha=0.1, gamma=0.99)

print(list(Q_values.items())[:10])  # Show first 10 Q-values
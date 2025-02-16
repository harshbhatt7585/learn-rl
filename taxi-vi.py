
import gymnasium as gym
import numpy as np


# Parameters
n_rounds = 15
steps_limit = 1000


# HyperParameters
gamma = 0.9
convergence_threshold = 0.1


# This is one iteration of the value iteration function
def iterate_value_function(state_values_current, gamma, env):    
    ns = env.observation_space.n # number of states
    na = env.action_space.n # number of actions
    
    state_values_new = np.zeros(ns)
    for state_id in range(ns):
        
        # Remember the value of each action for this state
        # We will pick the highest value
        action_values = np.zeros(na)
        for action in range(na):
            
            # env.env.unwrapped.P gives us a peek into what the result of the action gives us.
            # For Taxi-v3 this will always be a single next state.
            # Other problems might have possible other states if there's randomness.
            for (prob, dst_state, reward, is_final) in env.env.unwrapped.P[state_id][action]:
                action_values[action] = prob * (reward + gamma * state_values_current[dst_state] * (not is_final))
        state_values_new[state_id] = max(action_values)
    return state_values_new


# Once the State Values are finalised, create policy based on the highest value action
# This seems redundant since it can be done in the iteration, but I'm keeping the same
# architecture as the original
def build_greedy_policy(state_values, gamma, env):    
    ns = env.observation_space.n # number of states
    na = env.action_space.n # number of actions
    
    policy = np.zeros(ns)
    for state_id in range(ns):
        
        # Remember
        action_values = np.zeros(na)
        for action in range(na):
            for (prob, dst_state, reward, is_final) in env.env.unwrapped.P[state_id][action]:
                action_values[action] = prob * (reward + gamma * v[dst_state])
                
        # Returns the indices of the maximum values along an axis
        # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        policy[state_id] = np.argmax(action_values)
    
    # Actions are whole numbers so we don't want policy to return as a float
    return policy.astype(int)


# Solve for State Value and Policy
env = gym.make("Taxi-v3", render_mode='human')
ns = env.observation_space.n
na = env.action_space.n

# List of all state values (how much value the state is worth)
# Initialise with zeroes to start with.
# We update these for each iteration
v = np.zeros(ns)
        
should_iterate = True
while should_iterate:
    # Use a copy to check for when convergence occurs
    v_old = v.copy()        
    v = iterate_value_function(v, gamma, env)        
    should_iterate = not np.all(abs(v - v_old) < convergence_threshold)
         
print(f'State Values:\n{v}')

policy = build_greedy_policy(v, gamma, env)
print(f'Policy:\n{policy}')

# Run the problem using our policy
cum_reward = 0 # Stores this to figure out the average reward
for round_count in range(n_rounds):    
    state, info = env.reset()    
    
    total_reward = 0
    for step in range(1, steps_limit + 1):
        
        # Get action from the calculated policy
        action = policy[state]    
        
        # Apply action to the environment
        state, reward, terminated, truncated, info = env.step(action)
        
        # Calculate total reward
        total_reward += reward
        
        # Provide feedback on termination
        if terminated:
            print(f'Terminated in {step} steps with reward: {reward}')            
            break
        if step == steps_limit:
            print("Did not terminate")    
        
    # Every 50 steps, print out the current average reward  
    if round_count % 50 == 0 and round_count > 0:
        average_reward = cum_reward * 1.0 / (round_count + 1)
        print(f'Average reward: {average_reward}')
        
env.close()
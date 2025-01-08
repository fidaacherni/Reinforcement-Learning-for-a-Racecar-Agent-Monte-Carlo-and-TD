import numpy as np

import utils as utl
from environment import RaceTrack
from utils import Action, ActionValueDict, DistributionPolicy, State, StateAction


def fv_mc_estimation(states : list[State], actions: list[Action], rewards: list[float], discount: float) -> ActionValueDict: 

    """
        Runs Monte-Carlo prediction for given transitions with the first visit heuristic for estimating the values

        :param list[tuple] states: list of states of an episode generated from generate_episode
        :param list[int] actions: list of actions of an episode generated from generate_episode
        :param list[float] rewards: list of rewards of an episode generated from generate_episode
        :param float discount: discount factor

        :return visited_states_returns (dict[tuple,float]): dictionary where the keys are the unique state-action combinations visited during the episode
        and the value of each key is the estimated discounted return of the first visitation of that key (state-action pair)
    """
    visited_sa_returns = {} 
    G = 0  # Initialize return

    # Loop backward through the episode to calculate returns
    for t in reversed(range(len(states))):
        state = states[t]
        action = actions[t]
        reward = rewards[t]

        # Calculate the return at time t (discounted sum of rewards)
        G = reward + discount * G

        # Check if this is the first time the state-action pair has been encountered
        sa_pair = (*state, action)
        if sa_pair not in visited_sa_returns:
            visited_sa_returns[sa_pair] = G  # Store the return for the first visit of this state-action pair

    return visited_sa_returns
'''
    episode_length = len(states)
    
    # Handle the case where the episode is empty
    if episode_length == 0:
        return visited_sa_returns

    # Compute returns for each time step
    returns = np.zeros(episode_length)
    returns[-1] = rewards[-1] if rewards else 0  # Handle empty rewards list
    for t in range(episode_length - 2, -1, -1):
        returns[t] = rewards[t] + discount * returns[t + 1]
    
    # Track first visits and compute returns
    first_visit = {}
    for t in range(episode_length):
        sa_pair = (states[t], actions[t])
        if sa_pair not in first_visit:
            first_visit[sa_pair] = t
            visited_sa_returns[sa_pair] = returns[t]

    return visited_sa_returns
'''


def fv_mc_control(env: RaceTrack, epsilon: float, num_episodes: int, discount: float) -> tuple[ActionValueDict, list[float]]:
    """
        Runs Monte-Carlo control, using first-visit Monte-Carlo for policy evaluation and regular policy improvement

        :param RaceTrack env: environment on which to train the agent
        :param float epsilon: epsilon value to use for the epsilon-greedy policy
        :param int num_episodes: number of iterations of policy evaluation + policy improvement
        :param float discount: discount factor

        :return visited_states_returns (dict[tuple,float]): dictionary where the keys are the unique state-action combinations visited during the episode
        and the value of each key is the estimated discounted return of the first visitation of that key (state-action pair)
        :return all_returns (list[float]): list of all the cumulative rewards the agent earned in each episode
    """
    # Initialize memory of estimated state-action returns
    state_action_values = utl.init_q_and_v()[1]
    all_state_action_values = {}
    all_returns = []
    valid_episodes = 0

    for episode in range(num_episodes):
        policy = utl.make_eps_greedy_policy(state_action_values, epsilon)
        
        states, actions, rewards = utl.generate_episode(policy, env)
        
        # Skip empty episodes
        if not states:
            print(f"Episode {episode} is empty. Skipping.")
            continue
        
        valid_episodes += 1
        episode_returns = fv_mc_estimation(states, actions, rewards, discount)
        
        for sa_pair, return_value in episode_returns.items():
            if sa_pair not in all_state_action_values:
                all_state_action_values[sa_pair] = []
            all_state_action_values[sa_pair].append(return_value)
            state_action_values[sa_pair] = np.mean(all_state_action_values[sa_pair])
        
        all_returns.append(sum(rewards))
        
        if episode % 100 == 0:
            print(f"Completed {episode} episodes. Valid episodes: {valid_episodes}. Current return: {all_returns[-1]}")

    print(f"Total valid episodes: {valid_episodes}")
    return state_action_values, all_returns

def is_mc_estimate_with_ratios(
    states: list[State],
    actions: list[Action],
    rewards: list[float],
    target_policy: DistributionPolicy,
    behaviour_policy: DistributionPolicy,
    discount: float
) -> dict[StateAction, list[tuple[float, float]]]:
    """
        Computes Monte-Carlo estimated q-values for each state in an episode in addition to the importance sampling ratio
        associated to that state

        :param list[tuple] states: list of states of an episode generated from generate_episode
        :param list[int] actions: list of actions of an episode generated from generate_episode
        :param list[float] rewards: list of rewards of an episode generated from generate_episode
        :param (int -> list[float]) target_policy: The initial target policy that takes in a state and returns
                                            an action probability distribution (the one we are  learning)
        :param (int -> list[float]) behavior_policy: The behavior policy that takes in a state and returns
                                            an action probability distribution
        :param float discount: discount factor

        :return state_action_returns_and_ratios (dict[tuple,list[tuple]]):
        Keys are all the states visited in the input episode
        Values is a list of tuples. The first index of the tuple is
        the IS estimate of the discounted returns
        of that state in the episode. The second index is the IS ratio
        associated with each of the IS estimates.
        i.e: if state (2, 0, -1, 1) is visited 3 times in the episode and action '7' is always taken in that state,
        state_action_returns_and_ratios[(2, 0, -1, 1, 7)] should be a list of 3 tuples.
    """

    state_action_returns_and_ratios = {}
    G = 0  # Initialize the cumulative return (MC return)
    W = 1  # Initialize the importance sampling ratio

    for t in reversed(range(len(states))):
        state = states[t]
        action = actions[t]
        reward = rewards[t]

        # Update the cumulative return (discounted rewards)
        G = reward + discount * G

        # Compute the importance sampling ratio for the current state-action pair
        target_prob = target_policy(state)[action]  # Probability of taking 'action' under the target policy π
        behaviour_prob = behaviour_policy(state)[action]  # Probability of taking 'action' under the behavior policy μ
        if behaviour_prob == 0:  # Avoid division by zero
            break
        W *= target_prob / behaviour_prob  # Update the cumulative IS ratio

        # Store the (state, action) pair and its corresponding IS return and IS ratio
        state_action = (state[0], state[1], state[2], state[3], action)
        if state_action not in state_action_returns_and_ratios:
            state_action_returns_and_ratios[state_action] = []

        # Append the tuple (IS return, IS ratio) for this state-action pair
        state_action_returns_and_ratios[state_action].append((G, W))

    return state_action_returns_and_ratios


def ev_mc_off_policy_control(env: RaceTrack, behaviour_policy: DistributionPolicy, epsilon: float, num_episodes: int, discount: float):
    # Initialize memory of estimated state-action returns
    state_action_values = utl.init_q_and_v()[1]
    all_state_action_values = {}
    all_returns = []

    # TO IMPLEMENT
    # --------------------------------
    for episode in range(num_episodes):
        # Generate an episode using the behavior policy
        states, actions, rewards = utl.generate_episode(utl.convert_to_sampling_policy(behaviour_policy), env)
        
        # Initialize cumulative return and importance sampling ratio
        G = 0
        W = 1
        
        # Create target policy (epsilon-greedy policy based on current Q-values)
        target_policy = utl.make_eps_greedy_policy_distribution(state_action_values, epsilon)
        
        # Process the episode backwards
        for t in reversed(range(len(states))):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            
            # Update return
            G = discount * G + reward
            
            # Update importance sampling ratio
            W *= target_policy(state)[action] / behaviour_policy(state)[action]
            
            # Every-visit update
            sa_pair = (*state, action)
            if sa_pair not in all_state_action_values:
                all_state_action_values[sa_pair] = []
            all_state_action_values[sa_pair].append(G)
            
            # Update Q-value estimate
            state_action_values[sa_pair] += W * (G - state_action_values[sa_pair]) / len(all_state_action_values[sa_pair])
            
            # If the importance sampling ratio is zero, we can break
            if W == 0:
                break
        
        # Store the total return for this episode
        all_returns.append(sum(rewards))
    # --------------------------------

    return state_action_values, all_returns



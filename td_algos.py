from typing import Any
from environment import RaceTrack
import utils as utl
from utils import Action, Policy, State
import numpy as np
import random


class Agent():
    def __init__(self):
        pass

    def agent_init(self, agent_init_info: dict[str, Any]):
        """
            Setup for the agent called when the experiment first starts.

            :param dict[str,Any] agent_init_info: dictionary of parameters used to initialize the agent. The dictionary must contain the following fields:

            >>> {
                seed (int): The seed to use to initialize randomness,
                env (RaceTrack): The environment to train on,
                epsilon (float): The epsilon parameter for exploration,
                step_size (float): The learning rate alpha for the TD updates,
                discount (float): The discount factor,
            }
        """
        np.random.seed(agent_init_info['seed'])
        random.seed(agent_init_info['seed'])
        # Store the parameters provided in agent_init_info.
        self.env = agent_init_info["env"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]

        # Create a dictionary for action-value estimates and initialize it to zero.
        self.sa_array, self.q, _1, _2 = utl.init_q_and_v()

    def get_current_policy(self) -> Policy:
        """
            Returns the epsilon greedy policy of the agent following the previous implementation of
            make_eps_greedy_policy

            :return policy (callable): fun(state) -> action, the current policy used by the agent based in its q-values
        """
        return utl.make_eps_greedy_policy(self.q, epsilon=self.epsilon)

    def agent_step(self, prev_state: State, prev_action: Action, prev_reward: float, current_state: State, done: bool) -> Action:
        """
            A learning step for the agent given a state, action, reward, next state and done. Both updates the agent's q-values
            from the transition `s, a, r, s', a'` and returns the action `a'` taken at `s'`

            :param tuple prev_state: the state observation from the enviromnents last step
            :param int prev_action: the action taken given prev_state
            :param float prev_reward: The reward received for taking prev_action in prev_state
            :param tuple current_state: The state received for taking prev_action in prev_state
            :param bool done: Indicator that the episode is done

            :return action (int): the action the agent is taking given current_state
        """
        raise NotImplementedError

class Sarsa(Agent):
    def agent_step(self, prev_state: State, prev_action: Action, prev_reward: float, current_state: State, done: bool) -> Action:
        # TO IMPLEMENT
        # --------------------------
        
        # Create state-action pairs
        prev_sa = (*prev_state, prev_action)
        
        # Compute the TD target
        if done:
            # If the episode is done, there is no next state-action pair
            td_target = prev_reward
        else:
            # If not done, use the Q-value of the next state-action pair
            # We need to choose the next action first
            current_action = self.get_action(current_state)
            current_sa = (*current_state, current_action)
            td_target = prev_reward + self.discount * self.q[current_sa]
        
        # Compute the TD error
        td_error = td_target - self.q[prev_sa]
        
        # Update the Q-value of the previous state-action pair
        self.q[prev_sa] += self.step_size * td_error
        
        # Choose and return the next action
        return self.get_action(current_state)

    def get_action(self, state: State) -> Action:
        if np.random.random() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.env.nA)
        else:
            # Exploit: choose the best action based on Q-values
            q_values = [self.q[(*state, a)] for a in range(self.env.nA)]
            return np.argmax(q_values)
        
    # --------------------------


class QLearningAgent(Agent):
    def agent_step(self, prev_state: State, prev_action: Action, prev_reward: float, current_state: State, done: bool) -> Action:
        # TO IMPLEMENT
        # --------------------------
        
        # Create state-action pairs
        prev_sa = (*prev_state, prev_action)
        
        # Compute the TD target
        if done:
            # If the episode is done, there is no next state-action pair
            td_target = prev_reward
        else:
            # If not done, use the maximum Q-value of the next state
            next_q_values = [self.q[(*current_state, a)] for a in range(self.env.nA)]
            td_target = prev_reward + self.discount * max(next_q_values)
        
        # Compute the TD error
        td_error = td_target - self.q[prev_sa]
        
        # Update the Q-value of the previous state-action pair
        self.q[prev_sa] += self.step_size * td_error
        
        # Choose the next action
        if np.random.random() < self.epsilon:
            # Explore: choose a random action
            action = np.random.randint(self.env.nA)
        else:
            # Exploit: choose the best action based on Q-values
            q_values = [self.q[(*current_state, a)] for a in range(self.env.nA)]
            best_actions = np.argwhere(q_values == np.max(q_values)).flatten()
            action = np.random.choice(best_actions)
        
        return action
    
    # --------------------------


def train_episode(agent: Agent, env: RaceTrack) -> tuple[list[State], list[Action], list[float]]:
    """
        Trains an agent of an environment for one episode and returns the visited states, the actions taken and the rewards obtained

        :param Agent agent: the agent to train
        :param RaceTrack env: the environment to train the agent on

        :return states (list[tuple]): the sequence of states in the generated episode
        :return actions (list[int]): the sequence of actions in the generated episode
        :return rewards (list[float]): the sequence of rewards in the generated episode
    """
    states = []
    rewards = []
    actions = []

    # TO IMPLEMENT
    # --------------------------
    # Initialize the episode
    current_state = env.reset()
    done = False
    
    # Get the first action from the agent's policy
    current_action = agent.get_current_policy()(current_state)
    
    while not done:
        # Add current state and action to lists
        states.append(current_state)
        actions.append(current_action)
        
        # Take a step in the environment
        next_state, reward, done, _ = env.step(current_action)
        
        # Add reward to list
        rewards.append(reward)
        
        # Use agent_step to get next action and update the agent
        next_action = agent.agent_step(current_state, current_action, reward, next_state, done)
        
        # Update current state and action
        current_state = next_state
        current_action = next_action
    
    # --------------------------
    
    return states, actions, rewards


def td_control(env: RaceTrack, agent_class: type[Agent], info: dict[str, Any], num_episodes: int) -> tuple[list[float, Agent]]:
    """
        Trains an agent of an environment for `num_episodes` episodes and returns cumulative rewards per episode, as well
        as the trained agent

        :param type[Agent] agent: the class of the agent to train
        :param RaceTrack env: the encironment to train the agent on

        :return all_returns (list[float]): the list of cumulative rewards per episode
        :return trained_agent (Agent): the trained agent
    """
    agent = agent_class()
    agent.agent_init(info)

    # Set seed
    seed = info['seed']
    np.random.seed(seed)
    random.seed(seed)

    all_returns = []

    for j in range(num_episodes):
        states, actions, rewards = train_episode(agent, env)
        ep_ret = np.sum(rewards)
        if j % 10 == 0:
            print(f"Episode {j}: sum of rewards = {ep_ret}, initial state: {states[0]}, last state: {states[-1]}")
        all_returns.append(ep_ret)

    return all_returns, agent
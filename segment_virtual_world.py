import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
from mpl_toolkits.mplot3d import Axes3D
import argparse

MAX_SPEED = 10  # Fixed max speed 
HEALTH_INCREASE = 40
GOAL_REWARD = 100
GOAL_POSITION = 150
LOW_HEALTH_LOSS = 1
HEALTH_LOSS_MULTIPLIER = 2


STATE_SIZE = 3 

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        # cap the memory at 10,000
        if len(self.memory) > 10000:
            self.memory = self.memory[-10000:]

        
        minibatch = random.sample(self.memory, batch_size)


        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # convert states and next_states to 2D array
        states = np.squeeze(np.array(states))
        next_states = np.squeeze(np.array(next_states))

        targets = self.target_model.predict(states, verbose=0)
        next_state_targets = self.target_model.predict(next_states, verbose=0)



        targets[np.arange(batch_size), actions] = rewards + self.gamma * np.max(next_state_targets, axis=1) * (1 - dones)

        
        self.model.fit(states, targets, epochs=1, verbose=0)

        self.update_target_model()
        

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class VirtualWorld:
    def __init__(self, max_speed, goal_position):
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.reset()

    def reset(self):
        self.agent_position = 0
        self.agent_speed = 0
        self.agent_health = 100
        self.food_position = self.generate_food_position()
        return np.array([[self.agent_position, self.agent_health, self.food_position]])

    def generate_food_position(self):
        return self.agent_position + random.randint(0, 10)

    def step(self, action):
        self.agent_speed = action
        self.agent_position += self.agent_speed

        if self.agent_speed == 0:
            self.agent_health -= LOW_HEALTH_LOSS
        else:
            self.agent_health -= self.agent_speed * HEALTH_LOSS_MULTIPLIER


        if self.agent_position >= self.food_position:
            if self.agent_position == self.food_position and action > 0:
                self.agent_health += HEALTH_INCREASE
            self.food_position = self.generate_food_position()

        if self.agent_health <= 0:
            return np.array([[self.agent_position, self.agent_health, self.food_position]]), -GOAL_REWARD, True

        done = False
        reward = 0
        if self.agent_position >= self.goal_position:
            done = True
            reward = GOAL_REWARD
        else:
            done = False
            reward = self.agent_health/100

        return np.array([[self.agent_position, self.agent_health, self.food_position]]), reward, done

def run_experiment(max_speed=MAX_SPEED, num_episodes=500, goal_position=GOAL_POSITION):
    state_size = STATE_SIZE
    action_size = max_speed + 1 # Possible actions: 0 to max_speed
    world = VirtualWorld(max_speed=max_speed, goal_position=goal_position)
    agent = DQNAgent(state_size, action_size)

    total_actions = []
    rewards = []
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = world.reset()
        state = np.reshape(state, [1, state_size])
        episode_actions = []
        episode_reward = 0
        done = False

        count = 0

        while not done and count < 100:  # Run until the goal is reached
            action = agent.act(state)
            episode_actions.append(action)
            next_state, reward, done = world.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            count += 1

        total_actions.append(np.mean(episode_actions))  # Store the mean action for the episode
        rewards.append(episode_reward)

        if len(agent.memory) > 32:  # Start training after enough experiences are accumulated
            agent.replay(32)
    
    return agent, rewards  # Return the agent and rewards for further analysis

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode')
    plt.grid(True)
    plt.show()

def plot_health_vs_position(agent):
    positions = []
    healths = []
    for state, action, reward, next_state, done in agent.memory:
        positions.append(state[0][0])
        healths.append(state[0][1])

    plt.scatter(positions, healths, alpha=0.5)
    plt.xlabel('Position')
    plt.ylabel('Health')
    plt.title('Health vs. Position')
    plt.grid(True)
    plt.show()


def plot_3d_speed_position_food(agent):
    """
    Plot a 3D scatter plot of speed, position, and food position using the agent's memory.

    Parameters:
    agent (DQNAgent): The trained DQN agent.
    """
    speeds = []
    positions = []
    food_positions = []

    # Extract data from the agent's memory
    for state, action, reward, next_state, done in agent.memory[-500:]:
        speeds.append(action)
        positions.append(state[0][0])  # Extract the position from the state
        food_positions.append(state[0][2] - state[0][0])  # Extract the food position from the state

    # Plotting the 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(positions, food_positions, speeds, c=speeds, cmap='viridis', alpha=0.5)

    ax.set_xlabel('Position')
    ax.set_ylabel('Food Position')
    ax.set_zlabel('Speed')
    ax.set_title('3D Plot of Speed, Position, and Food Position')

    fig.colorbar(scatter, ax=ax, label='Speed')
    plt.show()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_episodes", type=int, default=500)
    args = argparser.parse_args()
    
    agent, rewards = run_experiment(num_episodes=args.num_episodes)
    
    # Plotting the results
    plot_rewards(rewards)
    
    # Plotting the 3D scatter plot
    plot_3d_speed_position_food(agent)

    # plot_health_vs_position(agent)

    



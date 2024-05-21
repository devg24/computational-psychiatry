import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections

FOOD_PROBABILITY_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
MAX_SPEED = 10  # Fixed max speed 
HEALTH_INCREASE = 20
GOAL_REWARD = 50
GOAL_POSITION = 150
SEGMENT_LENGTH = 30  # Length of each segment

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
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
    def __init__(self, max_speed, goal_position, segment_length=SEGMENT_LENGTH):
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.segment_length = segment_length
        self.segments = self.create_segments()
        self.reset()

    def create_segments(self):
        num_segments = self.goal_position // self.segment_length
        return [(i * self.segment_length, (i + 1) * self.segment_length, random.choice(FOOD_PROBABILITY_VALUES))
                for i in range(num_segments)]

    def reset(self):
        self.agent_position = 0
        self.agent_speed = 0
        self.agent_health = 100
        self.current_segment = 0
        return np.array([[self.agent_position, self.agent_health, self.segments[self.current_segment][2]]])

    def step(self, action):
        self.agent_speed = action
        self.agent_position += self.agent_speed
        self.agent_health -= self.agent_speed

        if self.agent_health <= 0:
            self.reset()
            return np.array([[self.agent_position, self.agent_health, self.segments[self.current_segment][2]]]), -50, True

        if self.agent_position >= self.segments[self.current_segment][1]:
            self.current_segment += 1

        if self.agent_position >= self.goal_position:
            done = True
            reward = self.agent_health + GOAL_REWARD
        else:
            done = False
            reward = self.agent_health / 100

        next_food_density = self.segments[self.current_segment][2] if self.current_segment < len(self.segments) else 0
        return np.array([[self.agent_position, self.agent_health, next_food_density]]), reward, done

def run_experiment(max_speed=MAX_SPEED, num_episodes=500, goal_position=GOAL_POSITION):
    state_size = 3 
    action_size = max_speed + 1  # Possible actions: 0 to max_speed
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

        while not done and count < 50:  # Run until the goal is reached
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

def plot_results(food_probability_values, average_actions):
    # Plotting the results
    plt.plot(food_probability_values, average_actions, marker='o')
    plt.xlabel('Food Probability')
    plt.ylabel('Average Activity')
    plt.title('Average Activity vs Food Probability')
    plt.grid(True)
    plt.show()

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode')
    plt.grid(True)
    plt.show()

def plot_average_speed_per_segment(agent, num_segments, segment_length):
    segment_speeds = [[] for _ in range(num_segments)]
    for state, action, reward, next_state, done in agent.memory:
        position = state[0][0]
        segment_index = int(position // segment_length)
        if segment_index < num_segments:
            segment_speeds[segment_index].append(action)

    average_speeds = [np.mean(speeds) if speeds else 0 for speeds in segment_speeds]
    plt.plot(range(num_segments), average_speeds, marker='o')
    plt.xlabel('Segment')
    plt.ylabel('Average Speed')
    plt.title('Average Speed per Segment')
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

def plot_policy(agent, num_segments, segment_length):
    state_samples = []
    actions = []
    for segment_index in range(num_segments):
        for health in range(0, 101, 10):  # Sample different health values
            food_density = agent.memory[0][0][0][2]  # Example food density (you can vary this)
            state_samples.append([segment_index * segment_length, health, food_density])

    state_samples = np.array(state_samples)
    state_samples = np.reshape(state_samples, (state_samples.shape[0], 1, state_samples.shape[1]))
    actions = [agent.act(state) for state in state_samples]

    segment_positions = [state[0][0] for state in state_samples]
    health_values = [state[0][1] for state in state_samples]

    plt.scatter(segment_positions, health_values, c=actions, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Action (Speed)')
    plt.xlabel('Position')
    plt.ylabel('Health')
    plt.title('Policy Visualization')
    plt.grid(True)
    plt.show()





if __name__ == '__main__':
    agent, rewards = run_experiment(num_episodes=500)
    
    # Plotting the results
    plot_rewards(rewards)
    
    # num_segments = GOAL_POSITION // SEGMENT_LENGTH
    # plot_average_speed_per_segment(agent, num_segments, SEGMENT_LENGTH)
    # plot_health_vs_position(agent)
    # plot_policy(agent, num_segments, SEGMENT_LENGTH)


import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections



FOOD_PROBABILITY_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
MAX_SPEED = 10  # Fixed max speed for this experiment
HEALTH_INCREASE = 20
GOAL_REWARD = 50
GOAL_POSITION = 150

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9
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

        targets = self.model.predict(states, verbose=0)
        next_state_targets = self.target_model.predict(next_states, verbose=0)



        targets[np.arange(batch_size), actions] = rewards + self.gamma * np.max(next_state_targets, axis=1) * (1 - dones)

        
        self.model.fit(states, targets, epochs=1, verbose=0)

        if len(self.memory) % 100 == 0:
            self.update_target_model()
        

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay






class VirtualWorld:

    def __init__(self, max_speed, food_probability, goal_position):
        self.max_speed = max_speed
        self.food_probability = food_probability
        self.goal_position = goal_position
        self.food_locations = [pos for pos in range(self.goal_position)
                        if random.random() < self.food_probability]
        self.reset()

    def reset(self):
        self.agent_position = 0
        self.agent_speed = 0
        self.agent_health = 100
        return np.array([[self.agent_position, self.agent_health]])

    def step(self, action):
        self.agent_speed = action
        self.agent_position += self.agent_speed
        self.agent_health -= self.agent_speed

        if self.agent_health <= 0:
            self.reset()
            return np.array([[self.agent_position, self.agent_health]]), -50, True

        if self.agent_position in self.food_locations:
            self.agent_health += HEALTH_INCREASE
            self.food_locations.remove(self.agent_position)

        if self.agent_position >= self.goal_position:
            done = True
            reward = self.agent_health + GOAL_REWARD
        else:
            done = False
            reward = self.agent_health / 100

        return np.array([[self.agent_position, self.agent_health]]), reward, done

def run_experiment(food_probability, max_speed=MAX_SPEED, num_episodes=500, goal_position=GOAL_POSITION):
    state_size = 2
    action_size = max_speed + 1  # Possible actions: 0 to max_speed
    world = VirtualWorld(max_speed=max_speed, food_probability=food_probability, goal_position=goal_position)
    agent = DQNAgent(state_size, action_size)

    total_actions = []


    rewards = []
    for episode in tqdm(range(num_episodes), desc=f"Food Prob {food_probability}"):
        state = world.reset()
        state = np.reshape(state, [1, state_size])
        episode_actions = []
        episode_reward = 0
        done = False

        count = 0
        while not done:  # Run until the goal is reached
            action = agent.act(state)
            episode_actions.append(action)
            next_state, reward, done = world.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            count += 1

            if count > 1000:
                break

        total_actions.append(np.mean(episode_actions))  # Store the mean action for the episode

        # print(f"Episode {episode} Reward: {episode_reward}")

        rewards.append(episode_reward)

        if len(agent.memory) > 32:  # Start training after enough experiences are accumulated
            agent.replay(32)
        
    
    # plot rewards vs episodes
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.grid(True)
    plt.show()

    return np.mean(total_actions)  # Return the mean of mean actions over episodes


def plot_results(food_probability_values, average_actions):
    # Plotting the results
    plt.plot(food_probability_values, average_actions, marker='o')
    plt.xlabel('Food Probability')
    plt.ylabel('Average Activity')
    plt.title('Average Activity vs Food Probability')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # average_actions = []
    # for food_probability in FOOD_PROBABILITY_VALUES:
    #     average_action = run_experiment(food_probability)
    #     average_actions.append(average_action)

    # plot_results(FOOD_PROBABILITY_VALUES, average_actions)

    # Run the experiment for a single food probability and plot convergence
    average_actions = []
    food_probability = 0.1
    run_experiment(food_probability)
    


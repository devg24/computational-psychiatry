import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse

MAX_SPEED = 10  # Fixed max speed
HEALTH_INCREASE = 5
GOAL_REWARD = 100
GOAL_POSITION = 175
LOW_HEALTH_LOSS = 1
HEALTH_LOSS_MULTIPLIER = 1.25
STATE_SIZE = 3
PROBABILITIES = [i/10 for i in range(1,7)]
FOOD_REWARD = 10

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.target_freq = 150
        self.target_counter = 0
        self.target_update_tau = 0.1
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))
        return model
    
    def soft_update_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        new_weights = []
        for mw, tw in zip(main_weights, target_weights):
            new_w = self.target_update_tau * mw + (1 - self.target_update_tau) * tw
            new_weights.append(new_w)
        
        self.target_model.set_weights(new_weights)


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
        if len(self.memory) > 50000:
            self.memory = self.memory[-50000:]

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        states = np.squeeze(np.array(states))
        next_states = np.squeeze(np.array(next_states))

        targets = self.target_model.predict(states, verbose=0)
        next_state_targets = self.target_model.predict(next_states, verbose=0)

        targets[np.arange(batch_size), actions] = rewards + self.gamma * np.max(next_state_targets, axis=1) * (1 - dones)

        self.model.fit(states, targets, epochs=1, verbose=0)

        self.target_counter += 1
        # if self.target_counter % self.target_freq == 0:
        #     self.update_target_model()



        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        self.model.save(filename)

    @classmethod
    def load(cls, filename, state_size, action_size):
        agent = cls(state_size, action_size)
        agent.model = tf.keras.models.load_model(filename)
        agent.target_model = tf.keras.models.load_model(filename)
        return agent
    
class VirtualWorld:
    def __init__(self, max_speed, goal_position):
        self.max_speed = max_speed
        self.goal_position = goal_position
        
        self.training = True
        self.reset()

    def reset(self):
        self.p = random.choice(PROBABILITIES)
        self.agent_position = 0
        self.agent_speed = 0
        self.agent_health = 100
        self.food_position = self.generate_food_position()
        self.iteration = 0
        return np.array([[self.agent_position, self.agent_health, self.food_position]])

    def generate_food_position(self):
        if random.random() < self.p:
            return self.agent_position # Food at agent's position means no food
        return self.agent_position + random.randint(5,10)


    def step(self, action):
        self.iteration += 1
        if self.training and (self.iteration % 10 == 0):
            self.p = random.choice(PROBABILITIES)

        previous_speed = self.agent_speed
        self.agent_speed = action
        self.agent_position += self.agent_speed

        # Calculate health loss
        if self.agent_speed == 0:
            health_loss = LOW_HEALTH_LOSS
        else:
            health_loss = self.agent_speed * HEALTH_LOSS_MULTIPLIER
        self.agent_health -= health_loss

        base_reward = self.agent_position / self.goal_position
        speed_bonus = (self.agent_speed - previous_speed) * 0.2  
        reward = base_reward + speed_bonus
        reward = self.agent_position / self.goal_position
        done = False

        # Food collection reward
        if self.agent_position >= self.food_position:
            if self.agent_position == self.food_position and action > 0:
                self.agent_health += HEALTH_INCREASE + (self.agent_speed * HEALTH_LOSS_MULTIPLIER)
                self.agent_health = min(self.agent_health, 100)
                reward += FOOD_REWARD
            self.food_position = self.generate_food_position()
        


        # Goal reaching reward
        if self.agent_position >= self.goal_position:
            reward += GOAL_REWARD
            done = True

        # Check for death
        if self.agent_health <= 0:
            reward -= GOAL_REWARD  # Large penalty for dying
            done = True

        return np.array([[self.agent_position, self.agent_health, self.food_position]]), reward, done
    

def run_experiment(max_speed=MAX_SPEED, num_episodes=500, goal_position=GOAL_POSITION):
    state_size = STATE_SIZE
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
        while not done:  # Run until the goal is reached
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
            agent.soft_update_target_model()

    return agent, rewards  # Return the agent and rewards for further analysis

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode')
    plt.grid(True)
    plt.show()

def animate_episode(agent, world):
    state = world.reset()
    state = np.reshape(state, [1, STATE_SIZE])
    t1, t2 = 30, 45
    episode_states = []
    speeds = []
    done = False
    curr_time = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done = world.step(action)
        next_state = np.reshape(next_state, [1, STATE_SIZE])
        state = next_state
        episode_states.append(state.flatten())
        speeds.append(action)
        curr_time += 1
    
    # print why the episode ended
    if episode_states[-1][0] >= GOAL_POSITION:
        print("Goal Reached!")
    else:
        print("Agent Died!")

    fig, ax = plt.subplots()
    ax.set_xlim(0, len(speeds))
    ax.set_ylim(0, max(max(speeds), 200))  # Adjust y-axis to fit both speed and health
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed / Health')
    ax.set_title('Agent Speed and Health Over Time')

    speed_line, = ax.plot([], [], lw=2, label='Agent Speed')
    health_line, = ax.plot([], [], lw=2, label='Agent Health', color='red')
    # food_dot, = ax.plot([], [], 'go', label='Food Position')

    # plot line at x = t1 and x = t2
    ax.axvline(x=t1, color='black', linestyle='--')
    ax.axvline(x=t2, color='black', linestyle='--')

    def init():
        speed_line.set_data([], [])
        health_line.set_data([], [])
        # food_dot.set_data([], [])
        return speed_line, health_line, # food_dot

    def animate(i):
        state = episode_states[i]
        speed_data = speeds[:i+1]
        healths = [s[1] for s in episode_states[:i+1]]
        food_position = state[2]
        
        speed_line.set_data(range(i+1), speed_data)
        health_line.set_data(range(i+1), healths)
        # food_dot.set_data(i, food_position)  # Food dot at current step
        return speed_line, health_line, # food_dot

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(episode_states), interval=200, blit=True)
    plt.legend()
    plt.show()
    ani.save('episode_animation_speed.gif', writer='pillow', fps=5)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_episodes", type=int, default=500)
    args = argparser.parse_args()

    agent, rewards = run_experiment(num_episodes=args.num_episodes)

    # Save the trained agent
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    agent.save(os.path.join(save_dir, 'trained_agent.h5'))
    print(f"Agent saved to {os.path.join(save_dir, 'trained_agent.h5')}")

    # Load the agent
    # agent = DQNAgent.load('saved_models/trained_agent.h5', STATE_SIZE, MAX_SPEED + 1)
    # agent.epsilon = 0  # Set epsilon to 0 to always choose the best action
    # print("Agent loaded successfully!")

    # Plotting the results
    plot_rewards(rewards)

    # Animate a single episode with the trained agent
    world = VirtualWorld(MAX_SPEED, GOAL_POSITION)
    animate_episode(agent, world)

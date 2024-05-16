import random

FOOD_PROBABILITY = 0.01
HEALTH_DECREASE = 1
HEALTH_INCREASE = 100

class VirtualWorld:
    def __init__(self, max_speed, food_probability):
        self.max_speed = max_speed
        self.food_probability = food_probability
        self.reset()

    def reset(self):
        self.agent_position = 0
        self.agent_speed = 0
        self.agent_health = 100
        return self.agent_position, self.agent_health

    def step(self, action):
        self.agent_speed = action
        self.agent_position += self.agent_speed
        self.agent_health -= HEALTH_DECREASE

        if self.agent_health <= 0:
            self.reset()
            return self.agent_position, self.agent_health, -1, True

        if random.random() < self.food_probability:
            self.agent_health += HEALTH_INCREASE

        done = False
        reward = self.agent_health / 100

        return self.agent_position, self.agent_health, reward, done

class VirtualAgent:
    def __init__(self, max_speed):
        self.max_speed = max_speed

    # policy
    def act(self, state):
        position, health = state
        return random.randint(0, self.max_speed)

# Example usage
world = VirtualWorld(max_speed=10, food_probability=FOOD_PROBABILITY)
agent = VirtualAgent(max_speed=10)

state = world.reset()
total_reward = 0
while True:
    action = agent.act(state)
    next_state, reward, done, _ = world.step(action)
    total_reward += reward
    state = next_state
    if done:
        break

print(f"Total reward: {total_reward}")
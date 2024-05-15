import random

class VirtualWorld:
    def __init__(self, max_speed, food_probability):
        self.max_speed = max_speed
        self.food_probability = food_probability
        self.reset()

    def reset(self):
        self.agent_position = 0
        self.agent_speed = 0
        self.agent_health = 100

    def step(self, action):
        self.agent_speed = action
        self.agent_position += self.agent_speed
        self.agent_health -= self.agent_speed

        if self.agent_health <= 0:
            self.reset()
            return self.agent_position, self.agent_health, -1, True

        if random.random() < self.food_probability:
            self.agent_health += 100

        done = False
        reward = self.agent_health / 100

        return self.agent_position, self.agent_health, reward, done

class VirtualAgent:
    def __init__(self, max_speed):
        self.max_speed = max_speed

    def act(self, state):
        position, health, _, _ = state
        if health < 50:
            return self.max_speed
        else:
            return 0

# Example usage
world = VirtualWorld(max_speed=10, food_probability=0.01)
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
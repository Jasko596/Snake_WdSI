import matplotlib.pyplot as plt
import numpy as np
import time

from snakegame import *
from agent import *


game = SnakeGame()
state = game.reset()
print("Initial State:", state)
game.render()

episode_count = 0
max_steps_per_episode = 200 # Limit steps to avoid infinite loops in bad scenarios

while not game.game_over and episode_count < max_steps_per_episode:
    action = random.randint(0, 2) # Random action: 0=straight, 1=right, 2=left
    state, reward, done, score = game.step(action)
    print(f"\n--- Episode Step {episode_count + 1} ---")
    print(f"Action: {action}, Reward: {reward}, Score: {score}, Done: {done}")
    game.render()
    episode_count += 1

if game.game_over:
    print(f"\nGame Over! Final Score: {game.score}")
else:
    print(f"\nEpisode ended after {max_steps_per_episode} steps. Final Score: {game.score}")




#TRENING AGENTA

# Initialize the SnakeGame environment
game = SnakeGame(width=10, height=10)

# Initialize the QLearningAgent with specified parameters
agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01)

# Define training parameters
num_episodes = 5000
max_steps_per_episode = 500

# Lists to store training results
scores_per_episode = []
rewards_per_episode = []
epsilons = []

print(f"Starting Q-learning training for {num_episodes} episodes...")

for episode in range(num_episodes):
    current_state = game.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done and steps < max_steps_per_episode:
        action = agent.get_action(current_state)
        next_state, reward, done, score = game.step(action)
        agent.update_q_table(current_state, action, reward, next_state, done)

        episode_reward += reward
        current_state = next_state
        steps += 1

    # Store results for the episode
    scores_per_episode.append(score)
    rewards_per_episode.append(episode_reward)
    epsilons.append(agent.epsilon)

    # Decay epsilon
    agent.decay_epsilon()

    if (episode + 1) % 100 == 0 or episode == 0 or episode == num_episodes - 1:
        print(f"Episode {episode + 1}/{num_episodes} | Score: {score} | Total Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Q-table size: {len(agent.q_table)}")

print("\nTraining complete!")
print(f"Average score over {num_episodes} episodes: {sum(scores_per_episode) / num_episodes:.2f}")
print(f"Average total reward over {num_episodes} episodes: {sum(rewards_per_episode) / num_episodes:.2f}")

# Plotting results
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(scores_per_episode)
plt.title('Score per Episode')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(rewards_per_episode)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(epsilons)
plt.title('Epsilon Decay over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.grid(True)
plt.show()


#OCENA WYDAJNOŚCI AGENTA

# Create a new instance of the game environment for evaluation
# Make sure to use the same dimensions as during training
eval_game = SnakeGame(width=10, height=10)

# Set epsilon to 0 for exploitation (no exploration during evaluation)
agent.epsilon = 0.0

# Define evaluation parameters
num_eval_episodes = 100
max_steps_per_eval_episode = 500 # Should be consistent with training or set higher if needed

# Lists to store evaluation results
eval_scores = []
eval_rewards = []
eval_episode_lengths = []

print(f"\nStarting Q-learning agent evaluation for {num_eval_episodes} episodes...")

for episode in range(num_eval_episodes):
    current_state = eval_game.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done and steps < max_steps_per_eval_episode:
        action = agent.get_action(current_state) # Agent acts greedily now
        next_state, reward, done, score = eval_game.step(action)

        episode_reward += reward
        current_state = next_state
        steps += 1

    eval_scores.append(score)
    eval_rewards.append(episode_reward)
    eval_episode_lengths.append(steps)

    if (episode + 1) % 10 == 0 or episode == 0 or episode == num_eval_episodes - 1:
        print(f"Eval Episode {episode + 1}/{num_eval_episodes} | Score: {score} | Total Reward: {episode_reward:.2f} | Steps: {steps}")

print("\nEvaluation complete!")

# Calculate and print summary metrics
mean_score = np.mean(eval_scores)
mean_reward = np.mean(eval_rewards)
mean_episode_length = np.mean(eval_episode_lengths)

print(f"Mean Score over {num_eval_episodes} evaluation episodes: {mean_score:.2f}")
print(f"Mean Total Reward over {num_eval_episodes} evaluation episodes: {mean_reward:.2f}")
print(f"Mean Episode Length over {num_eval_episodes} evaluation episodes: {mean_episode_length:.2f}")

# Plotting evaluation results
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(eval_scores)
plt.title('Evaluation Score per Episode')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(eval_rewards)
plt.title('Evaluation Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)

plt.tight_layout()
plt.show()

#DEMONSTRACJA DZIAŁANIA NAUCZONEGO AGENTA


# Create a fresh game instance for demonstration
demo_game = SnakeGame(width=10, height=10)

# Set agent's epsilon to 0 for pure exploitation
agent.epsilon = 0.0

print("\n--- Demonstracja wytrenowanego agenta ---")

current_state = demo_game.reset()
done = False
steps = 0

# Limit steps to avoid extremely long demonstrations if the agent gets stuck
max_demo_steps = 1000

while not done and steps < max_demo_steps:
    print(f"\nStep: {steps}, Score: {demo_game.score}")
    demo_game.render()

    action = agent.get_action(current_state) # Get action from the trained agent
    next_state, reward, done, score = demo_game.step(action)

    current_state = next_state
    steps += 1
    time.sleep(0.1) # Small delay to make visualization readable

print("\n--- Koniec demonstracji ---")
print(f"Wynik końcowy: {demo_game.score}")
print(f"Gra zakończona: {demo_game.game_over}")
print(f"Liczba kroków: {steps}")
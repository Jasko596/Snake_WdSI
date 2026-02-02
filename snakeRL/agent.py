#IMPLEMENTACJA AGENTA
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        # Actions for Snake: 0=straight, 1=right turn, 2=left turn
        self.actions = [0, 1, 2]

    def _get_q_values(self, state):
        """Helper to get Q-values for a given state, initializing if not present."""
        state_tuple = tuple(state) # Convert list state to tuple for dictionary key
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {action: 0.0 for action in self.actions}
        return self.q_table[state_tuple]

    def get_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.choice(self.actions)
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = self._get_q_values(state)
            # If multiple actions have the same max Q-value, pick one randomly
            max_q = max(q_values.values())
            best_actions = [action for action, q_val in q_values.items() if q_val == max_q]
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, done):
        """Updates the Q-value for the (state, action) pair."""
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        current_q = self._get_q_values(state_tuple)[action]

        if done:
            max_q_next = 0.0 # No future reward if the game is over
        else:
            max_q_next = max(self._get_q_values(next_state_tuple).values())

        # Q-learning update formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)
        self.q_table[state_tuple][action] = new_q

    def decay_epsilon(self):
        """Decreases epsilon to reduce exploration over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

print("QLearningAgent class defined successfully.")
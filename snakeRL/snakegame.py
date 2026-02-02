import random
from collections import deque


#PRZYGOTOWANIE ŚRODOWISKA SNAKE
class SnakeGame:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.score = 0
        self.game_over = False
        self.snake = deque()
        self.food = None
        self.direction = (0, 1) # (dx, dy) - initial direction: right
        self.reset()

    def reset(self):
        self.score = 0
        self.game_over = False
        self.snake = deque([(self.width // 2, self.height // 2),
                            (self.width // 2 - 1, self.height // 2),
                            (self.width // 2 - 2, self.height // 2)])
        self.direction = (0, 1) # Reset to initial direction: right
        self._place_food()
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def _get_state(self):
        # Returns a simplified state for the agent
        head_x, head_y = self.snake[0]

        # Danger straight, right, left relative to current direction
        point_l = self._get_point_ahead(self.direction, 'left')
        point_r = self._get_point_ahead(self.direction, 'right')
        point_s = self._get_point_ahead(self.direction, 'straight')

        danger_straight = self._is_collision(point_s)
        danger_right = self._is_collision(point_r)
        danger_left = self._is_collision(point_l)

        # Food direction relative to head
        food_x, food_y = self.food
        food_dir_left = self.direction[0] == 0 and food_x < head_x or \
                        self.direction[1] == 0 and food_y > head_y
        food_dir_right = self.direction[0] == 0 and food_x > head_x or \
                         self.direction[1] == 0 and food_y < head_y
        food_dir_up = self.direction[1] == 0 and food_y < head_y or \
                      self.direction[0] == 0 and food_x < head_x
        food_dir_down = self.direction[1] == 0 and food_y > head_y or \
                        self.direction[0] == 0 and food_x > head_x

        # Current direction
        dir_l = self.direction == (0, -1) # Left
        dir_r = self.direction == (0, 1)  # Right
        dir_u = self.direction == (-1, 0) # Up
        dir_d = self.direction == (1, 0)  # Down

        state = [
            danger_straight,
            danger_right,
            danger_left,
            food_dir_left,
            food_dir_right,
            food_dir_up,
            food_dir_down,
            dir_l,
            dir_r,
            dir_u,
            dir_d
        ]
        return state

    def _get_point_ahead(self, current_direction, turn):
        x, y = self.snake[0]
        dx, dy = current_direction

        if turn == 'straight':
            return (x + dx, y + dy)
        elif turn == 'right':
            # Rotate direction 90 degrees clockwise
            if dx == 0 and dy == 1: return (x + 1, y) # Right -> Down
            if dx == 0 and dy == -1: return (x - 1, y) # Left -> Up
            if dx == 1 and dy == 0: return (x, y - 1) # Down -> Left
            if dx == -1 and dy == 0: return (x, y + 1) # Up -> Right
        elif turn == 'left':
            # Rotate direction 90 degrees counter-clockwise
            if dx == 0 and dy == 1: return (x - 1, y) # Right -> Up
            if dx == 0 and dy == -1: return (x + 1, y) # Left -> Down
            if dx == 1 and dy == 0: return (x, y + 1) # Down -> Right
            if dx == -1 and dy == 0: return (x, y - 1) # Up -> Left
        return (x,y) # Should not happen

    def _is_collision(self, point):
        x, y = point
        # Wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # Self collision
        if point in list(self.snake)[1:]:
            return True
        return False

    def step(self, action):
        # action: 0=straight, 1=right turn, 2=left turn
        reward = 0

        # Update direction based on action
        if action == 1: # Right turn
            if self.direction == (0, 1): self.direction = (1, 0)   # R -> D
            elif self.direction == (0, -1): self.direction = (-1, 0) # L -> U
            elif self.direction == (1, 0): self.direction = (0, -1) # D -> L
            elif self.direction == (-1, 0): self.direction = (0, 1) # U -> R
        elif action == 2: # Left turn
            if self.direction == (0, 1): self.direction = (-1, 0)  # R -> U
            elif self.direction == (0, -1): self.direction = (1, 0)  # L -> D
            elif self.direction == (1, 0): self.direction = (0, 1)   # D -> R
            elif self.direction == (-1, 0): self.direction = (0, -1) # U -> L

        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Check for game over conditions
        if self._is_collision(new_head):
            self.game_over = True
            reward = -100
            return self._get_state(), reward, self.game_over, self.score

        self.snake.appendleft(new_head)

        # Check if food was eaten
        if new_head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1 # Small penalty for every step to encourage efficiency

        return self._get_state(), reward, self.game_over, self.score

    def render(self):
        # Optional: A simple text-based rendering for debugging
        board = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.snake:
            board[y][x] = 'S'
        board[self.snake[0][1]][self.snake[0][0]] = 'H' # Head
        if self.food:
            board[self.food[1]][self.food[0]] = 'F'

        for row in board:
            print(' '.join(row))
        print(f"Score: {self.score}, Game Over: {self.game_over}")

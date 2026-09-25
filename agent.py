import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(14, 256, 3)
        self.target_model = Linear_QNet(14, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(
            self.model, self.target_model, lr=LR, gamma=self.gamma
        )
        self.target_update_counter = 0
        self.target_update_freq = 500

    def _sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # точки straight / right / left под текущее направление
        if dir_r:
            p_s, p_r, p_l = point_r, point_d, point_u
        elif dir_l:
            p_s, p_r, p_l = point_l, point_u, point_d
        elif dir_u:
            p_s, p_r, p_l = point_u, point_r, point_l
        else:  # dir_d
            p_s, p_r, p_l = point_d, point_l, point_r

        free_s, free_r, free_l = self._get_fill_parts(game, p_s, p_r, p_l)

        state = [
            # danger straight
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # danger left
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # free spaces (0..1)
            free_s,
            free_r,
            free_l,

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]
        return np.array(state, dtype=float)

    def _get_fill_percentage(self, pt, occupied, cols, rows):
        start = (int(pt.x // BLOCK_SIZE), int(pt.y // BLOCK_SIZE))

        if not (0 <= start[0] < cols and 0 <= start[1] < rows):
            return 0.0
        if start in occupied:
            return 0.0

        visited = set()
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows \
                   and (nx, ny) not in visited and (nx, ny) not in occupied:
                    queue.append((nx, ny))

        return len(visited) / (cols * rows)

    def _get_fill_parts(self, game, p_s, p_r, p_l):
        cols = game.width // BLOCK_SIZE
        rows = game.height // BLOCK_SIZE

        # тело без хвоста — хвост сдвинется на следующем шаге
        occupied = set()
        for p in game.snake[:-1]:
            occupied.add((int(p.x // BLOCK_SIZE), int(p.y // BLOCK_SIZE)))

        fill_s = self._get_fill_percentage(p_s, occupied, cols, rows)
        fill_r = self._get_fill_percentage(p_r, occupied, cols, rows)
        fill_l = self._get_fill_percentage(p_l, occupied, cols, rows)
        return fill_s, fill_r, fill_l

    # ---------- MEMORY ----------
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self._sync_target()

    def get_action(self, state):
        self.epsilon = max(0.01, 1.0 - self.n_games / 150)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)
            if agent.n_games % 10 == 0:
                loss = agent.trainer.train_step(state_old, final_move, reward, state_new, done)
                print(f"loss = {loss:.4f}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
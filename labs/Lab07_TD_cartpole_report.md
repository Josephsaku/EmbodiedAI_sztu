# Lab07 TD CartPole 实验报告

## 1. 实验环境

本实验使用 `conda` 环境 `rl_env` 运行。实验中安装了 `pygame`，因此 `CartPole-v1` 的 `rgb_array` 渲染可以正常生成动画帧。

## 2. CartPole 环境观察

### 代码

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib import animation
from gymnasium.error import DependencyNotInstalled

# Use rgb_array so env.render() returns frames when pygame is available.
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)

# Simple rule: push in the direction that reduces pole angle.
# If theta > 0 (pole tilts to the right), push right; else push left.
def greedy_policy(observation):
    _, _, theta, theta_dot = observation
    return 1 if theta > 0 else 0

frames = []
rewards = []
num_steps = 100

try:
    # capture initial frame
    frames.append(env.render())

    for t in range(num_steps):
        action = greedy_policy(obs)
        obs, r, terminated, truncated, info = env.step(action)
        rewards.append(r)
        frames.append(env.render())
        if terminated or truncated:
            # If episode ends early, reset and keep going until 100 frames collected
            obs, info = env.reset()
            frames.append(env.render())

    # Stack frames into a numpy array (T, H, W, C), dtype=uint8
    frames = np.asarray(frames, dtype=np.uint8)
    print(f"Frames shape: {frames.shape}, dtype: {frames.dtype}")

    fig, ax = plt.subplots()
    ax.set_axis_off()
    img = ax.imshow(frames[0])

    def animate(i):
        img.set_data(frames[i])
        return [img]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=30, blit=True)
    plt.close(fig)  # prevent duplicate static display
    display(HTML(ani.to_jshtml()))
except DependencyNotInstalled as exc:
    print("CartPole animation skipped because pygame is not installed in rl_env.")
    print(str(exc))
finally:
    env.close()

# Recreate a non-rendering environment for the learning experiments below.
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)
```

### 输出

```text
Frames shape: (103, 400, 600, 3), dtype: uint8
<IPython.core.display.HTML object>

/Users/alaia/.matplotlib is not a writable directory
Matplotlib created a temporary cache directory at /var/folders/g0/_bjcdf_n70q6kkjqjb7wz46c0000gn/T/matplotlib-suoxyn4i because there was an issue with the default path (/Users/alaia/.matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Matplotlib is building the font cache; this may take a moment.
/opt/miniconda3/envs/rl_env/lib/python3.10/site-packages/pygame/pkgdata.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_stream, resource_exists

<CartPole animation rendered with pygame in rl_env>
```

### 理解

输出说明 CartPole 动画成功渲染，帧数组形状为 `(103, 400, 600, 3)`，表示共 103 帧，每帧大小为 400×600，RGB 三通道，类型为 `uint8`。这是合理的，因为初始帧加上 100 步交互，中间若 episode 提前结束会额外 reset 并加入新帧。Matplotlib 和 pygame 的 warning 是本地缓存目录和依赖 API 提醒，不影响算法结果。

## 3. 状态离散化

### 代码

```python
import numpy as np

n_actions = env.action_space.n
np.set_printoptions(precision=3, suppress=True)

# ----- Discretization -----
NUM_BINS = np.array([6, 6, 12, 12])
STATE_BOUNDS = np.array([
    [-2.4,   2.4],
    [-3.0,   3.0],
    [-0.418, 0.418],
    [-2.0,   2.0]
])

def discretize_state(obs):
    lo, hi = STATE_BOUNDS[:,0], STATE_BOUNDS[:,1]
    ratios = (np.clip(obs, lo, hi) - lo) / (hi - lo)
    bins = (ratios * NUM_BINS).astype(int)
    return tuple(np.clip(bins, 0, NUM_BINS - 1))
```

```python
# now print the new discrete state space
obs, _ = env.reset(seed=0)
discrete_state = discretize_state(obs)
print("Observation:", obs)
print("Discretized State:", discrete_state)
```

### 输出

```text
Observation: [ 0.014 -0.023 -0.046 -0.048]
Discretized State: (np.int64(3), np.int64(2), np.int64(5), np.int64(5))
```

### 理解

这个输出是合理的。原始 observation 是连续值，离散化后得到 4 个整数 bin index。初始位置和速度都接近 0，因此离散索引落在各维中间附近，符合 CartPole 初始状态分布。

## 4. ε-greedy 策略

### 代码

```python
# ----- ε-greedy policy -----
def greedy_action(Q, s, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[s]))
```

### 输出

```text
无文本输出。
```

### 理解

这个函数只定义策略，不打印内容，所以没有输出是正确的。它以 `epsilon` 概率随机探索，否则选择当前 Q 值最大的动作。

## 5. Monte Carlo 控制

### 代码

```python
# ----- Initialize tables -----
Q = np.zeros((*NUM_BINS, n_actions))
N = np.zeros((*NUM_BINS, n_actions), dtype=int)


num_episodes = 5000
gamma = 0.99
epsilon = 1.0
eps_min, eps_decay = 0.05, 0.995
returns_history = []


for ep in range(1, num_episodes + 1):
    obs, _ = env.reset()
    episode = []

    done = False
    while not done:
        s = discretize_state(obs)
        a = greedy_action(Q, s, epsilon)
        obs_next, r, term, trunc, _ = env.step(a)
        episode.append((s, a, r))
        obs = obs_next
        done = term or trunc

    T = len(episode)
    G = 0.0
    returns = np.zeros(T)
    for t in range(T - 1, -1, -1):
        _, _, r = episode[t]
        G = gamma * G + r
        returns[t] = G

    for t, (s, a, _) in enumerate(episode):
        
        N[s][a] += 1
        n = N[s][a]
        Q[s][a] += (returns[t] - Q[s][a]) / n


    epsilon = max(eps_min, epsilon * eps_decay)
    ep_return = sum(r for _, _, r in episode)
    returns_history.append(ep_return)

    if ep % 500 == 0:
        avg = np.mean(returns_history[-100:])
        print(f"Episode {ep:5d} | ε={epsilon:.3f} | AvgReturn(100)={avg:.1f}")
```

```python
def evaluate_greedy(Q, episodes=20):
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            s = discretize_state(obs)
            a = np.argmax(Q[s])
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += r
            done = term or trunc
        total += ep_ret
    return total / episodes

avg_eval = evaluate_greedy(Q)
print(f"\nGreedy policy average return: {avg_eval:.1f}")
```

### 输出

```text
Episode   500 | ε=0.082 | AvgReturn(100)=467.0
Episode  1000 | ε=0.050 | AvgReturn(100)=492.2
Episode  1500 | ε=0.050 | AvgReturn(100)=488.3
Episode  2000 | ε=0.050 | AvgReturn(100)=478.9
Episode  2500 | ε=0.050 | AvgReturn(100)=491.3
Episode  3000 | ε=0.050 | AvgReturn(100)=488.0
Episode  3500 | ε=0.050 | AvgReturn(100)=498.8
Episode  4000 | ε=0.050 | AvgReturn(100)=485.0
Episode  4500 | ε=0.050 | AvgReturn(100)=492.9
Episode  5000 | ε=0.050 | AvgReturn(100)=495.2

Greedy policy average return: 499.5
```

### 理解

这个答案是对的，而且效果很好。CartPole-v1 的单次 episode 最高回报是 500，MC 的训练后 100 回合平均回报接近 500，贪心策略评估平均回报为 499.5，说明离散化后的 Q 表已经学到了几乎最优的平衡策略。由于 MC 使用完整 episode 回报，训练后期表现高但仍会有轻微波动，这是正常现象。

## 6. n-step SARSA

### 代码

```python
# ----- Tables & hyperparams -----
Q = np.zeros((*NUM_BINS, n_actions))
num_episodes = 4000
max_steps = 1000
gamma = 0.99
alpha = 0.1
n = 4                           # n-step horizon (tune me)
epsilon = 1.0
eps_min, eps_decay = 0.05, 0.995

returns_hist = []

for ep in range(1, num_episodes + 1):
    obs, _ = env.reset()
    s0 = discretize_state(obs)
    a0 = greedy_action(Q, s0, epsilon)

    # Buffers (index from 0); r[t] stores r_{t}, so shift by 1 for clarity with algorithm
    S = [s0]            # states S_0, S_1, ...
    A = [a0]            # actions A_0, A_1, ...
    R = [0.0]           # R[0] unused; will append R_1, R_2, ...
    T = np.inf

    t = 0
    ep_return = 0.0

    while True:
        if t < T:
            # Step in env using A_t
            obs_next, r, term, trunc, _ = env.step(A[t])
            ep_return += r
            done = bool(term or trunc)
            R.append(r)                       # this is R_{t+1}
            if done:
                T = t + 1
            else:
                s_next = discretize_state(obs_next)
                S.append(s_next)              # S_{t+1}
                a_next = greedy_action(Q, s_next, epsilon)
                A.append(a_next)              # A_{t+1}

        tau = t - n + 1                       # state to update
        if tau >= 0:
            # Compute G (n-step return starting at tau)
            # G = sum_{i=tau+1}^{min(tau+n, T)} gamma^{i-(tau+1)} R_i

            G = 0.0
            upper = int(min(tau + n, T))
            power = 0
            for i in range(tau + 1, upper + 1):
                G += (gamma ** power) * R[i]
                power += 1
            if tau + n < T:                   # bootstrap if within episode
                
                G += (gamma ** n) * Q[S[tau + n]][A[tau + n]]

            s_tau = S[tau]
            a_tau = A[tau]
            Q[s_tau][a_tau] += alpha * (G - Q[s_tau][a_tau])

        if tau == T - 1:
            break
        t += 1

    # ε schedule & logging
    epsilon = max(eps_min, epsilon * eps_decay)
    returns_hist.append(ep_return)
    if ep % 500 == 0:
        avg = np.mean(returns_hist[-100:])
        print(f"Episode {ep:5d} | ε={epsilon:.3f} | AvgReturn(100)={avg:.1f}")
```

```python
avg_eval = evaluate_greedy(Q, episodes=20)
print(f"\nGreedy policy average return (n={n} SARSA): {avg_eval:.1f}")
```

### 输出

```text
Episode   500 | ε=0.082 | AvgReturn(100)=199.2
Episode  1000 | ε=0.050 | AvgReturn(100)=207.3
Episode  1500 | ε=0.050 | AvgReturn(100)=195.6
Episode  2000 | ε=0.050 | AvgReturn(100)=192.1
Episode  2500 | ε=0.050 | AvgReturn(100)=203.2
Episode  3000 | ε=0.050 | AvgReturn(100)=187.5
Episode  3500 | ε=0.050 | AvgReturn(100)=280.0
Episode  4000 | ε=0.050 | AvgReturn(100)=207.7

Greedy policy average return (n=4 SARSA): 127.4
```

### 理解

实现公式是对的：它累加从 `tau+1` 到 `min(tau+n,T)` 的折扣奖励，并在 episode 未结束时加入 `gamma**n * Q[S[tau+n], A[tau+n]]`。不过结果不如 MC，贪心评估只有 127.4。这个结果仍然是合理的，因为 CartPole 被粗粒度离散化后状态别名较严重，SARSA 又是 on-policy，训练中保留 `epsilon=0.05` 的探索，学习到的策略会更保守、更受探索轨迹影响。若调小学习率、增加 episode、改进离散化或调 `n`，结果可能进一步提高。

## 7. One-step Q-learning

### 代码

```python
Q = np.zeros((*NUM_BINS, n_actions), dtype=float)
num_episodes = 4000
gamma = 0.99
alpha = 0.1
epsilon = 1.0
eps_min, eps_decay = 0.05, 0.995
returns_hist = []

for ep in range(1, num_episodes + 1):
    obs, _ = env.reset()
    done = False
    ep_return = 0.0

    while not done:
        s = discretize_state(obs)
        a = greedy_action(Q, s, epsilon)

        obs_next, r, term, trunc, _ = env.step(a)
        done = bool(term or trunc)
        ep_return += r

        s_next = discretize_state(obs_next)
        
        if done: 
            td_target = r
        else:
            td_target = r + gamma * np.max(Q[s_next])
            
        Q[s][a] += alpha * (td_target - Q[s][a])

        obs = obs_next

    # ε schedule & logging
    epsilon = max(eps_min, epsilon * eps_decay)
    returns_hist.append(ep_return)
    if ep % 500 == 0:
        avg = np.mean(returns_hist[-100:])
        print(f"Episode {ep:5d} | ε={epsilon:.3f} | AvgReturn(100)={avg:.1f}")
```

```python
avg_eval = evaluate_greedy(Q, episodes=20)
print(f"\nGreedy policy average return (1-step Q-learning): {avg_eval:.1f}")
```

### 输出

```text
Episode   500 | ε=0.082 | AvgReturn(100)=18.2
Episode  1000 | ε=0.050 | AvgReturn(100)=37.0
Episode  1500 | ε=0.050 | AvgReturn(100)=223.8
Episode  2000 | ε=0.050 | AvgReturn(100)=233.4
Episode  2500 | ε=0.050 | AvgReturn(100)=217.7
Episode  3000 | ε=0.050 | AvgReturn(100)=172.0
Episode  3500 | ε=0.050 | AvgReturn(100)=199.8
Episode  4000 | ε=0.050 | AvgReturn(100)=202.0

Greedy policy average return (1-step Q-learning): 148.8
```

### 理解

Q-learning 更新公式是正确的：非终止状态使用 `r + gamma * max_a Q(s_next,a)`，终止状态只使用即时奖励 `r`。训练结果中后期有明显提升，但最终评估只有 148.8，没有达到 MC 的接近满分。这也是合理的：one-step TD 对离散化质量、学习率和探索策略更敏感，当前粗离散表格会让不同连续状态共享同一个 Q 值，造成学习不稳定。它比随机策略好很多，但还没有达到最优。

## 8. 自定义 3×3 GridWorld

### 代码

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ----------------------------
# 1) Custom 3x3 Grid Environment
# ----------------------------
class Grid3x3Env(gym.Env):
    """
    A tiny 3x3 GridWorld.
      - Start at (0,0), goal at (2,2)
      - Actions: 0=Up, 1=Down, 2=Left, 3=Right
      - Rewards: +1 on reaching goal, -0.01 per step, -0.05 for bumping into a wall (stay in place)
      - Episode ends on goal or step limit
    Observation: Discrete(9) index r*3 + c
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None, max_steps=30):
        super().__init__()
        self.N = 3
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.N * self.N)

        self.start = (0, 0)
        self.goal  = (2, 2)
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._pos = None
        self._steps = 0

        # (dr, dc) for Up, Down, Left, Right
        self._moves = [(-1,0), (1,0), (0,-1), (0,1)]

    def _rc_to_obs(self, r, c): return r * self.N + c
    def _obs_to_rc(self, obs):  return divmod(obs, self.N)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = self.start
        self._steps = 0
        return self._rc_to_obs(*self._pos), {}

    def step(self, action):
        self._steps += 1
        r, c = self._pos
        dr, dc = self._moves[int(action)]
        nr, nc = r + dr, c + dc

        reward = -0.01
        terminated = False
        truncated = self._steps >= self.max_steps

        # check bounds
        if 0 <= nr < self.N and 0 <= nc < self.N:
            self._pos = (nr, nc)
        else:
            # bump wall: stay & extra penalty
            reward -= 0.05

        if self._pos == self.goal:
            reward = 1.0
            terminated = True

        return self._rc_to_obs(*self._pos), reward, terminated, truncated, {}

    def render(self):
        board = [[" . "]*self.N for _ in range(self.N)]
        gr, gc = self.goal
        board[gr][gc] = "[G]"
        r, c = self._pos
        board[r][c] = " A "
        return "\n".join("".join(row) for row in board)
```

```python
# ----------------------------
# Random trajectory demo
# ----------------------------
import time

env = Grid3x3Env()
obs, _ = env.reset()
done = False

print("Initial state:")
print(env.render(), "\n")

for t in range(10):  # up to 10 random steps
    if done:
        break
    a = env.action_space.sample()
    obs, r, term, trunc, _ = env.step(a)
    done = term or trunc
    print(f"Step {t+1}, Action={a}, Reward={r:.2f}")
    print(env.render(), "\n")
    time.sleep(0.5)

if done:
    print("Episode ended.")
else:
    print("Trajectory finished without termination.")
```

### 输出

```text
Initial state:
 A  .  . 
 .  .  . 
 .  . [G] 

Step 1, Action=2, Reward=-0.06
 A  .  . 
 .  .  . 
 .  . [G] 

Step 2, Action=1, Reward=-0.01
 .  .  . 
 A  .  . 
 .  . [G] 

Step 3, Action=0, Reward=-0.01
 A  .  . 
 .  .  . 
 .  . [G] 

Step 4, Action=3, Reward=-0.01
 .  A  . 
 .  .  . 
 .  . [G] 

Step 5, Action=1, Reward=-0.01
 .  .  . 
 .  A  . 
 .  . [G] 

Step 6, Action=0, Reward=-0.01
 .  A  . 
 .  .  . 
 .  . [G] 

Step 7, Action=2, Reward=-0.01
 A  .  . 
 .  .  . 
 .  . [G] 

Step 8, Action=2, Reward=-0.06
 A  .  . 
 .  .  . 
 .  . [G] 

Step 9, Action=2, Reward=-0.06
 A  .  . 
 .  .  . 
 .  . [G] 

Step 10, Action=2, Reward=-0.06
 A  .  . 
 .  .  . 
 .  . [G] 

Trajectory finished without termination.
```

### 理解

随机轨迹没有到达终点是合理的。动作 2 表示向左，在起点或最左列向左会撞墙，因此奖励是 `-0.01 - 0.05 = -0.06`，输出中多次出现 `-0.06` 正好验证了撞墙惩罚逻辑。

## 9. GridWorld 上的 Q-learning

### 代码

```python
# ----------------------------
# Q-learning on the custom GridWorld
# ----------------------------
def epsilon_greedy_grid(Q, state, epsilon, env):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(Q[state]))

np.random.seed(0)
env = Grid3x3Env()
Q_grid = np.zeros((env.observation_space.n, env.action_space.n), dtype=float)

num_episodes = 500
gamma = 0.95
alpha = 0.2
epsilon = 1.0
eps_min, eps_decay = 0.05, 0.98
grid_returns = []

for ep in range(1, num_episodes + 1):
    obs, _ = env.reset()
    done = False
    ep_return = 0.0

    while not done:
        action = epsilon_greedy_grid(Q_grid, obs, epsilon, env)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_return += reward

        if done:
            target = reward
        else:
            target = reward + gamma * np.max(Q_grid[next_obs])
        Q_grid[obs, action] += alpha * (target - Q_grid[obs, action])
        obs = next_obs

    epsilon = max(eps_min, epsilon * eps_decay)
    grid_returns.append(ep_return)

    if ep % 100 == 0:
        avg = np.mean(grid_returns[-100:])
        print(f"Episode {ep:3d} | epsilon={epsilon:.3f} | AvgReturn(100)={avg:.3f}")
```

```python
# Show the learned greedy policy.
action_symbols = np.array(["U", "D", "L", "R"])
policy = action_symbols[np.argmax(Q_grid, axis=1)].reshape(3, 3)
policy[2, 2] = "G"
print("Learned greedy policy:")
for row in policy:
    print(" ".join(row))

print("\nState-action values Q_grid:")
print(np.round(Q_grid, 3))
```

```python
# Evaluate one deterministic greedy trajectory from the start.
env = Grid3x3Env()
obs, _ = env.reset()
done = False
total_reward = 0.0
trajectory = [obs]

action_names = ["Up", "Down", "Left", "Right"]
for step in range(10):
    action = int(np.argmax(Q_grid[obs]))
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    trajectory.append(obs)
    print(f"Step {step + 1}: Action={action_names[action]}, Reward={reward:.2f}, NextState={obs}")
    if terminated or truncated:
        done = True
        break

print(f"\nTrajectory: {trajectory}")
print(f"Total reward: {total_reward:.2f}")
print("Reached goal:", done and obs == env._rc_to_obs(*env.goal))
```

### 输出

```text
Episode 100 | epsilon=0.133 | AvgReturn(100)=0.859
Episode 200 | epsilon=0.050 | AvgReturn(100)=0.963
Episode 300 | epsilon=0.050 | AvgReturn(100)=0.967
Episode 400 | epsilon=0.050 | AvgReturn(100)=0.966
Episode 500 | epsilon=0.050 | AvgReturn(100)=0.966

Learned greedy policy:
D D L
R D D
R R G

State-action values Q_grid:
[[ 0.711  0.829  0.698  0.82 ]
 [ 0.521  0.883  0.148  0.306]
 [ 0.009  0.179  0.626 -0.012]
 [ 0.743  0.854  0.773  0.883]
 [ 0.813  0.94   0.801  0.749]
 [ 0.076  0.945  0.552  0.229]
 [ 0.452  0.545  0.509  0.939]
 [ 0.762  0.858  0.822  1.   ]
 [ 0.     0.     0.     0.   ]]

Step 1: Action=Down, Reward=-0.01, NextState=3
Step 2: Action=Right, Reward=-0.01, NextState=4
Step 3: Action=Down, Reward=-0.01, NextState=7
Step 4: Action=Right, Reward=1.00, NextState=8

Trajectory: [0, 3, 4, 7, 8]
Total reward: 0.97
Reached goal: True
```

### 理解

GridWorld 的结果是正确的。最短路径需要 4 步到达目标，前三步每步 `-0.01`，最后到达目标 `+1.00`，总奖励为 `0.97`，和输出完全一致。学到的策略从起点选择 Down、Right、Down、Right，可以在 4 步内到达终点；虽然某些非最短路径状态的动作看起来不唯一，但从起点出发的贪心轨迹是最优的。

## 10. 总结

本实验完成了 CartPole 上的 Monte Carlo、n-step SARSA、one-step Q-learning，以及一个自定义 3×3 GridWorld 环境和其 Q-learning 训练。MC 在当前离散化配置下效果最好，接近 CartPole 满分；SARSA 和 Q-learning 的公式实现正确，但受粗离散化、学习率和探索策略影响，表现中等。GridWorld 结果清晰验证了自定义环境和 Q-learning 更新逻辑。

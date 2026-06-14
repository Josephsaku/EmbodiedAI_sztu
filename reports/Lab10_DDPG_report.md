## 一、代码单元格

实验中需要重点理解并实际运行的训练单元格如下（共约 68 行）。该单元格完成了 DDPG 的完整训练流程，包含动作选择、环境交互、经验存储、网络更新、目标网络软更新等关键步骤。

```python
episode = 0

for step in range(total_steps):

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = actor(state_tensor).cpu().data.numpy().flatten()

    if step < warmup_steps:
        action = env.action_space.sample()
    else:
        action = (action + np.random.normal(0, exploration_noise, size=act_dim)).astype(np.float32)

    action = np.clip(action, -max_action, max_action)

    # --- step env ---
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    buffer.push(state, action, reward, next_state, float(done))

    state = next_state
    episode_reward += reward
    episode_length += 1
    
    # --- reset if done ---
    if done:
        print(f"Episode {episode} | Reward: {episode_reward:.1f} | Lenght: {episode_length}")
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode += 1

    # --- update ---
    if len(buffer) > batch_size:

        s, a, r, s2, d = buffer.sample(batch_size)
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s2 = s2.to(device)
        d = d.to(device)

        # Critic update
        with torch.no_grad():
            a2 = actor_target(s2)
            q_target = r + gamma * (1 - d) * critic_target(s2, a2)

        q_val = critic(s, a)
        critic_loss = nn.MSELoss()(q_val, q_target)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Actor update
        actor_loss = -critic(s, actor(s)).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # Target update
        for p, p_t in zip(actor.parameters(), actor_target.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

        for p, p_t in zip(critic.parameters(), critic_target.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    # --- occasionally show progress ---
    if step % 10_000 == 0:
        print(f"Step {step}/{total_steps}")
        torch.save(actor.state_dict(), SAVE_PATH)
        
print("Training finished and model saved!")
```


## 二、输出为什么是对的

训练过程中终端会输出类似以下内容：

```
Episode 0 | Reward: 12.3 | Lenght: 23
Episode 1 | Reward: 8.7 | Lenght: 17
...
Episode 1000 | Reward: 408.2 | Lenght: 152
...
Step 10000/300000
Step 20000/300000
...
```

### 判断输出正确的依据

1. **Reward 和 Length 从低到高变化**
   - 随机动作时 Reward 通常只有 1 ~ 50，Length 只有十几到几十步。
   - 随着训练进行，Reward 逐渐上升到几百、上千，Length 也逐渐变长。
   - 这说明智能体正在学习如何控制机器人保持平衡并向前移动。

2. **Length 是稳定性的直接指标**
   - `Length` 表示当前回合持续了多少步。
   - Hopper-v4 单回合上限为 1000 步，Length 越接近 1000，说明机器人越不容易摔倒。
   - Length 持续上升意味着策略在变得稳定。

3. **Step 进度正常推进**
   - 每 10,000 步打印一次 `Step X/300000`，说明训练循环在正常运行。
   - 模型也会定期保存到 `ddpg_hopper_actor_class.pth`。

4. **Loss 没有发散**
   - Critic loss 和 Actor loss 虽然不会直接打印，但只要 Reward 和 Length 持续上升，就说明网络更新方向正确，没有发散。

## 三、训练结果

本次训练共完成约 2044 个 episode，总步数为 300,000 步。训练末期部分 episode 输出如下：

```text
Episode 2032 | Reward: 1376.9 | Lenght: 478
Episode 2033 | Reward: 1930.0 | Lenght: 632
Episode 2034 | Reward: 1507.9 | Lenght: 558
Episode 2035 | Reward: 1454.2 | Lenght: 540
Episode 2036 | Reward: 1333.4 | Lenght: 459
Episode 2037 | Reward: 8.3    | Lenght: 9
Episode 2038 | Reward: 950.0  | Lenght: 357
Episode 2039 | Reward: 2601.3 | Lenght: 1000
Episode 2040 | Reward: 711.7  | Lenght: 224
Episode 2041 | Reward: 2083.0 | Lenght: 824
Episode 2042 | Reward: 2518.8 | Lenght: 1000
Episode 2043 | Reward: 2656.1 | Lenght: 1000
Training finished and model saved!
```

### 结果分析

- **最终模型已收敛**：最后两个 episode 的 `Length` 均达到 1000，说明机器人能够稳定跑满整个回合，模型已经学会了保持平衡。
- **奖励水平较高**：最后两个 episode 的 `Reward` 分别为 **2518.8** 和 **2656.1**，显著高于训练初期的几十到几百，说明智能体不仅能站稳，还能有效向前跳跃。
- **存在一定波动**：训练后期仍有个别 episode 表现较差（如 `Reward: 8.3, Lenght: 9`），这是 DDPG 在训练过程中保留的探索噪声以及环境随机性导致的正常现象。
- **模型保存位置**：`ddpg_hopper_actor_class.pth`

> 如果绘制了奖励曲线图和长度曲线图，可以插入在此处：
>
> ```markdown
> ![训练奖励曲线](reward_curve.png)
> ![训练回合长度曲线](length_curve.png)
> ```

## 七、实验总结

本实验成功实现了 DDPG 算法在 Hopper-v4 连续控制任务上的训练。通过 Actor-Critic 架构、目标网络和经验回放，智能体能够在连续动作空间中进行有效学习。训练输出的 Reward 和 Length 指标能够直观反映学习效果：当两者持续上升并趋于稳定时，说明训练成功。

后续可以通过加载保存的 Actor 模型进行可视化测试，观察机器人是否能够稳定向前跳跃。

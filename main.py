import torch
import gym
import copy
import numpy as np
from algorithm.util.net import QValueNetwork, PolicyNetwork
from algorithm.util.replaybuffer import ReplayBuffer

def adjust(next_state, reward, min_x, max_x):
    if min_x is None or max_x is None:
        return reward, next_state[0], next_state[0]

    position, velocity = next_state

    if position > max_x:
        # reward += (position - max_x) * 10 + 2
        max_x = position
    elif position < min_x:
        # reward += (min_x - position) * 5 + 1
        min_x = position

    if position > 0.5:
        return 100, min_x, max_x

    button_x = -0.5
    distance = abs(position - button_x)
    reward += (distance + 10*abs(velocity))  * 10
    return reward, min_x, max_x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = gym.make("MountainCar-v0", render_mode="human")
env = gym.make("MountainCar-v0")
# env = gym.make("CartPole-v1")
# env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128
gamma = 0.99
episilon = 0.1
batch_size = 128
num_episodes = 1000

q_net = QValueNetwork(state_dim, hidden_dim)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
q_target_net = copy.deepcopy(q_net)
policy_target_net = copy.deepcopy(policy_net)
q_target_net.load_state_dict(q_net.state_dict())
policy_target_net.load_state_dict(policy_net.state_dict())

q_net.to(device)
policy_net.to(device)
q_target_net.to(device)
policy_target_net.to(device)
q_target_net.eval()
policy_target_net.eval()


q_optimizer = torch.optim.Adam(q_net.parameters(), lr=0.001)
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001)

buffer = ReplayBuffer(state_dim, action_dim, capacity=200*100)

positions = []
velocities = []


for i in range(num_episodes):

    state = env.reset()[0].reshape(1, -1)
    min_x, max_x = None, None
    max_v = -1
    _actions = []
    _states = []
    while True:
        _states.append((float(state[0][0]), float(state[0][1])))
        with torch.no_grad():
            if np.random.rand() < episilon:
                action = env.action_space.sample()
            else:
                state = torch.from_numpy(state).float().to(device)
                action_prob = policy_target_net(state)
                action_dist = torch.distributions.Categorical(action_prob)
                action = action_dist.sample().item()
                state = state.cpu().numpy()

        _actions.append(action)
        next_state, reward, done, terminal, _ = env.step(action)
        reward, min_x, max_x = adjust(next_state, reward, min_x, max_x)

        if max_v < abs(next_state[1]):
            max_v = abs(next_state[1])

        action = np.array(action).reshape(1, -1)
        reward = np.array(reward).reshape(1, -1)
        next_state = np.array(next_state.reshape(1, -1))
        done = np.array(done).reshape(1, -1)

        buffer.store(state, action, reward, next_state, done)
        if done or terminal:
            break
        state = next_state
    
    positions.append(max_x)
    velocities.append(max_v)
    
    if len(buffer) > 1000:
        batch = buffer.sample(batch_size)
        states = torch.from_numpy(batch[0]).float().to(device)
        actions = torch.from_numpy(batch[1]).long().to(device)
        rewards = torch.from_numpy(batch[2]).float().to(device)
        next_states = torch.from_numpy(batch[3]).float().to(device)
        dones = torch.from_numpy(batch[4]).float().to(device)


        q_values = q_net(states)
        with torch.no_grad():
            next_actions_prob = policy_target_net(next_states)
            next_actions = torch.argmax(next_actions_prob, dim=1, keepdim=True)
            next_q_values = q_target_net(next_states)
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
            td_error = target_q_values - q_values
        
        
        q_loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        log_probs  = torch.log(policy_net(states).gather(1, actions))
        policy_loss = torch.mean(-log_probs * td_error)

        q_optimizer.zero_grad()
        policy_optimizer.zero_grad()
        q_loss.backward()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        q_optimizer.step()
        policy_optimizer.step()

        if i % 100 == 0:
            print(f"Iteration: {i}, Avg Position: {np.mean(positions):.4f}, Max Position: {max(positions):.4f}, Avg Velocity: {np.mean(velocities):.4f}, Max Velocity: {max(velocities):.4f}, Q_Loss: {q_loss.item():.4f}, P_Loss: {policy_loss.item():.4f}")
            ""
            positions = []
            q_target_net.load_state_dict(q_net.state_dict())
            policy_target_net.load_state_dict(policy_net.state_dict())
            torch.save(q_net.state_dict(), f"ckpt/q_net_I{i}.pth")
            torch.save(policy_net.state_dict(), f"ckpt/policy_net_I{i}.pth")

    else:
        print("Iteration:", i, "Buffer Length:", len(buffer))


        
        


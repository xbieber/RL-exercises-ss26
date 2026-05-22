from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
import matplotlib.pyplot as plt


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed random number generators for reproducibility.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment to seed.
    seed : int, optional
        Seed value for NumPy, PyTorch, and environment (default is 0).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

def moving_average(values, window=10):

    if len(values) < window:
        return values

    weights = np.ones(window) / window

    return np.convolve(values, weights, mode="valid")

#   -----------------
#       REINFORCE
#   -----------------

class Policy(nn.Module):
    """
    Multi-layer perceptron mapping states to action probabilities.

    Implements a linear feed-forward network with one hidden layer and softmax output.

    Parameters
    ----------
    state_space : gym.spaces.Box
        Observation space defining the dimensionality of inputs.
    action_space : gym.spaces.Discrete
        Action space defining number of output classes.
    hidden_size : int, optional
        Number of units in the hidden layer (default is 128).
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_size: int = 128,
    ):
        """
        Initialize the policy network.

        Parameters
        ----------
        state_space : gym.spaces.Box
            Observation space of the environment.
        action_space : gym.spaces.Discrete
            Action space of the environment.
        hidden_size : int, optional
            Number of hidden units. Defaults to 128.
        """
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute action probabilities for given state(s).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (state_dim,) or (batch_size, state_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities over actions, shape (batch_size, n_actions).
        """
        # TODO: Apply fc1 followed by ReLU (Flatten input if needed)
        # TODO: Apply fc2 to get logits
        # TODO: Return softmax over logits along the last dimension

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.view(x.size(0), -1) # flatten
        x = torch.relu(self.fc1(x))

        logits = self.fc2(x)
        probs = torch.softmax(logits, dim=-1)

        return probs



class REINFORCEAgent(AbstractAgent):
    """
    REINFORCE agent performing on-policy Monte Carlo policy gradient updates.

    Wraps an MLP policy network and optimizer, providing train, predict, save, load, and evaluate methods.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment for interaction.
    lr : float, optional
        Learning rate for optimizer (default is 1e-2).
    gamma : float, optional
        Discount factor for returns (default is 0.99).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-2,
        gamma: float = 0.99,
        seed: int = 0,
        hidden_size: int = 128,
    ) -> None:
        """
        Initialize the REINFORCE agent.

        Args:
            env (gym.Env): Environment for training.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            seed (int, optional): Random seed. Defaults to 0.
        """

        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.total_episodes = 0

        # plotting
        self.train_returns = []
        self.eval_returns = []

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action according to the current policy.

        In training mode, samples stochastically and returns log probability.
        In evaluation mode, returns the argmax action deterministically.

        Parameters
        ----------
        state : np.ndarray
            Current observation from the environment.
        info : dict, optional
            Additional info (unused here, default is empty).
        evaluate : bool, optional
            If True, use deterministic policy (default is False).

        Returns
        -------
        action : int
            Selected action index.
        info_out : dict
            Contains 'log_prob' if in training mode; empty if evaluating.
        """
        # TODO: Pass state through the policy network to get action probabilities
        # If evaluate is True, return the action with highest probability
        # Otherwise, sample from the action distribution and return the log-probability as a key in the dictionary (Hint: use torch.distributions.Categorical)

        state_t = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state_t).squeeze(0)

        if evaluate:
            action = torch.argmax(probs).item()
            return action, {}

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), {'log_prob': log_prob}

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted reward-to-go for each timestep.

        Parameters
        ----------
        rewards : list of float
            Sequence of rewards for one episode.

        Returns
        -------
        torch.Tensor
            Discounted returns tensor of shape (len(rewards),).
        """
        # TODO: Initialize running return R = 0
        # TODO: Iterate over rewards and compute the return-to-go:
        #       - Update R = r + gamma * R
        #       - Insert R at the beginning of the returns list
        # TODO: Convert the list of returns to a torch.Tensor and return
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32)

    def update_agent(
        self,
        training_batch: List[
            Tuple[np.ndarray, int, float, np.ndarray, bool, Dict[str, Any]]
        ],
    ) -> float:
        """
        Perform a policy-gradient update using one full episode.

        Parameters
        ----------
        training_batch : list of tuples
            Each tuple is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            Scalar loss value after update.
        """
        # unpack log_probs & rewards
        log_probs = [t[5]["log_prob"] for t in training_batch]
        rewards = [t[2] for t in training_batch]

        # compute discounted returns
        returns_t = self.compute_returns(rewards)

        # normalize advantages
        # TODO: Normalize advantages with mean and standard deviation,
        # and add 1e-8 to the denominator to avoid division by zero
        # advantages = returns_t
        advantages = (returns_t - returns_t.mean()) / ( returns_t.std() + 1e-8)


        lp_tensor = torch.stack(log_probs)
        loss = -torch.sum(lp_tensor * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str) -> None:
        """
        Save policy network and optimizer state to a checkpoint.

        Parameters
        ----------
        path : str
            File path to save checkpoint.
        """
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load policy network and optimizer state from checkpoint.

        Parameters
        ----------
        path : str
            File path of checkpoint to load.
        """
        ckpt = torch.load(path)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        """
        Evaluate policy over multiple episodes.

        Parameters
        ----------
        eval_env : gym.Env
            Environment for evaluation.
        num_episodes : int, optional
            Number of episodes to run (default is 10).

        Returns
        -------
        mean_return : float
            Average episode return.
        std_return : float
            Standard deviation of returns.
        """
        self.policy.eval()
        returns: List[float] = []
        # TODO: rollout num_episodes in eval_env and aggregate undiscounted returns across episodes

        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            total_return = 0.0

            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                next_state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_return += reward
                state = next_state

            returns.append(total_return)

        self.policy.train()  # Set back to training mode

        # TODO: Return the mean and std of the returns across episodes
        mean = np.mean(returns) if returns else 0.0
        std = np.std(returns) if returns else 0.0
        return mean, std

    def train(
        self,
        num_episodes: int,
        eval_interval: int = 10,
        eval_episodes: int = 5,
    ) -> None:
        """
        Train the agent on-policy for a number of episodes.

        Parameters
        ----------
        num_episodes : int
            Total number of training episodes.
        eval_interval : int, optional
            Frequency of evaluation prints (default is 10).
        """
        eval_env = gym.make(self.env.spec.id)  # fresh copy for eval
        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            batch: List[Tuple[Any, ...]] = []

            while not done:
                action, info = self.predict_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                batch.append((state, action, float(reward), next_state, done, info))
                state = next_state

            loss = self.update_agent(batch)
            total_return = sum(r for _, _, r, *_ in batch)
            self.train_returns.append(total_return)
            self.total_episodes += 1

            if ep % 10 == 0:
                print(f"[Train] Ep {ep:3d} Return {total_return:5.1f} Loss {loss:.3f}")

            if ep % eval_interval == 0:
                mean_ret, std_ret = self.evaluate(eval_env, num_episodes=eval_episodes)
                self.eval_returns.append(mean_ret)
                print(f"[Eval ] Ep {ep:3d} AvgReturn {mean_ret:5.1f} ± {std_ret:4.1f}")

        print("Training complete.")

#   -----------------
#       DDGP
#   -----------------
# AI Disclaimer
# The AI was used to adapt the code to the formatting, structure,
# and style of the provided example code


from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity: int = 100000):

        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states, dtype=np.float32))
        actions = torch.from_numpy(np.array(actions, dtype=np.float32))
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1)
        # m
        return states, actions, rewards, next_states, dones

    def __len__(self):

        return len(self.buffer)

class Actor(nn.Module):

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 256,
    ):

        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.max_action = float(action_space.high[0])

        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = torch.tanh(self.fc3(x))

        return self.max_action * x

class Critic(nn.Module):

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 256,
    ):

        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:

        if state.dim() == 1:
            state = state.unsqueeze(0)

        if action.dim() == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.fc3(x)

class DDPGAgent(AbstractAgent):

    def __init__(
        self,
        env: gym.Env,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 64,
        noise_std: float = 0.1,
        seed: int = 0,
        hidden_size: int = 256,
    ):

        set_seed(env, seed)

        self.env = env

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.noise_std = noise_std

        # actor
        self.actor = Actor(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        self.target_actor = Actor(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        # critic
        self.critic = Critic(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        self.target_critic = Critic(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        # copy weights
        self.target_actor.load_state_dict(
            self.actor.state_dict()
        )

        self.target_critic.load_state_dict(
            self.critic.state_dict()
        )

        # optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.total_episodes = 0

        # plotting
        self.train_returns = []
        self.eval_returns = []

    def predict_action(
            self,
            state: np.ndarray,
            info: Dict[str, Any] = {},
            evaluate: bool = False,
    ):
        state_t = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state_t).detach().numpy()[0]

        if not evaluate:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise

        max_action = self.env.action_space.high[0]
        action = np.clip( action, -max_action, max_action)

        return action, {}

    def update_agent(self) -> Tuple[float, float]:

        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        ( states, actions, rewards, next_states, dones,
        ) = self.replay_buffer.sample(self.batch_size)

        # Critic target
        with torch.no_grad():

            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * target_q

        # Critic update
        current_q = self.critic(
            states,
            actions,
        )

        critic_loss = nn.MSELoss()(
            current_q,
            y,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        actor_loss = -self.critic(
            states,
            self.actor(states),
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Softttarget update
        for param, target_param in zip(
                self.actor.parameters(),
                self.target_actor.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data
                + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
                self.critic.parameters(),
                self.target_critic.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data
                + (1 - self.tau) * target_param.data
            )

        return (
            float(actor_loss.item()),
            float(critic_loss.item()),
        )

    def evaluate(
            self,
            eval_env: gym.Env,
            num_episodes: int = 10,
    ):

        returns = []

        for _ in range(num_episodes):

            state, _ = eval_env.reset()

            done = False

            total_return = 0.0

            while not done:
                action, _ = self.predict_action(
                    state,
                    evaluate=True,
                )

                next_state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_return += reward
                state = next_state

            returns.append(total_return)

        mean = np.mean(returns)
        std = np.std(returns)

        return mean, std

    def train(
            self,
            num_episodes: int,
            eval_interval: int = 10,
            eval_episodes: int = 5,
    ):

        eval_env = gym.make(self.env.spec.id)

        for ep in range(1, num_episodes + 1):

            state, _ = self.env.reset()
            done = False
            episode_return = 0.0

            while not done:
                action, _ = self.predict_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc

                self.replay_buffer.push(state, action, reward, next_state, done)
                actor_loss, critic_loss = self.update_agent()
                episode_return += reward

                state = next_state

            self.train_returns.append(episode_return)
            self.total_episodes += 1

            if ep % 10 == 0:
                print(
                    f"[Train] Ep {ep:3d} "
                    f"Return {episode_return:7.1f} "
                    f"ActorLoss {actor_loss:.3f} "
                    f"CriticLoss {critic_loss:.3f}"
                )

            if ep % eval_interval == 0:
                mean_ret, std_ret = self.evaluate(eval_env, num_episodes=eval_episodes,)
                self.eval_returns.append(mean_ret)

                print(
                    f"[Eval ] Ep {ep:3d} "
                    f"AvgReturn {mean_ret:7.1f} ± {std_ret:5.1f}"
                )

        print("Training complete.")



@hydra.main(
    config_path="../configs/agent/", config_name="reinforce", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training with Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with fields:
          env:
            name: str        # Gym environment id
          seed: int
          agent:
            lr: float
            gamma: float
            hidden_size: int
          train:
            episodes: int
            eval_interval: int
            eval_episodes: int
    """
    #   -----------------
    #       REINFORCE
    #   -----------------

    # Initialize environment and seed
    print(f"config: {cfg}")
    reinforce_env = gym.make(cfg.env.name)
    # REINFORCE needs discrete environment conflict with DDPG, needs continuously environment
    # Easy solution different environments
    # TODO better solution?
    set_seed(reinforce_env, cfg.seed)

    # Instantiate agent with hyperparameters from config
    reinforce_agent  = REINFORCEAgent(
        env=reinforce_env,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
    )

    # Train agent
    reinforce_agent.train(
        num_episodes=cfg.train.episodes,
        eval_interval=cfg.train.eval_interval,
        eval_episodes=cfg.train.eval_episodes,
    )

    #   -----------------
    #       DDGP
    #   -----------------

    # Initialize environment and seed
    ddpg_env = gym.make("Pendulum-v1")
    set_seed(ddpg_env, cfg.seed)

    # Instantiate agent with hyperparameters
    ddpg_agent = DDPGAgent(
        env=ddpg_env,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=64,
        noise_std=0.1,
        seed=cfg.seed,
        hidden_size=256,
    )

    # Train agent
    ddpg_agent.train(
        num_episodes=cfg.train.episodes,
        eval_interval=cfg.train.eval_interval,
        eval_episodes=cfg.train.eval_episodes,
    )

    # Plotting
    plt.figure(figsize=(12, 6))

    reinforce_smoothed = moving_average(reinforce_agent.train_returns, window=10, )
    ddpg_smoothed = moving_average(ddpg_agent.train_returns, window=10, )

    plt.plot(reinforce_smoothed, label="REINFORCE", )
    plt.plot(ddpg_smoothed, label="DDPG", )

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DDPG vs REINFORCE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ddpg_vs_reinforce.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()

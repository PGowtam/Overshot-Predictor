import torch
import torch.nn.functional as F
import numpy as np
from Agents.iql_networks import QNetwork, ValueNetwork


class IQLAgent:
    def __init__(
        self,
        obs_dim,
        action_dim=2,
        hidden_dim=256,
        gamma=0.99,
        expectile=0.7,
        lr=3e-4,
        device="cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.expectile = expectile

        self.q = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.v = ValueNetwork(obs_dim, hidden_dim).to(device)

        self.q_target = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=lr)

    def train_epoch(self, df, batch_size=256):
        N = len(df)
        idxs = np.random.permutation(N)
        losses = []

        for i in range(0, N, batch_size):
            batch_idxs = idxs[i:i + batch_size]
            batch_data = df.iloc[batch_idxs]

            batch = {
                "obs": torch.tensor(np.stack(batch_data["obs"]), dtype=torch.float32).to(self.device),
                "next_obs": torch.tensor(np.stack(batch_data["next_obs"]), dtype=torch.float32).to(self.device),
                "actions": torch.tensor(batch_data["action"].values, dtype=torch.long).to(self.device),
                "rewards": torch.tensor(batch_data["reward"].values, dtype=torch.float32).to(self.device),
                "dones": torch.tensor(batch_data["done"].values, dtype=torch.float32).to(self.device),
            }

            metrics = self.update(batch)
            losses.append(metrics["q_loss"] + metrics["v_loss"])

        return losses

    def state_dict(self):
        return {
            "q": self.q.state_dict(),
            "v": self.v.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "v_opt": self.v_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.q.load_state_dict(state_dict["q"])
        self.v.load_state_dict(state_dict["v"])
        self.q_target.load_state_dict(state_dict["q_target"])
        self.q_opt.load_state_dict(state_dict["q_opt"])
        self.v_opt.load_state_dict(state_dict["v_opt"])

    def expectile_loss(self, diff):
        weight = torch.where(
            diff > 0,
            self.expectile,
            1 - self.expectile
        )
        return weight * diff.pow(2)

    def update(self, batch):
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        acts = batch["actions"]
        rews = batch["rewards"]
        dones = batch["dones"]

        # -----------------------
        # V UPDATE
        # -----------------------
        with torch.no_grad():
            q_vals = self.q(obs)
            q_taken = q_vals.gather(1, acts.unsqueeze(1)).squeeze(1)

        v = self.v(obs)
        v_loss = self.expectile_loss(q_taken - v).mean()

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # -----------------------
        # Q UPDATE
        # -----------------------
        with torch.no_grad():
            v_next = self.v(next_obs)
            target = rews + self.gamma * (1 - dones) * v_next

        q_pred = self.q(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_pred, target)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # Soft target update
        tau = 0.005
        for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item()
        }

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        q = self.q(obs).detach().cpu().numpy()[0]

        action = int(q.argmax())
        confidence = float(abs(q[1] - q[0]))

        return action, confidence

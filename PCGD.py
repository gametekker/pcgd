import torch
import torch.nn as nn

#math
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

# Polymatrix Competitive Gradient Descent (PCGD)
class PCGD:
    def __init__(self, policies, eta):
        self.policies = policies
        self.eta = eta
        self.agents = ["adversary_0", "agent_0", "agent_1"]

    # Empty gradients
    def zero_grad(self):
      for agent in self.agents:
          for param in self.policies[agent].parameters():
              if param.grad is not None:
                  param.grad.detach()
                  param.grad.zero_()

    # Flatten tensors into 1D array
    def custom_flatten(self, gp):
        flattened = []
        for g in gp:
            flattened.append(g.flatten())
        return torch.concat(flattened)

    # Generate matrix of losses for PCGD Hessian
    def loss_matrix(self, log_probs, cum_rewards):
        agents = self.agents
        losses = torch.zeros((len(agents), len(agents)))
        for row in range(len(agents)):
            for col in range(len(agents)):
                losses[row, col] = (cum_rewards[agents[row]] * log_probs[agents[col]] * log_probs[agents[row]]).mean()
        return losses

    # Vector of first derivatives of loss (recouping SimGD)
    def zeta(self, log_probs, cum_rewards):
        self.zero_grad()
        agents = self.agents
        zeta = []
        for row in agents:
            reward = (log_probs[row] * cum_rewards[row]).mean()
            grads = torch.autograd.grad(reward, self.policies[row].parameters(), retain_graph=True, create_graph=True)
            zeta.append(self.custom_flatten(grads))
        return torch.concat(zeta)

    # Matrix-vector product for A = (I + nH_o) for solving the linear system in
    # PCGD numerically using Krylov subspace methods
    def mvp(self, loss_mat, vec):
        self.zero_grad()
        vec = vec.reshape(-1, 1)
        agents = self.agents
        split = sum(p.numel() for p in self.policies[agents[0]].parameters())
        split2 = sum(p.numel() for p in self.policies[agents[1]].parameters())
        blocks = [vec[:split], vec[split:split+split2], vec[split+split2:]]
        new_blocks = []
        for row in range(len(agents)):
            acc = blocks[row].clone()
            for col in range(len(agents)):
                if row != col:
                    reward = loss_mat[row, col]

                    if agents[col] in self.policies:
                        # Your existing code that accesses self.policies[agents[col]]
                        grads = self.custom_flatten(torch.autograd.grad(reward, self.policies[agents[col]].parameters(), retain_graph=True, create_graph=True)).reshape(-1, 1)
                    else:
                        print(f"Key error: {agents[col]} not found in self.policies")

                    vjp = self.custom_flatten(torch.autograd.grad(grads, self.policies[agents[row]].parameters(), [blocks[col]], retain_graph=True)).reshape(-1, 1)
                    acc += self.eta * vjp
            new_blocks.append(acc)
        return torch.concat(new_blocks)

    # Solve for PCGD parameter update
    def compute_loss_mat_update_iterative(self, loss_mat, zeta):
        mv = lambda v: self.mvp(loss_mat, torch.tensor(v)).detach().numpy()
        A = LinearOperator((zeta.shape[0], zeta.shape[0]), matvec=mv)
        b = zeta.detach().numpy()
        return self.eta * torch.tensor(gmres(A, b)[0])

    # Magic to safely update parameters
    def update_parameters(self, update):
        agents = self.agents
        split = sum(p.numel() for p in self.policies[agents[0]].parameters())
        split2 = sum(p.numel() for p in self.policies[agents[1]].parameters())
        first = update[:split]
        second = update[split:split+split2]
        third = update[split+split2:]
        grad_like_policy = []
        idx = 0
        for param in self.policies[agents[0]].parameters():
            grad_like_policy.append(first[idx : idx + torch.numel(param)].reshape(param.shape))
            idx += torch.numel(param)
        for param, grad in zip(self.policies[agents[0]].parameters(), grad_like_policy):
            param.data += grad

        grad_like_policy = []
        idx = 0
        for param in self.policies[agents[1]].parameters():
            grad_like_policy.append(second[idx : idx + torch.numel(param)].reshape(param.shape))
            idx += torch.numel(param)
        for param, grad in zip(self.policies[agents[1]].parameters(), grad_like_policy):
            param.data += grad

        grad_like_policy = []
        idx = 0
        for param in self.policies[agents[2]].parameters():
            grad_like_policy.append(third[idx : idx + torch.numel(param)].reshape(param.shape))
            idx += torch.numel(param)
        for param, grad in zip(self.policies[agents[2]].parameters(), grad_like_policy):
            param.data += grad
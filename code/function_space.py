import torch
import torch.nn as nn
import torch.nn.functional as F
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

class FunctionSpaceRegularizer:
    def __init__(self, input_dim, temperature=2.0):
        super(FunctionSpaceRegularizer, self).__init__()
        self.temperature = temperature

    def distillation_loss(self, student_outputs, teacher_outputs):
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_outputs / self.temperature, dim=1)
        return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)

    def gaussian_process_loss(self, model_outputs, target_outputs, x):
        kernel = ScaleKernel(RBFKernel())
        K = kernel(x, x).evaluate()
        diff = model_outputs - target_outputs
        return torch.mm(torch.mm(diff.t(), torch.inverse(K + 1e-5 * torch.eye(K.size(0)))), diff)

class DualRepresentation(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DualRepresentation, self).__init__()
        self.function_mapper = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gp_mean = ConstantMean()
        self.gp_kernel = ScaleKernel(RBFKernel())

    def forward(self, x):
        param_output = x
        func_embedding = self.function_mapper(x)
        mean = self.gp_mean(func_embedding)
        covar = self.gp_kernel(func_embedding)
        func_output = MultivariateNormal(mean, covar)
        return param_output, func_output

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Dict, List, Optional
from torch import Tensor
class ParallelLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd #(cast_inputs=torch.float32)
    def forward(ctx, input, expert_size, weight, bias=None):
        output = ParallelLinear.forward_scriptable(input, expert_size, weight, bias)
        # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
        ctx.save_for_backward(input, expert_size, weight, bias)
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(input: Tensor, expert_size: Tensor,
                           weight: Tensor, bias: Optional[Tensor]):
        output_buf: Tensor = torch.empty((input.size(0), weight.size(2)),
                                         device=input.device, dtype=input.dtype)
        num_linears = weight.size(0)

        expert_size_list: List[int] = expert_size.tolist()
        # print('expert_size: ', expert_size)
        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)

        for i in range(num_linears):
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_linears):
                output_buf_list[i].add_(bias[i])

        output = output_buf
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, expert_size, weight, bias = ctx.saved_tensors
        return ParallelLinear.backward_scriptable(
            grad_out, input, expert_size,
            weight, bias
        )

    @staticmethod
    @torch.jit.script
    def backward_scriptable(grad_out: Tensor,
                 input: Tensor, expert_size: Tensor,
                 weight: Tensor, bias: Optional[Tensor]):
        num_linears = weight.size(0)
        expert_size_list: List[int] = expert_size.tolist()
        input_list = input.t().split(expert_size_list, dim=1)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        weight_t = weight.permute(0, 2, 1)

        for i in range(num_linears):
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
            torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

        d_input = d_input_buf
        d_weight = d_weight_buf

        if bias is not None:
            d_bias_buf = torch.empty_like(bias)
            for i in range(num_linears):
                torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias

@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts
class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.bias = None
        self.reset_parameters()

    def extra_repr(self):
        return 'num_experts={}, input_size={}, output_size={}'.format(
            self.weight.size(0), self.weight.size(1), self.weight.size(2))

    def reset_parameters(self) -> None:
        # std = math.sqrt(2.0 / float(self.weight.size(1) + self.weight.size(2)))
        # a = math.sqrt(3.0) * std
        nn.init.uniform_(self.weight, -1. / self.weight.size(1), 1. / self.weight.size(1))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, expert_size):
        results = ParallelLinear.apply(inputs, expert_size, self.weight, self.bias)
        return results



class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, head_size, num_experts, k,
                 cvloss=0, switchloss=0, zloss=0,
                 bias=True, gating_activation=None,
                 activation=None, noisy_gating=True, usage_mem = 10000,
                 acc_aux_loss=False):
        super(MoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.activation = activation
        # self.usage = np.random.randint(num_experts, size=(usage_mem, k))
        # self.cur = 0


        self.acc_aux_loss = acc_aux_loss
        if self.acc_aux_loss:
            self.init_aux_statistics()

        if True:
            if gating_activation is None:
                gating_activation = nn.ReLU()
            self.f_gate = nn.Sequential(
                # nn.Linear(input_size, input_size),
                # gating_activation,
                nn.Linear(input_size,
                          2 * num_experts if noisy_gating else num_experts,
                          bias=False)
            )
            nn.init.zeros_(self.f_gate[-1].weight)
        else:
            self.f_gate = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(self.f_gate.weight)


    def extra_repr(self):
        return 'k={}, cvloss={}, switchloss={}, zloss={}, noisy_gating={}'.format(
            self.k, self.cvloss, self.switchloss, self.zloss, self.noisy_gating)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def init_aux_statistics(self):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        # self._gates = []
        # self._probs = []
        # self._logits = []
        # self._expert_sizes = []

    def update_aux_statistics(self, logits, probs, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def get_aux_loss_and_clear(self):
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        # cvloss = self.acc_gates.mean() / 10000.0
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        # loss = (self.cvloss * cvloss)
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss)

        # print("cvloss")
        # true_cvloss = self.compute_cvloss(torch.cat(self._gates, dim=0))
        # print(self.cvloss, cvloss, true_cvloss)

        # print("switchloss")
        # cat_probs = torch.cat(self._probs, dim=0)
        # true_switchloss = self.compute_switchloss(cat_probs, sum(self._expert_sizes))
        # print(self.switchloss, switchloss, true_switchloss)

        # print("zloss")
        # true_zloss = self.compute_zloss(torch.cat(self._logits, dim=0))
        # print(self.zloss, zloss, true_zloss)

        # assert torch.allclose(cvloss, true_cvloss)
        # assert torch.allclose(switchloss, true_switchloss)
        # assert torch.allclose(zloss, true_zloss)

        self.init_aux_statistics()
        return loss

    # def compute_topk_loss(self, probs):


    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def top_k_gating(self, x, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate(x)
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        assert sample_topk == 0
        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]


        top_k_gates = top_k_gates

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        # print('probs: ', probs)
        # print('top_k_gates: ', top_k_gates)
        # print('top_k_indices: ', top_k_indices)
        # print('expert_size: ', expert_size)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        if self.acc_aux_loss:
            # if self.training:
            self.update_aux_statistics(logits, probs, gates)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # y_ = self.forward_(x, skip_mask, sample_topk, multiply_by_gates)
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss

    def forward_(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # FOR DEBUGGING: naive forward
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate(x)
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        # if multiply_by_gates:
        #     expert_outputs = expert_outputs * self.batch_gates[:, None]
        return y

    def map(self, x, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y


class TaskMoE(MoE):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self,  input_size, head_size, num_experts, k, w_MI=0, w_H=0, w_finetune_MI=0, limit_k=0, w_topk_loss=0.0, task_num=9, noisy_gating=True, gating_activation=None, **kwargs):
        self.task_num = task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI
        self.w_H = w_H
        self.w_finetune_MI = w_finetune_MI
        self.limit_k = max(k, limit_k)
        self.gating_activation = gating_activation
        self.kwargs = kwargs

        super(TaskMoE, self).__init__(input_size, head_size, num_experts, k, noisy_gating=noisy_gating, gating_activation=gating_activation, **kwargs)
        
        if gating_activation is None:
            gating_activation = nn.GELU()


        if w_finetune_MI < -100: ## hack
            w_finetune_MI = 0
            self.w_finetune_MI = 0
            self.f_gate = nn.ModuleList([nn.Sequential(
                                                nn.Linear(input_size,
                                                      2 * num_experts if noisy_gating else num_experts,
                                                      bias=False)
                                        ) for i in range(task_num)])
        else:
            self.f_gate = nn.ModuleList([nn.Sequential(
                                            nn.Linear(input_size, input_size//4),
                                            gating_activation,
                                            nn.Linear(input_size//4,
                                                      2 * num_experts if noisy_gating else num_experts,
                                                      bias=True)
                                        ) for i in range(task_num)])

        for i in range(task_num):
            nn.init.zeros_(self.f_gate[i][-1].weight)

        self.register_buffer('PTE', torch.zeros(self.task_num, self.num_experts))
        self.register_buffer('PE', torch.zeros(self.num_experts))
        self.momentum = 0.0
        self.register_buffer('times',torch.zeros(1))
        # self.times = 0

        self.task_gate_freq = [0] * self.task_num
        self.topk_acc_probs = [0] * self.task_num
        self.token_probs = [0] * self.task_num

    def get_MIloss(self, logits, probs, gates, task_bh):

        if not self.training:
            return 0.0

        top_k_gates, _ = probs.topk(self.k, dim=1)
        self.token_probs[task_bh] = self.token_probs[task_bh] * 0.95 + top_k_gates.mean(0).detach()*0.05

        # a = ((gates > 0).float().sum(0)).detach()*0.05
        # b = self.task_gate_freq[task_bh]*0.95
        # print(a, b)
        self.task_gate_freq[task_bh] = self.task_gate_freq[task_bh]*0.95 + ((gates > 0).float().sum(0)).detach()*0.05

        self.topk_acc_probs[task_bh] = self.topk_acc_probs[task_bh]*0.95 + (probs.mean(0)).detach()*0.05
        
        PT = 1 / self.task_num # since we want each task to have equal weight

        # probs = P(E|T) in this batch
        # P(T,E) = P(E|T) * P(T) 
        self.PTE[task_bh] = self.PTE[task_bh] * self.momentum + (1-self.momentum) * (probs.mean(0).detach() * PT)


        # entropy loss
        # loss = 0.
        loss = -self.w_H * (probs * torch.log(probs + 0.0001)).sum(1).mean() # maximize the entropy

        # print('times: ', self.times[0])
        if self.times[0] < 100:
            self.times[0] = self.times[0] + 1
            self.momentum = 1 - 1/(self.times[0]) 
            return loss
        else:
            self.momentum = 0.99

        # P(E) = \sum_T (P(E,T))
        PE = self.PTE.sum(0).detach()

        # P(E,T) in this batch
        MI_task_gate = torch.zeros(self.task_num, self.num_experts).to(probs.device)
        MI_task_gate[task_bh] = MI_task_gate[task_bh] + probs.mean(0) * PT

        # P(E) in this batch
        P_EI = probs.mean(0) * PT

        # get the MI loss
        # MI_loss = -((MI_task_gate * (1 + torch.log(self.PTE.detach() + 0.0001)) ).sum() - P_EI * (1 + torch.log(PE + 0.0001))).sum()
        MI_loss = -((MI_task_gate * (1 + torch.log(self.PTE.detach() + 0.0001)) ).sum() - (P_EI * (1 + torch.log(PE + 0.0001))).sum())

        finetune_MI_loss = -((MI_task_gate * (1 + torch.log(self.PTE.detach() + 0.0001)) ).sum())

        loss = loss + self.w_MI * MI_loss + self.w_finetune_MI * finetune_MI_loss 
        
        return loss


    def get_topk_loss_and_clear(self):
        top_k_probs, top_k_indices = self.topk_acc_probs.topk(self.limit_k, dim=0)
        zeros = torch.zeros_like(self.topk_acc_probs)
        gates = zeros.scatter(0, top_k_indices, top_k_probs)
        topk_loss = ((self.topk_acc_probs - gates) * (self.topk_acc_probs - gates)).sum()

        self.topk_acc_probs = 0.
        return topk_loss * self.w_topk_loss # 0.004 * 12 * 2 = 0.09

    def top_k_gating(self, x, task_bh, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate[task_bh](x)
        # if self.noisy_gating and self.training:
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1) + 1e-4

        # # Convert the tensor to a NumPy array
        # probs_np = probs.detach().cpu().numpy()

        # # Specify the path to save the .npy file
        # path = '/project_data/ramanan/pengliaj/sparse-diffusion-policy-8experts/sparse-diffusion-policy-master-1/probs.npy'

        # # Save the array to a .npy file
        # np.save(path, probs_np)

        # print(f"Probabilities saved to {path}")
        
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]
       

        top_k_gates = top_k_gates

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        # print('here: ', expert_size)
        # print('probs: ', probs)
        # print('x: ', x)
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        return self.get_MIloss(logits, probs, gates, task_bh),probs

    def forward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # y_ = self.forward_(x, skip_mask, sample_topk, multiply_by_gates)
        # print('x: ', x)
        # x = x.permute(0, 2, 1)
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss,probs = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # y = y.view(bsz,self.input_size,length)
        # assert torch.allclose(y, y_)
        # print('y: ', y)
        return y, loss,probs

    def forward_(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # FOR DEBUGGING: naive forward
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate[task_bh](x)
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, task_bh, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        # if multiply_by_gates:
        #     expert_outputs = expert_outputs * self.batch_gates[:, None]
        return y


    def map(self, x, task_bh, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        # print('batch_index: ', batch_index)
        # print('expert_inputs: ', expert_inputs)
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y
if __name__ == '__main__':
  batch_size = 3
  sequence_length = 10
  input_size = 20
  model = TaskMoE(input_size=20,head_size=10,num_experts=5,k=2,activation=nn.Sequential(
                        nn.GELU(),
                    ),noisy_gating=False)

  num_param = sum(p.numel() for p in model.parameters())
  print('num_param: ', num_param)

  input_data = torch.randn(batch_size, sequence_length, input_size)

  # Specify the task or task batch you want to perform inference for.
  task_batch_index = int(0) # Replace with the appropriate task batch index.

  # You can skip certain tokens during inference by providing a skip_mask. 
  # Set to None if you don't want to skip any tokens.
  skip_mask = None

  # Perform inference (forward pass) using the TaskMoE model for the specified task.
  output, loss, prob = model(input_data, task_batch_index, skip_mask=skip_mask)
  print(output.shape)

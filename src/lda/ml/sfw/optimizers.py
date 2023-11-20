# ===========================================================================
# Project:      StochasticFrankWolfe 2020 / IOL Lab @ ZIB
# File:         pytorch/optimizers.py
# Description:  Pytorch implementation of Stochastic Frank Wolfe, AdaGradSFW and SGD with projection
# ===========================================================================
import torch
#from s_const import*
from math import sqrt
from torch import Tensor
from typing import List, Optional
import math
class SFW(torch.optim.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        learning_rate (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of learning_rate rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, learning_rate=0.1, rescale='diameter', momentum=0.9):
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if not (rescale in ['diameter', 'gradient', None]):
            raise ValueError("Rescale type must be either 'diameter', 'gradient' or None.")

        # Parameters
        self.rescale = rescale

        defaults = dict(lr=learning_rate, momentum=momentum)
        super(SFW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                # Add momentum
                momentum = group['momentum']
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state['momentum_buffer']
                        
                v = constraints[idx].lmo(d_p)  # LMO optimal solution
                if self.rescale == 'diameter':
                    # Rescale lr by diameter
                    factor = 1. / constraints[idx].get_diameter()
                elif self.rescale == 'gradient':
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)
                else:
                    # No rescaling
                    factor = 1

                lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]

                p.mul_(1 - lr)
                p.add_(v, alpha=lr)
                idx += 1
        return loss

class pos_SFW(torch.optim.Optimizer):
    """                                 
       Stochastic Frank Wolfe on v  == Tr(Lambda   Grad c )
                                  p             p     

        Where Lambda  is the p-th Gell'Mann matrix
                    p

        and c is the Kossakowski's matrix:

              1                                  i   ijk    
       c   =  _  (w   + w  ) - delta    w     +  _  e    b 
        ij    2    ij    ji          ij  kk      8        k 

                                       1
       Stochastic Gradient descent on  _ (w  - w  )
                                       2   ij   ji   
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        learning_rate (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of learning_rate rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, learning_rate=0.1, rescale='diameter', momentum=0.9):
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if not (rescale in ['diameter', 'gradient', None]):
            raise ValueError("Rescale type must be either 'diameter', 'gradient' or None.")

        # Parameters
        self.rescale = rescale

        defaults = dict(lr=learning_rate, momentum=momentum)
        super(pos_SFW, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
#            for p in group['params']:
                w = group['params'][0]
                b = group['params'][1]
                if w.grad is None and b.grad is None:
                    continue

                d_w = w.grad
                d_w_h = 0.5*(d_w - torch.transpose(d_w,0,1))
                d_b = b.grad
                epislon = LeviCivita()

                d_k_im = 1./8*torch.einsum('ijk,k->ij',epsilon,d_b)                
                d_k_re = 0.5*(d_w+torch.transpose(d_w,0,1))
                
                Lambda = get_basis(3)
                
                v_im = -torch.einsum('ijk,kj->i',torch.imag(Lambda) , d_k_im)*sqrt(3)
                v_re =  torch.einsum('ijk,kj->i',torch.real(Lambda) , d_k_re)*sqrt(3)
                
                d_p = v_im + v_re
                # Add momentum
                #momentum = group['momentum']
                #if momentum > 0:
                #    w_state = self.state[w]
                #    b_state = self.state[b]
                #    if 'momentum_buffer' not in w_state or 'momentum_buffer' not in b_state:
                #        param_state['momentum_buffer'] = d_p.detach().clone()
                #    else:
                #        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                #        d_p = param_state['momentum_buffer']
                
                
                v = constraints[idx].lmo(d_p)  # LMO optimal solution
                if self.rescale == 'diameter':
                    # Rescale lr by diameter
                    factor = 1. / constraints[idx].get_diameter()
                elif self.rescale == 'gradient':
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)
                else:
                    # No rescaling
                    factor = 1
                d_k_re = 1./sqrt(3)*torch.einsum('i,ijk->jk',v,torch.real(Lambda) )
                d_k_im = 1./sqrt(3)*torch.einsum('i,ijk->jk',v,torch.imag(Lambda) )
                d_b =  8*torch.einsum('ijk,jk->i',epsilon, d_k_im)
                d_w_d =  8*torch.einsum('ijk,jk->i',epsilon, d_k_re)
                lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]
                w.add_(d_w_h, alpha=-group['lr'])
                w_1 = w.mul(-1-0.5*lr)
                w_2 = torch.transpose(w,0,1)*( 1-0.5*lr)
                w.add_(w_1)
                w.add_(w_2)
                w_sim = 0.5*(torch.transpose(w,0,1) + w)*(1 - lr)
                b.mul_(1-lr)
                b.add_(d_b, alpha = lr)
                idx += 1
        return loss



class AdaGradSFW(torch.optim.Optimizer):
    """AdaGrad Stochastic Frank-Wolfe algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        inner_steps (integer, optional): number of inner iterations (default: 2)
        learning_rate (float, optional): learning rate (default: 1e-2)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        momentum (float, optional): momentum factor
    """

    def __init__(self, params, inner_steps=2, learning_rate=1e-2, delta=1e-8, momentum=0.9):
        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Momentum must be between [0, 1].")
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not int(inner_steps) == inner_steps and not 0.0 <= inner_steps:
            raise ValueError("Number of inner iterations needs to be a positive integer: {}".format(inner_steps))

        self.K = inner_steps

        defaults = dict(lr=learning_rate, delta=delta, momentum=momentum)
        super(AdaGradSFW, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                param_state = self.state[p]
                momentum = group['momentum']

                if momentum > 0:
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state['momentum_buffer']

                param_state['sum'].addcmul_(d_p, d_p, value=1)  # Holds the cumulative sum
                H = torch.sqrt(param_state['sum']).add(group['delta'])

                y = p.detach().clone()
                for _ in range(self.K):
                    d_Q = d_p.addcmul(H, y - p, value=1. / group['lr'])
                    y_v_diff = y - constraints[idx].lmo(d_Q)
                    gamma = group['lr'] * torch.div(torch.sum(torch.mul(d_Q, y_v_diff)),
                                                    torch.sum(H.mul(torch.mul(y_v_diff, y_v_diff))))
                    gamma = max(0.0, min(gamma, 1.0))  # Clamp between [0, 1]

                    y.add_(y_v_diff, alpha=-gamma)  # -gamma needed as we want to add v-y, not y-v
                p.copy_(y)
                idx += 1
        return loss


class SGD(torch.optim.Optimizer):
    """Modified SGD which allows projection via Constraint class"""

    def __init__(self, params, lr=0.5, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if momentum is None:
            momentum = 0
        if weight_decay is None:
            weight_decay = 0
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            idx = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

                # Project if necessary
                if not constraints[idx].is_unconstrained():
                    p.copy_(constraints[idx].euclidean_project(p))
                idx += 1

        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)







class Adam(torch.optim.Optimizer ):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, constraints,closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
                    


            adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])

                    # Project if necessary
        for group in self.param_groups:
            idx = 0
            for p in group['params']:
                    if not constraints[idx].is_unconstrained():
                            p.copy_(constraints[idx].euclidean_project(p))
                    idx += 1




        return loss


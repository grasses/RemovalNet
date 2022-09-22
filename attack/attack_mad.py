#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import os.path as osp
import os
import copy
import logging
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from exp import vis as exp_F
from utils import helper, metric
sys_args = helper.get_args()
args = helper.get_args()
filename = str(osp.basename(__file__)).split(".")[0]
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-8s \033[1;35m %(message)s \033[0m")


class MAD:
    def __init__(self, bench, model1, model2, epsilon=0.5, optim='linesearch', max_grad_layer=None, ydist='l1',
                 oracle='extreme', disable_jacobian=False, objmax=True, out_path=None, batch_size=100, **kwargs):

        print('=> MAD ({})'.format([epsilon, optim, ydist, oracle]))

        self.alpha = 0.7
        self.T = 20
        self.call_count = 0
        self.device = sys_args.device
        self.logger = logging.getLogger('RemovealNet')

        self.model1 = model1
        self.model2 = model2
        self.source_name = str(model1)
        self.attack_name = str(model2)
        self.source_model = model1.torch_model.to(self.device)
        self.origin_model = model2.torch_model.to(self.device)
        self.attack_model = copy.deepcopy(model2.torch_model).to(self.device)

        self.train_loader = bench.get_dataloader(model2.dataset_id, split='train', batch_size=batch_size, shuffle=True)
        self.test_loader = bench.get_dataloader(model2.dataset_id, split='test', batch_size=batch_size, shuffle=True)

        self.scope_name = f"{model2}stealthnet({self.alpha},{self.T})-"
        self.torch_model_path = osp.join(bench.models_dir, self.scope_name)

        self.epsilon = epsilon
        self.out_path = out_path
        self.disable_jacobian = bool(disable_jacobian)
        if self.disable_jacobian:
            print('')
            print('!!!WARNING!!! Using G = eye(K)')
            print('')

        self.objmax = bool(objmax)
        self.K = None
        self.D = None

        self.ydist = ydist
        assert ydist in ['l1', 'l2', 'kl']

        # Which oracle to use
        self.oracle = oracle
        assert self.oracle in ['extreme', 'random', 'argmin', 'argmax']

        # Which algorithm to use to optimize
        self.optim = optim
        assert optim in ['linesearch', 'projections', 'greedy']

        # Gradients from which layer to use?
        assert max_grad_layer in [None, 'all']
        self.max_grad_layer = max_grad_layer

        # Track some data for debugging
        self.queries = []  # List of (x_i, y_i, y_i_prime, distance)
        self.jacobian_times = []


    @staticmethod
    def compute_jacobian_nll(x, model_adv_proxy, device=sys_args.device, K=None, max_grad_layer=None):
        assert x.shape[0] == 1, 'Does not support batching'
        x = x.to(device)

        # Determine K
        if K is None:
            with torch.no_grad():
                z_a = model_adv_proxy(x)
            _, K = z_a.shape

        # ---------- Precompute G (k x d matrix): where each row represents gradients w.r.t NLL at y_gt = k
        G = []
        z_a = model_adv_proxy(x)

        nlls = -F.log_softmax(z_a, dim=1).mean(dim=0)  # NLL over K classes, Mean over rows
        #for k, v in model_adv_proxy.named_parameters():
        #    print(k)

        assert len(nlls) == K
        for k in range(K):
            nll_k = nlls[k]

            _params = [p for p in model_adv_proxy.parameters()]
            w_idx = -4  # Default to FC layer
            # Manually compute gradient only on the required parameters prevents backprop-ing through entire network
            # This is significantly quicker
            # Verified and compared the below with nll_k.backward(retain_graph=True)
            grads, *_ = torch.autograd.grad(nll_k, _params[w_idx], retain_graph=True)
            G.append(grads.flatten().clone())
        G = torch.stack(G).to(device)

        return G

    @staticmethod
    def calc_objective(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            u = u / u.norm()
            v = torch.matmul(G.t(), y)
            v = v / v.norm()
            objval = (u - v).norm() ** 2
        return objval

    @staticmethod
    def calc_objective_batched(ytilde, y, G):
        K, D = G.shape
        _K, B = ytilde.shape
        assert ytilde.size() == y.size() == torch.Size([K, B]), 'Failed: {} == {} == {}'.format(ytilde.size(), y.size(),
                                                                                                torch.Size([K, B]))

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            u = u / u.norm(dim=0)

            v = torch.matmul(G.t(), y)
            v = v / v.norm(dim=0)

            objvals = (u - v).norm(dim=0) ** 2

        return objvals

    @staticmethod
    def calc_objective_numpy(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = G.T @ ytilde
        u /= np.linalg.norm(u)

        v = G.T @ y
        v /= np.linalg.norm(v)

        objval = np.linalg.norm(u - v) ** 2

        return objval

    @staticmethod
    def calc_objective_numpy_batched(ytilde, y, G):
        K, D = G.shape
        _K, N = ytilde.shape
        assert ytilde.shape == y.shape == torch.Size([K, N]), 'Does not support batching'

        u = np.matmul(G.T, ytilde)
        u /= np.linalg.norm(u, axis=0)

        v = np.matmul(G.T, y)
        v /= np.linalg.norm(v, axis=0)

        objvals = np.linalg.norm(u - v, axis=0) ** 2

        return objvals

    @staticmethod
    def calc_surrogate_objective(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'ytilde = {}\ty = {}'.format(ytilde.shape, y.shape)

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            v = torch.matmul(G.t(), y)

            objval = (u - v).norm() ** 2

        return objval

    @staticmethod
    def calc_surrogate_objective_batched(ytilde, y, G):
        K, D = G.shape
        _K, B = ytilde.shape
        assert ytilde.size() == y.size() == torch.Size([K, B]), 'Failed: {} == {} == {}'.format(ytilde.size(), y.size(),
                                                                                                torch.Size([K, B]))

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            v = torch.matmul(G.t(), y)
            objvals = (u - v).norm(dim=0) ** 2

        return objvals

    @staticmethod
    def calc_surrogate_objective_numpy(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = G.T @ ytilde
        v = G.T @ y

        objval = np.linalg.norm(u - v) ** 2

        return objval

    @staticmethod
    def calc_surrogate_objective_numpy_batched(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = np.matmul(G.T, ytilde)
        v = np.matmul(G.T, y)
        objvals = np.linalg.norm(u - v, axis=0) ** 2

        return objvals

    @staticmethod
    def oracle_extreme(G, y, max_over_obj=False):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        argmax_k = -1
        argmax_val = -1.

        for k in range(K):
            yk = torch.zeros_like(y)
            yk[k] = 1.

            if max_over_obj:
                kval = MAD.calc_objective(yk, y, G)
            else:
                kval = MAD.calc_surrogate_objective(yk, y, G)

            if kval > argmax_val:
                argmax_val = kval
                argmax_k = k

        ystar = torch.zeros_like(y)
        ystar[argmax_k] = 1.

        return ystar, argmax_val

    @staticmethod
    def oracle_argmax_preserving(G, y, max_over_obj=False):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        if K > 10:
            # return MAD.oracle_argmax_preserving_approx(G, y, max_over_obj)
            return MAD.oracle_argmax_preserving_approx_gpu(G, y, max_over_obj)

        max_k = y.argmax()
        G_np = G.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        ystar = None
        max_val = -1.

        # Generate a set of 1-hot encoded vertices
        # This command produces vertex sets e.g., for K=3: 000, 001, 010, 011, ..., 111
        # Idea is to distribute prob. mass equally over vertices set to 1
        vertices = np.asarray(list(itertools.product([0, 1], repeat=K)), dtype=np.float32)
        # Select vertices where k-th vertex = 1
        vertices = vertices[vertices[:, max_k] > 0]
        # Iterate over these vertices to find argmax k
        for y_extreme in vertices:
            # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
            # Upweigh k-th label by epsilon to maintain argmax label
            y_extreme[max_k] += 1e-5
            # Convert to prob vector
            y_extreme = y_extreme / y_extreme.sum()

            # Doing this on CPU is much faster (I guess this is because we don't need a mem transfer each iteration)
            if max_over_obj:
                kval = MAD.calc_objective_numpy(y_extreme, y_np, G_np)
            else:
                kval = MAD.calc_surrogate_objective_numpy(y_extreme, y_np, G_np)

            if kval > max_val:
                max_val = kval
                ystar = y_extreme

        ystar = torch.tensor(ystar).to(G.device)
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_argmax_preserving_approx_gpu(G, y, max_over_obj=False, max_iters=1024):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        max_k = y.argmax().item()
        G_np = G.detach()
        y_np = y.detach().clone()

        # To prevent underflow
        y_np += 1e-8
        y_np /= y_np.sum()

        ystar = None
        max_val = -1.
        niters = 0.

        # By default, we have (K-1)! vertices -- this does not scale when K is large (e.g., CIFAR100).
        # So, perform the search heuristically.
        # Search strategy used:
        #   a. find k which maximizes objective
        #   b. fix k
        #   c. repeat
        fixed_verts = [max_k, ]  # Grow this set

        while niters < max_iters:
            y_prev_extreme = torch.zeros(K)
            y_prev_extreme[fixed_verts] = 1.

            # Find the next vertex extreme
            k_list = np.array(sorted((set(range(K)) - set(fixed_verts))), dtype=int)
            y_extreme_batch = []
            for i, k in enumerate(k_list):
                y_extreme = y_prev_extreme.clone().detach()
                # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
                y_extreme[k] = 1.

                # Upweigh k-th label by epsilon to maintain argmax label
                y_extreme[max_k] += 1e-5
                # Convert to prob vector
                y_extreme = y_extreme / y_extreme.sum()

                y_extreme_batch.append(y_extreme)

            y_extreme_batch = torch.stack(y_extreme_batch).transpose(0, 1).to(G_np.device)
            assert y_extreme_batch.size() == torch.Size([K, len(k_list)]), '{} != {}'.format(y_extreme_batch.size(),
                                                                                             (K, len(k_list)))
            B = y_extreme_batch.size(1)

            y_np_batch = torch.stack([y_np.clone().detach() for i in range(B)]).transpose(0, 1)

            kvals = MAD.calc_objective_batched(y_extreme_batch, y_np_batch, G_np)

            max_i = kvals.argmax().item()
            max_k_val = kvals.max().item()

            if max_k_val > max_val:
                max_val = max_k_val
                ystar = y_extreme_batch[:, max_i]

            next_k = k_list[max_i]
            fixed_verts.append(next_k)

            niters += B

        try:
            ystar = ystar.clone().detach()
        except AttributeError:
            import ipdb;
            ipdb.set_trace()
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_argmax_preserving_approx(G, y, max_over_obj=False, max_iters=1024):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        max_k = y.argmax()
        G_np = G.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        ystar = None
        max_val = -1.
        niters = 0.

        # By default, we have (K-1)! vertices -- this does not scale when K is large (e.g., CIFAR100).
        # So, perform the search heuristically.
        # Search strategy used:
        #   a. find k which maximizes objective
        #   b. fix k
        #   c. repeat
        fixed_verts = np.array([max_k, ], dtype=int)  # Grow this set
        while niters < max_iters:
            y_prev_extreme = np.zeros((K,), dtype=np.float32)
            y_prev_extreme[fixed_verts] = 1.

            # Find the next vertex extreme
            k_list = np.array(sorted((set(range(K)) - set(fixed_verts))), dtype=int)
            y_extreme_batch = []
            for i, k in enumerate(k_list):
                y_extreme = y_prev_extreme.copy()
                # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
                y_extreme[k] = 1.

                # Upweigh k-th label by epsilon to maintain argmax label
                y_extreme[max_k] += 1e-5
                # Convert to prob vector
                y_extreme = y_extreme / y_extreme.sum()

                y_extreme_batch.append(y_extreme)

            y_extreme_batch = np.array(y_extreme_batch).T
            assert y_extreme_batch.shape == (K, len(k_list)), '{} != {}'.format(y_extreme_batch.shape, (K, len(k_list)))
            B = y_extreme_batch.shape[1]

            y_np_batch = np.stack([y_np.copy() for i in range(B)]).T.astype(np.float32)

            kvals = MAD.calc_objective_numpy_batched(y_extreme_batch, y_np_batch, G_np)

            max_i = np.argmax(kvals)
            max_k_val = np.max(kvals)

            if max_k_val > max_val:
                max_val = max_k_val
                ystar = y_extreme_batch[:, max_i]

            next_k = k_list[max_i]
            fixed_verts = np.concatenate((fixed_verts, [next_k, ]), axis=0)

            niters += B

        ystar = torch.tensor(ystar).to(G.device)
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_rand(G, y):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        rand_k = np.random.randint(low=0, high=K)

        ystar = torch.zeros_like(y)
        ystar[rand_k] = 1.
        return ystar, torch.tensor(-1.)

    @staticmethod
    def oracle_argmin(G, y):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        argmin_k = y.argmin().item()
        ystar = torch.zeros_like(y)

        ystar[argmin_k] = 1.
        return ystar, torch.tensor(-1.)

    @staticmethod
    def calc_distance(y, ytilde, ydist, device=torch.device('cuda')):
        assert y.shape == ytilde.shape, 'y = {}, ytile = {}'.format(y.shape, ytilde.shape)
        assert len(y.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']

        ytilde = ytilde.to(device)
        if ydist == 'l1':
            return (ytilde - y).norm(p=1)
        elif ydist == 'l2':
            return (ytilde - y).norm(p=2)
        elif ydist == 'kl':
            return F.kl_div(ytilde.log(), y, reduction='sum')
        else:
            raise ValueError('Unrecognized ydist contraint')

    @staticmethod
    def project_ydist_constraint(delta, epsilon, ydist, y=None):
        assert len(delta.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']

        device = delta.device
        K, = delta.shape

        assert delta.shape == torch.Size([K, ])
        if ydist == 'l1':
            delta_numpy = delta.detach().cpu().numpy()
            delta_projected = euclidean_proj_l1ball(delta_numpy, s=epsilon)
            delta_projected = torch.tensor(delta_projected)
        elif ydist == 'l2':
            delta_projected = epsilon * delta / delta.norm(p=2).clamp(min=epsilon)
        elif ydist == 'kl':
            raise NotImplementedError()
        delta_projected = delta_projected.to(device)
        return delta_projected

    @staticmethod
    def project_simplex_constraint(ytilde):
        assert len(ytilde.shape) == 1, 'Does not support batching'
        K, = ytilde.shape
        device = ytilde.device

        ytilde_numpy = ytilde.detach().cpu().numpy()
        ytilde_projected = euclidean_proj_simplex(ytilde_numpy)
        ytilde_projected = torch.tensor(ytilde_projected)
        ytilde_projected = ytilde_projected.to(device)
        return ytilde_projected

    @staticmethod
    def closed_form_alpha_estimate(y, ystar, ydist, epsilon):
        assert y.shape == ystar.shape, 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert len(y.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']
        K, = y.shape

        if ydist == 'l1':
            p = 1.
        elif ydist == 'l2':
            p = 2.
        else:
            raise ValueError('Only supported for l1/l2')
        alpha = epsilon / ((y - ystar).norm(p=p) + 1e-7)
        alpha = alpha.clamp(min=0., max=1.)
        return alpha

    @staticmethod
    def linesearch(G, y, ystar, ydist, epsilon, closed_alpha=True):
        """
        Let h(\alpha) = (1 - \alpha) y + \alpha y*
        Compute \alpha* = argmax_{\alpha} h(\alpha)
        s.t.  dist(y, h(\alpha)) <= \epsilon

        :param G:
        :param y:
        :param ystar:
        :return:
        """
        K, D = G.shape
        assert y.shape == ystar.shape == torch.Size([K, ]), 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert ydist in ['l1', 'l2', 'kl']

        # Definition of h
        h = lambda alpha: (1 - alpha) * y + alpha * ystar

        # Short hand for distance function
        dist_func = lambda y1, y2: MAD.calc_distance(y1, y2, ydist)

        if ydist in ['l1', 'l2'] and closed_alpha:
            # ---------- Optimally compute alpha
            alpha = MAD.closed_form_alpha_estimate(y, ystar, ydist, epsilon)
            ytilde = h(alpha)
        else:
            # ---------- Bisection method
            alpha_low, alpha_high = 0., 1.
            h_low, h_high = h(alpha_low), h(alpha_high)

            # Sanity check
            feasible_low = dist_func(y, h_low) <= epsilon
            feasible_high = dist_func(y, h_high) <= epsilon
            assert feasible_low or feasible_high

            if feasible_high:
                # Already feasible. Our work here is done.
                ytilde = h_high
                delta = ytilde - y
                return delta
            else:
                ytilde = h_low

            # Binary Search
            for i in range(15):
                alpha_mid = (alpha_low + alpha_high) / 2.
                h_mid = h(alpha_mid)
                feasible_mid = dist_func(y, h_mid) <= epsilon

                if feasible_mid:
                    alpha_low = alpha_mid
                    ytilde = h_mid
                else:
                    alpha_high = alpha_mid

        delta = ytilde - y
        return delta

    @staticmethod
    def greedy(G, y, ystar):
        NotImplementedError()

    @staticmethod
    def is_in_dist_ball(y, ytilde, ydist, epsilon, tolerance=1e-4):
        assert y.shape == ytilde.shape, 'y = {}, ytile = {}'.format(y.shape, ytilde.shape)
        assert len(y.shape) == 1, 'Does not support batching'
        return (MAD.calc_distance(y, ytilde, ydist) - epsilon).clamp(min=0.) <= tolerance

    @staticmethod
    def is_in_simplex(ytilde, tolerance=1e-4):
        assert len(ytilde.shape) == 1, 'Does not support batching'
        return torch.abs(ytilde.clamp(min=0., max=1.).sum() - 1.) <= tolerance

    @staticmethod
    def projections(G, y, ystar, epsilon, ydist, max_iters=100):
        K, D = G.shape
        assert y.shape == ystar.shape == torch.Size([K, ]), 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert ydist in ['l1', 'l2', 'kl']

        ytilde = ystar
        device = G.device

        for i in range(max_iters):
            # (1) Enforce distance constraint
            delta = ytilde - y
            delta = MAD.project_ydist_constraint(delta, epsilon, ydist).to(device)
            ytilde = y + delta

            # (2) Project back into simplex
            ytilde = MAD.project_simplex_constraint(ytilde).to(device)

            # Break out if constraints are met
            if MAD.is_in_dist_ball(y, ytilde, ydist, epsilon) and MAD.is_in_simplex(ytilde):
                break

        delta = ytilde - y
        return delta

    def calc_delta(self, x, y, debug=False):
        # Jacobians G
        if self.disable_jacobian or self.oracle in ['random', 'argmin']:
            G = torch.eye(self.K).to(self.device)
        else:
            # _start = time.time()
            G = MAD.compute_jacobian_nll(x, self.origin_model, device=self.device, K=self.K)
            # _end = time.time()
            # self.jacobian_times.append(_end - _start)
            # if np.random.random() < 0.05:
            #     print('mean = {:.6f}\tstd = {:.6f}'.format(np.mean(self.jacobian_times), np.std(self.jacobian_times)))
            # # print(_end - _start)
        if self.D is None:
            self.D = G.shape[1]

        # y* via oracle
        if self.oracle == 'random':
            ystar, ystar_val = self.oracle_rand(G, y)
        elif self.oracle == 'extreme':
            ystar, ystar_val = self.oracle_extreme(G, y, max_over_obj=self.objmax)
        elif self.oracle == 'argmin':
            ystar, ystar_val = self.oracle_argmin(G, y)
        elif self.oracle == 'argmax':
            ystar, ystar_val = self.oracle_argmax_preserving(G, y, max_over_obj=self.objmax)
        else:
            raise ValueError()

        # y* maybe outside the feasible set - project it back
        if self.optim == 'linesearch':
            delta = self.linesearch(G, y, ystar, self.ydist, self.epsilon)
        elif self.optim == 'projections':
            delta = self.projections(G, y, ystar, self.ydist, self.epsilon)
        elif self.optim == 'greedy':
            raise NotImplementedError()
        else:
            raise ValueError()

        # Calc. final objective values
        ytilde = y + delta
        objval = self.calc_objective(ytilde, y, G)
        objval_surrogate = self.calc_surrogate_objective(ytilde, y, G)

        print("-> ytilde", ytilde)
        return delta, objval, objval_surrogate

    @staticmethod
    def calc_query_distances(queries):
        l1s, l2s, kls = [], [], []
        for i in range(len(queries)):
            y_v, y_prime, *_ = queries[i]
            y_v, y_prime = torch.tensor(y_v), torch.tensor(y_prime)
            l1s.append((y_v - y_prime).norm(p=1).item())
            l2s.append((y_v - y_prime).norm(p=2).item())
            kls.append(F.kl_div(y_prime.log(), y_v, reduction='sum').item())
        l1_mean, l1_std = np.mean(l1s), np.std(l1s)
        l2_mean, l2_std = np.mean(l2s), np.std(l2s)
        kl_mean, kl_std = np.mean(kls), np.std(kls)
        return l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std

    @staticmethod
    def eval(model, test_loader, device=sys_args.device, epoch=0, debug=True):
        test_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = F.cross_entropy(output, y)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= (1.0 * len(test_loader.dataset))
        acc = 100.0 * correct / len(test_loader.dataset)
        msg = "-> For E{:d}, [Test] loss={:.5f}, acc={:.3f}%".format(
            int(epoch),
            test_loss,
            acc
        )
        if debug:
            print(msg)
        return acc, test_loss

    def save_torch_model(self, torch_model):
        if not osp.exists(self.torch_model_path):
            os.makedirs(self.torch_model_path)
        ckpt_path = osp.join(self.torch_model_path, 'final_ckpt.pth')
        torch.save(
            {'state_dict': torch_model.cpu().state_dict()},
            ckpt_path,
        )
        torch_model.to(self.device)
        self.logger.info(f"-> save model to: {self.torch_model_path}")

    def loss_kd(self, preds, labels, teacher_preds):
        alpha = self.alpha
        T = self.T
        loss = alpha * T * T * F.kl_div(
            F.log_softmax(preds / T, dim=1),
            F.softmax(teacher_preds / T, dim=1), reduction='batchmean') + \
               (1. - alpha) * F.cross_entropy(preds, labels)
        return loss

    def latent_space_poison(self, proxy_model, x, lr=0.1, alpha=0.5, steps=40):
        proxy_model.eval()
        ben_feats = proxy_model.mid_forward(x, layer_index=5).detach()
        adv_feats = ben_feats.clone().detach()
        adv_feats += torch.empty_like(adv_feats).uniform_(-0.5, 0.5)
        y = proxy_model(x).argmax(dim=1).detach().clone()
        for step in range(steps):
            adv_feats.requires_grad = True
            y_hat = proxy_model.bak_forward(adv_feats, layer_index=5)
            loss = alpha * F.cross_entropy(y_hat, y) + \
                   (1-alpha) * F.cosine_similarity(adv_feats.view(1, -1), ben_feats.view(1, -1)) - \
                   (1-alpha) * 0.1 * torch.abs(adv_feats).mean()

            grad = torch.autograd.grad(loss, adv_feats, retain_graph=False, create_graph=False)[0]
            adv_feats = adv_feats.detach() - lr * grad.sign()
            #print(f"-> step:{step} y:{y.item()} y_hat:{y_hat.argmax(dim=1).item()} loss:{loss.item()} grad:{torch.norm(grad.view(-1))}, adv_feats:{torch.norm(adv_feats.view(-1))}")
        return adv_feats.detach()


    def perturbate(self, origin_model, x):
        model = copy.deepcopy(origin_model).to(self.device)
        with torch.no_grad():
            x = x.to(self.device)
            z_v = model(x)  # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1).detach()
            self.call_count += x.shape[0]
        y_prime = []
        feats_prime = []

        # No batch support yet. So, perturb individually.
        for i in range(x.shape[0]):
            x_i = x[i].unsqueeze(0)
            y_v_i = y_v[i]

            with torch.enable_grad():
                latent_feats_i = self.latent_space_poison(model, x_i)
                if self.epsilon > 0.:
                    delta_i, objval, sobjval = self.calc_delta(x_i, y_v_i)
                else:
                    delta_i = torch.zeros_like(y_v_i)
                    objval, sobjval = torch.tensor(0.), torch.tensor(0.)
            y_prime_i = y_v_i + delta_i
            y_prime.append(y_prime_i)
            feats_prime.append(latent_feats_i)
        y_prime = torch.stack(y_prime).detach()
        feats_prime = torch.cat(feats_prime).detach()
        return feats_prime, y_prime

    def fingerprint_unlearning2(self, epochs=10, alpha=0.2):
        self.attack_model.train()
        self.origin_model.eval()
        optimizer = torch.optim.SGD(
            self.attack_model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=5e-3,
        )

        for epoch in range(epochs):
            running_loss = 0.0
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                feats_prime, y_prime = self.perturbate(origin_model=self.origin_model, x=x)

                self.attack_model.train()
                self.origin_model.eval()
                optimizer.zero_grad()
                out = self.attack_model(x)
                latent_feats = self.attack_model.mid_forward(x, layer_index=5)

                kd_loss = self.loss_kd(out, y, y_prime)
                feats_loss = ((latent_feats - feats_prime) ** 2).mean()
                loss = (1-alpha) * kd_loss + alpha * feats_loss

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                cos_sim = torch.nn.CosineSimilarity(dim=1)(out, y_prime).sum() / y.size(0)
                self.logger.info(f"-> E{epoch} [{step}/{len(self.train_loader)}] kd_loss:{kd_loss.item()} feats_loss:{feats_loss.item()} cos_sim:{cos_sim}")
                if step % 8 == 0:
                    metric.topk_test(self.attack_model, self.test_loader, epoch=epoch, debug=True, device=self.device)
                    print()
            self.save_torch_model(self.attack_model)
            exp_F.plot_logist_embedding(self.attack_model, self.origin_model, self.test_loader, file_name=f"{self.scope_name}_e{epoch+1}.pdf")
            print()
            print()

    def fingerprint_unlearning(self, epochs=10, layer_index=4):
        self.attack_model.train()
        self.origin_model.eval()
        optimizer = torch.optim.SGD(
            self.attack_model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=5e-3,
        )
        self.eval(self.attack_model, self.test_loader, epoch=0)

        for epoch in range(epochs):
            running_loss = 0.0
            self.attack_model.train()
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                # 修改中间层神经元输出
                feats1 = self.origin_model.mid_forward(x, layer_index=layer_index).detach()
                feats2 = self.attack_model.mid_forward(x, layer_index=layer_index).detach().clone()

                self.attack_model.eval()
                for _ in range(20):
                    print(feats2.shape)
                    feats2.requires_grad = True
                    outputs = self.attack_model.bak_forward(feats2, layer_index=5)

                    cosine_loss = torch.nn.CosineSimilarity()(feats2, feats1)
                    ce_loss = torch.nn.CrossEntropyLoss()(outputs, y)

                    loss = cosine_loss + ce_loss
                    grad = torch.autograd.grad(loss, feats2, retain_graph=False, create_graph=False)[0]
                    feats2 = feats2.detach() + 0.1 * grad.sign()

                    print(feats2)
                    print(f"-> ce_loss:{ce_loss} cosine_loss:{cosine_loss}")
                    print()


            self.save_torch_model(self.attack_model)
            exp_F.plot_logist_embedding(self.attack_model, self.origin_model, self.test_loader,
                                        file_name=f"feats_poison_{self.scope_name}_e{epoch + 1}.pdf")
            print()
            print()



def main():
    """
    Example command:

    SCRIPT="attack.attack_mad"
    <===========================  Flower102-mbnetv2  ===========================>
    python -m "attack.attack_mad" -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-steal(resnet18)-" -device 3
    python -m "attack.attack_mad" -model1 "pretrain(mbnetv2,ImageNet)-" -model2 "pretrain(mbnetv2,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-" -device 2


    <===========================  Flower102-resnet18  ===========================>
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-" -device 2
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-" -device 2
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-" -device 2



    d python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-" -device 3
    r python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.5)-" -device 3
    r python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.8)-" -device 3
    d python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-distill()-" -device 3
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-steal(resnet18)-" -device 3
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-steal(mbnetv2)-" -device 3


    r python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.2)-" -device 2
    r python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.5)-" -device 2
    r python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-prune(0.8)-" -device 2
    d python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-distill()-" -device 2
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-steal(resnet18)-" -device 2
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.5)-steal(mbnetv2)-" -device 2


    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.2)-" -device 3
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.5)-" -device 3
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-prune(0.8)-" -device 3
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-distill()-" -device 3
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-steal(resnet18)-" -device 3
    python -m $SCRIPT -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,1)-steal(mbnetv2)-" -device 3





    python -m "attack.attack_mad" -model1 "pretrain(resnet18,ImageNet)-" -model2 "pretrain(resnet18,ImageNet)-transfer(Flower102,0.1)-prune(0.2)-" -device 0



    python -m "attack.attack_mad" -model1 "pretrain(mbnetv2,ImageNet)-" -model2 "pretrain(mbnetv2,ImageNet)-transfer(Flower102,1)-" -device 3



    """

    from benchmark import ImageBenchmark
    args = helper.get_args()
    benchmark = ImageBenchmark(
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir
    )
    model1 = benchmark.get_model_wrapper(args.model1)
    model2 = benchmark.get_model_wrapper(args.model2)
    net = MAD(benchmark, model1=model1, model2=model2)
    net.fingerprint_unlearning2()


if __name__ == "__main__":
    main()
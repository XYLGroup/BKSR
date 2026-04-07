"""
This code started out as a PyTorch port of the following:
https://github.com/HJ-harry/MCG_diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
The conditions are changed and coefficient matrix estimation is added.
"""

import enum
import numpy as np
import torch as th
from torch.autograd import grad
import torch.nn.functional as nF
import torch.nn.parameter as Para
from .utils import *
from os.path import join as join
from utility import *
from utility.utils import *
import torch.distributions as dist


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

class Param(th.nn.Module):
    def __init__(self, data):
        super(Param, self).__init__()
        self.E = Para.Parameter(data=data)
    
    def forward(self,):
        return self.E

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe10216a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    """

    def __init__(
        self,
        *,
        betas
    ):

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., self.alphas_cumprod))



    def p_sample_loop(
        self,
        model,
        shape,
        Rr,
        step=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_condition=None,
        param=None,
        save_root=None,
        progress=True
    ):
        finalX = None
        finalE = None

        for (sample, E) in self.p_sample_loop_progressive(
            model,
            shape,
            Rr,
            step=step,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_condition=model_condition,
            param=param,
            save_root=save_root
        ):
            finalX = sample
            finalE = E
             
        return finalX["sample"], finalE
        # return finalX["pred_xstart"], finalE

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        Rr,
        step=None,
        noise=None,
        clip_denoised=True,
        model_condition=None,
        device=None,
        param=None,
        denoised_fn=None,
        save_root=None   # use it for output intermediate predictions
        ):
        """
        Generator that runs the progressive sampling loop and yields intermediate
        outputs and estimated mapping matrices.

        Args:
            model: the denoising model callable
            shape: tuple (B, C, H, W) of the expected output shape
            Rr: reduced spectral dimensionality used by the sampler
            step: number of sampling steps (if None uses self.num_timesteps)
            noise: optional initial noise tensor
            clip_denoised: whether to clamp predicted x_start
            model_condition: dict with condition tensors (e.g., 'input', 'k_gt')
            device: torch device for sampling
            param: additional algorithm parameters dict
            denoised_fn: optional hook for denoised output
            save_root: optional path root for saving intermediate outputs

        Yields:
            (out, E) pairs where out is a dict with 'sample' and 'pred_xstart',
            and E is the current mapping matrix.
        """

        Bb, Cc, Hh, Ww = shape
        Rr = Rr
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn((Bb, Rr, Hh, Ww), device=device) # initial size is [B, r, H, W]

        if step is None:
            step = self.num_timesteps
        indices = list(np.arange(0, self.num_timesteps, self.num_timesteps//step))[::-1]
        indices_next = indices[1:] + [-1]
        from tqdm import tqdm
        pbar = tqdm(enumerate(zip(indices, indices_next)), total=len(indices))

        self.Cc = Cc

        self.best_result, self.best_psnr = None, 0
        norm_list, psnr_list, result_list = [], [], []
        # self.kernel_motion_list = []
        # TKE_psnr_list = []
        alphas_bar_list = []

        # randomly select 3 initial bands
        self.select_C = torch.randint(0, self.Cc, (3,))

        # TKE initialization
        n_k = 201
        self.k_s = model_condition['k_s']
        # store kernel type chosen in main.py for sampling strategy
        self.kernel_type = model_condition.get('kernel_type', 'Guassian')
        if param['task'] == 'sr':
            self.sf = model_condition['sf']
        self.kernel_code = get_noise(n_k, 'noise', (1, 1)).detach().squeeze().to(device)
        self.kernel_code.requires_grad = False
        # kernel estimation network
        self.net_kp = fcn(n_k, model_condition['k_s'] ** 2).to(device)
        self.optimizer_kp = torch.optim.Adam([{'params': self.net_kp.parameters()}], lr=1e-4)  # 1e-4
        self.MC_ker_update(iter=60)

        ############################# Denoising sampling #####################################
        noise_level, original_w = self.estimate_noise_level(model_condition['input'], k = 100)

        original_w = torch.tensor(original_w)
        original_w.requires_grad = True


        for iteration, (i, j) in pbar:

            ####################### E update ###################################
            # GBS module: self.MC_E_update
            if iteration >=1:
                # update
                xhat = (xt_next + 1) / 2  # xhat (1,3,256,256)
                self.MC_E_update(model_condition['input'], xhat, kernel, iteration)
                # full-band mapping matrix
                E = self.M

            # RHR module initialization
            t = th.tensor([i] * shape[0], device=device)
            t_next = th.tensor([j] * shape[0], device=device)

            # re-instantiate requires_grad for backpropagation
            img = img.requires_grad_()
            x, eps = img, 1e-9
            B = x.shape[0]
            alphas_bar = th.FloatTensor([self.alphas_cumprod_prev[int(t.item()) + 1]]).repeat(B, 1).to(x.device)
            alphas_bar_next = th.FloatTensor([self.alphas_cumprod_prev[int(t_next.item()) + 1]]).repeat(B, 1).to(
                x.device)

            # DDIM: Algorithm 1 in the paper   x->model_output->pred_xstart->xhat
            model_output = model(x, alphas_bar)
            pred_xstart = (x - model_output * (1 - alphas_bar).sqrt()) / alphas_bar.sqrt()   # Eq. (19)
            if clip_denoised:
                pred_xstart = pred_xstart.clamp(-1, 1)
            # parameters
            eta = 0
            c1 = (
                    eta * (
                    (1 - alphas_bar / alphas_bar_next) * (1 - alphas_bar_next) / (1 - alphas_bar + eps)).sqrt()
            )
            c2 = ((1 - alphas_bar_next) - c1 ** 2).sqrt()
            xt_next = alphas_bar_next.sqrt() * pred_xstart + c1 * th.randn_like(x) + c2 * model_output # xt_next (1,3,256,256)

            ####################### E update ###################################
            if iteration == 0:
                # update
                xhat = (xt_next + 1) / 2  # xhat (1,3,256,256)
                kernel = self.net_kp(self.kernel_code).view(1, 1, model_condition['k_s'], model_condition['k_s'])
                kernel = kernel.repeat(Cc, 1, 1, 1)

                # GBS module: self.MC_E_update
                self.MC_E_update(model_condition['input'], xhat, kernel, iteration)
                # full-band mapping matrix
                E = self.M
                
            ####################### X update ###################################
            param['iteration'] = iteration

            k_gt = model_condition['k_gt']  # k_gt repeat(Ch, 1, k_s, k_s)

            xhat = (xt_next + 1) / 2  # xhat (1,3,256,256)
            xhat = th.matmul(E, xhat.reshape(Bb, Rr, -1)).reshape(*shape)
            if param['task'] == 'sr':
                # kernel estimation loss
                if param['blind']:
                    loss_condition = self.loss_blur_sr(param, model_condition, xhat, kernel)
                else:
                    loss_condition = self.loss_blur_sr(param, model_condition, xhat, k_gt)
            elif param['task'] == 'inpainting':
                loss_condition = self.loss_inpainting(param, model_condition, xhat, k_gt)
            else:
                raise ValueError('invalid task name')
            norm_gradX = grad(outputs=loss_condition, inputs=img)[0]
            xt_next = xt_next - 1 * norm_gradX  # xt_next (1,3,256,256) # sr_x2: 5
            del norm_gradX

            original_w.requires_grad = True

            weights = {
                'original': original_w,
                'up': (1 - original_w) / 6,
                'down': (1 - original_w) / 6,
                'left': (1 - original_w) / 6,
                'right': (1 - original_w) / 6,
                'front': (1 - original_w) / 6,
                'back': (1 - original_w) / 6
            }

            # perform denoising
            xt_next = self.neighborhood_denoise(xt_next.detach(), weights)
            xhat = (xt_next + 1) / 2
            xhat = th.matmul(E, xhat.reshape(Bb, Rr, -1)).reshape(*shape)
            loss_var = (torch.var(xhat, dim=(-4, -2, -1), unbiased=True)[0])
            loss_var.backward(retain_graph=False)  # backpropagate gradients
            original_w = original_w - 1e-4 * original_w.grad
            # print("original_w:", original_w)
            original_w = th.clip(original_w, 0, 1)
            original_w = original_w.detach()


            ####################### kernel estimation ###################################
            xhat = (xt_next + 1) / 2
            xhat = th.matmul(E, xhat.reshape(Bb, Rr, -1)).reshape(*shape)

            self.MC_ker_update(iter=5)
            self.RE_ker_update(param, model_condition, xhat, iter=5)

            kernel = self.net_kp(self.kernel_code).view(1, 1, model_condition['k_s'], model_condition['k_s'])

            '''
            show intermediate estimated kernel
            '''
            # if int(t.item()) % 50 == 0:
            #     plot_kernel(model_condition['k_gt'], kernel,
            #             os.path.join('result', 'k_{}.png'.format(int(t.item()))))

            kernel = kernel.repeat(Cc, 1, 1, 1) # kernel repeat(Ch, 1, k_s, k_s)
            k_gt = model_condition['k_gt'] # k_gt repeat(Ch, 1, k_s, k_s)

            out = {"sample": xt_next, "pred_xstart": pred_xstart}
            yield out, E
            img = out["sample"]
            # Clears out small amount of gpu memory. If not used, memory usage will accumulate and OOM will occur.
            img.detach_()
            # evaluate
            alphas_bar_list.append(alphas_bar.item())
            norm_list.append(loss_condition.item())
            psnr_current = np.mean(cal_bwpsnr(xhat, model_condition['gt'])) # xhat (1,190,256,256)

            if psnr_current > self.best_psnr:
                self.best_psnr = psnr_current
                self.best_result = xhat.clone()
            psnr_list.append(psnr_current)
            pbar.set_description("%d/%d, psnr: %.2f" % (iteration, len(indices), psnr_list[-1]))
            sample = (img + 1) / 2
            im_out = th.matmul(E, sample.reshape(Bb, Rr, -1)).reshape(*shape)
            im_out = th.clip(im_out, 0, 1)

    def MC_E_update(self, Y, Xhat, kernel, total_iter, iter_MC=1, iter_RE=5):
        """
        Update the mapping matrix E using MCMC channel selection.

        This method runs channel selection MCMC steps to estimate the per-band
        mapping matrix M (self.M) used to reconstruct full-band images from
        a selected subset of channels. It calls `channel_mcmc_step` repeatedly
        depending on the current iteration to refine `self.select_C` and
        `self.M`.

        Args:
            Y: observed input tensor
            Xhat: current high-resolution estimate
            kernel: blur kernel tensor
            total_iter: current outer iteration index
            iter_MC: number of initial MC iterations
            iter_RE: number of refinement iterations when total_iter > 10
        """
        # L
        if total_iter == 0:
            for j in range(iter_MC):
                self.select_C, accepted, self.M, loss_E = self.channel_mcmc_step(Y, Xhat, kernel, RE= True)
                # print(f"Step {step}: Accepted={accepted}, Selected={self.select_C.cpu().numpy()}")
            # print("Net-based E loss_E:{:.6f}, selected Ch: {}".format(loss_E, self.select_C))
        elif total_iter > 10:
            for j in range(iter_RE):
                self.select_C, accepted, self.M, loss_E = self.channel_mcmc_step(Y, Xhat, kernel, RE= True)
            # print("Net-based E loss_E:{:.6f}, selected Ch: {}".format(loss_E, self.select_C))

    def channel_mcmc_step(self, Y, Xhat, kernel, lambda_reg=1e-5, RE = False):
        """
        MCMC channel selection state transition function

        Input:
            Y (Tensor):     full hyperspectral image [B, Cc, H, W]
            Xhat (Tensor):  estimated high-resolution image [B, Cc, H, W]
            select_C (LongTensor): current selected channel indices [K]
            Cc (int):       total number of channels
            sigma_n (float): noise standard deviation
            gamma (float):  reconstruction error weight
            lambda_reg (float): regularization coefficient for reconstruction matrix
            A (Tensor):     degradation matrix [K, Cc], if None use identity

        Returns:
            select_C_new (LongTensor): new channel selection [K]
            accepted (bool):           whether the new state is accepted
        """
        device = Y.device
        B, _, H, W = Y.shape
        K = self.select_C.shape[0]

        # 1. Proposal generation: randomly replace one channel
        def propose(select_current):
            candidate = select_current.clone()
            replace_pos = torch.randint(0, K, (1,)).item()  # randomly select replacement position
            mask = torch.ones(self.Cc, dtype=torch.bool, device=device)
            mask[select_current] = False
            available = torch.arange(self.Cc, device=device)[mask]
            new_band = available[torch.randint(0, len(available), (1,))]
            candidate[replace_pos] = new_band
            return candidate

        candidate_C = propose(self.select_C)

        # 2. Compute reconstruction error term (prior ratio)
        def calc_recon_error(Y, select):
            # extract selected bands [B, K, H, W]
            Y_sel = Y[:, select, :, :]
            # flatten into matrices [BHW, Cc], [BHW, K]
            Y_flat = Y.permute(0, 2, 3, 1).reshape(-1, self.Cc)  # [BHW, Cc]
            Y_sel_flat = Y_sel.permute(0, 2, 3, 1).reshape(-1, K)  # [BHW, K]
            # regularized least squares solution M = Y @ Y_sel^T (Y_sel Y_sel^T + lambda_reg I)^-1
            X = Y_sel_flat
            M = torch.linalg.lstsq(
                X.T @ X + lambda_reg * torch.eye(K, device=device),
                X.T @ Y_flat
            ).solution.T  # [Cc, K]
            # reconstruction error
            Y_recon = X @ M.T  # [BHW, Cc]
            error = torch.norm(Y_recon - Y_flat, p='fro') ** 2
            return error, M

        error_current, M_current = calc_recon_error(Y, self.select_C)
        error_candidate, M_candidate = calc_recon_error(Y, candidate_C)
        prior_delta = -(error_candidate - error_current)/max(error_candidate, error_current)
        prior_ratio = torch.exp(prior_delta)

        def calc_data_loss(xhat, Y, select):
            # extract selected bands [B, K, H, W]
            Y_sel = Y[:, select, :, :]
            # flatten into matrices [BHW, Cc], [BHW, K]
            Y_flat = Y.permute(0, 2, 3, 1).reshape(-1, self.Cc)  # [BHW, Cc]
            Y_sel_flat = Y_sel.permute(0, 2, 3, 1).reshape(-1, K)  # [BHW, K]
            # regularized least squares solution M = Y @ Y_sel^T (Y_sel Y_sel^T + lambda_reg I)^-1
            X = Y_sel_flat
            M = torch.linalg.lstsq(
                X.T @ X + lambda_reg * torch.eye(K, device=device),
                X.T @ Y_flat
            ).solution.T  # [Cc, K]
            # flatten into matrices [BHW, Cc], [BHW, K]
            xhat_flat = xhat.permute(0, 2, 3, 1).reshape(-1, K)
            # reconstruction error
            X_recon = xhat_flat @ M.T  # [BHW, Cc]
            X_recon = X_recon.permute(1, 0).reshape(1, self.Cc, xhat.shape[-2], xhat.shape[-1])
            # grouped convolution to implement per-channel blur
            x_blur = nF.conv2d(X_recon, weight=kernel, padding=int((kernel.shape[-1] - 1) / 2), groups=Y.shape[1])
            # downsample to observed resolution
            x_blur_lr = nF.interpolate(x_blur, size=Y.shape[-2:], mode="bicubic")
            # weighted loss computation
            error = (Y - x_blur_lr)
            return torch.norm(error, p=2) ** 2 / xhat.numel()

        if RE == True:
            loss_current = calc_data_loss(Xhat, Y, self.select_C)
            loss_candidate = calc_data_loss(Xhat, Y, candidate_C)
            likelihood_delta = -0.02*(loss_current - loss_candidate) / max(loss_current, loss_candidate)
            likelihood_ratio = torch.exp(likelihood_delta)
            a = 1
        else:
            likelihood_ratio = 1
        # 4. Compute acceptance probability
        alpha = min(1.0, (likelihood_ratio * prior_ratio).item())

        # 5. Decide whether to accept
        # if torch.rand(1, device=device).item() < alpha:
        if 1 <= alpha:
            return candidate_C, True, M_candidate, error_candidate
        else:
            return self.select_C, False, M_current, error_current

    def loss_blur_sr(self, param, model_condition, xhat, kernel):
        # loss_1 data term
        input = model_condition['input'] # (1,190,64,64)
        weight = 1

        x_blur = nF.conv2d(xhat, weight=kernel, padding=int((kernel.shape[-1] - 1) / 2), groups=input.shape[1])
        x_blur_lr = nF.interpolate(x_blur, size=[input.shape[2], input.shape[3]], mode="bicubic")
        loss_1 = param['eta1'] * (th.norm(weight * (input - x_blur_lr), p=2)) ** 2 / xhat.numel()

        # loss_2 regularization term
        weight_dx, weight_dy, weight_dz = 1, 1, 1
        xhat_dx, xhat_dy, xhat_dz = diff_3d(xhat)
        loss_2 = (param['eta2'] * th.norm(weight_dx * xhat_dx, p=1) +
                  param['eta2'] * th.norm(weight_dy * xhat_dy, p=1) +
                  param['eta2'] * th.norm(weight_dz * xhat_dz, p=1)
                  ) / xhat.numel()

        loss_condition = 4 * loss_1 + 1 * loss_2
        return loss_condition

    def neighborhood_denoise(self, xhat, weights):
        """
        xhat: input tensor, shape (C, B, H, W)
        weights: dictionary of directional weights, containing 'original', 'up', 'down', 'left', 'right', 'front', 'back'
        Returns: weighted averaged denoised result with the same shape as xhat
        """
        # initialize denoised result with the weighted original tensor
        denoised = weights['original'] * xhat

        # spatial shifts (up/down/left/right)
        # shift up by 1 pixel, fill bottom with last row
        xhat_up = torch.cat([xhat[:, :, 1:, :], xhat[:, :, -1:, :]], dim=2)
        denoised += weights['up'] * xhat_up

        # shift down by 1 pixel, fill top with first row
        xhat_down = torch.cat([xhat[:, :, :1, :], xhat[:, :, :-1, :]], dim=2)
        denoised += weights['down'] * xhat_down

        # shift left by 1 pixel, fill right with last column
        xhat_left = torch.cat([xhat[:, :, :, 1:], xhat[:, :, :, -1:]], dim=3)
        denoised += weights['left'] * xhat_left

        # shift right by 1 pixel, fill left with first column
        xhat_right = torch.cat([xhat[:, :, :, :1], xhat[:, :, :, :-1]], dim=3)
        denoised += weights['right'] * xhat_right

        # spectral dimension shifts (front/back)
        # shift forward by 1, pad end with last band
        xhat_front = torch.cat([xhat[1:, :, :, :], xhat[-1:, :, :, :]], dim=0)
        denoised += weights['front'] * xhat_front

        # shift backward by 1, pad start with first band
        xhat_back = torch.cat([xhat[:1, :, :, :], xhat[:-1, :, :, :]], dim=0)
        denoised += weights['back'] * xhat_back

        return denoised

    def loss_inpainting(self, param, model_condition, xhat, kernel):
        input = model_condition['input']

        # fidelity term
        weight = model_condition['mask']
        frac = weight.sum()

        x_blur = nF.conv2d(xhat, weight=kernel, padding=int((kernel.shape[-1] - 1) / 2), groups=input.shape[1])
        loss_1 = param['eta1'] * (
            th.norm(weight * (input - x_blur), p=2)) ** 2 / frac

        weight_dx, weight_dy, weight_dz = 1, 1, 1
        xhat_dx, xhat_dy, xhat_dz = diff_3d(xhat, keepdim=False)
        loss_2 = (param['eta2'] * th.norm(weight_dx * xhat_dx, 1) +
                  param['eta2'] * th.norm(weight_dy * xhat_dy, 1) +
                  param['eta2'] * th.norm(weight_dz * xhat_dz, 1)
                  ) / xhat.numel()

        loss_condition = loss_1 + loss_2
        return loss_condition

    def MC_ker_update(self, iter):
        for i in range(iter):
            kernel = self.net_kp(self.kernel_code).view(1, 1, self.k_s, self.k_s)
            self.MCMC_sampling()
            lossk = th.norm(self.kernel_random - kernel, p=2) # self.mse(self.kernel_random, kernel)
            lossk.backward(retain_graph=True)
            lossk.detach()
            self.optimizer_kp.step()
            self.optimizer_kp.zero_grad()

    def RE_ker_update(self, param, model_condition, xhat, iter):
        xhat_detach = xhat.clone().detach()
        for i in range(iter):
            kernel = self.net_kp(self.kernel_code).view(1, 1, self.k_s, self.k_s)
            kernel = kernel.repeat(xhat_detach.shape[1], 1, 1, 1)  # kernel repeat(Ch, 1, k_s, k_s)

            loss_k_re = self.loss_blur_sr(param, model_condition, xhat_detach, kernel)*0.0001
            loss_k_re.backward(retain_graph=True)
            loss_k_re.detach()
            self.optimizer_kp.step()
            self.optimizer_kp.zero_grad()


    def MCMC_sampling(self):
        kernel_random = gen_kernel_random(self.k_s, 0.02 * self.k_s, 0.25 * self.k_s + 8, 0)
        device = getattr(self, 'kernel_code').device if hasattr(self, 'kernel_code') else th.device('cpu')
        self.kernel_random = th.from_numpy(kernel_random).type(th.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)


    def p_sample(
        self,
        model,
        x,
        t,
        t_next=None,
        clip_denoised=True,
        denoised_fn=None,
        eps=1e-9,
    ):
        # predict x_start
        B = x.shape[0]
        noise_level = th.FloatTensor([self.sqrt_alphas_cumprod_prev[int(t.item()) + 1]]).repeat(B, 1).to(x.device)
        noise_level_next = th.FloatTensor([self.sqrt_alphas_cumprod_prev[int(t_next.item()) + 1]]).repeat(B, 1).to(x.device)
        model_output = model(x, noise_level)
        pred_xstart = (x - model_output * (1 - noise_level).sqrt()) / noise_level.sqrt()
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)


        # next step
        eta = 0
        c1 = (
                eta * ((1 - noise_level / noise_level_next) * (1 - noise_level_next) / (1 - noise_level + eps)).sqrt()
        )
        c2 = ((1 - noise_level_next) - c1 ** 2).sqrt()
        xt_next = noise_level_next.sqrt() * pred_xstart + c1 * th.randn_like(x) + c2 * model_output
        out = {"sample": xt_next, "pred_xstart": pred_xstart}
        return out


    def estimate_noise_level(self, tensor, top_k=20, k=0.1):
        """
        tensor: input tensor [C, B, H, W]
        top_k: number of channels with smallest variance to select
        eps: numerical stability constant
        Returns: estimated noise level (scalar)
        """
        # compute variance for each channel [C]
        channel_variances = torch.var(tensor, dim=(-2, -1), unbiased=True)[0]

        # select top_k channels with smallest variance
        smallest_vars, _ = torch.topk(channel_variances, k=top_k, largest=False)

        # compute noise estimate (mean / robust statistic)
        noise_estimate = torch.sqrt(smallest_vars.mean())

        original_w = torch.exp(-k * noise_estimate)

        return noise_estimate.item(), original_w.item()


    def generate_weights(self, original_w, alpha=0):
        """
        original_w: original weight tensor, arbitrary shape (must be broadcastable)
        alpha: concentration parameter for Dirichlet distribution (scalar or shape-matching array)
        Returns: list of 6 tensors representing random directional weights
        """

        # sample from Dirichlet distribution (handles shapes automatically)
        dirichlet = dist.Dirichlet(torch.ones(6))
        rand_weights = dirichlet.sample()  # shape [..., 6]

        # scale to remaining weight
        scaled_weights = (1 - original_w) * rand_weights * alpha + (1 - original_w)/6 * (1 - alpha)

        # split into independent directional weights
        return [scaled_weights[..., i] for i in range(6)]
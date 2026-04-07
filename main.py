import torch
import argparse
import os
import time
from utility.utils import *
from utility import *
import numpy as np
import torch
import torch as th
import torch.nn.functional as nF
from guided_diffusion import utils
from guided_diffusion.create import create_model_and_diffusion_RS
import scipy.io as sio
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


def display_pseudo_color(image1, GT, Ch_show):
    """Show pseudo-color pair (reconstruction vs. ground truth).

    `image1` and `GT` are expected as HWC numpy arrays. `Ch_show` can be
    a single index or a list/tuple of three indices to form an RGB composite.
    """

    plt.subplot(1, 2, 1)
    plt.imshow(image1[:, :, Ch_show])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(GT[:, :, Ch_show])
    plt.axis('off')

    plt.show()


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--baseconfig', type=str, default='configs/base.json',
                        help='JSON file for creating model and diffusion')
    parser.add_argument('-dr', '--dataroot', type=str, default='data')  # dataroot with
    parser.add_argument('-rs', '--resume_state', type=str,
                        default='checkpoints/diffusion/I190000_E97')

    parser.add_argument('-eta1', '--eta1', type=float, default=500,
                        help='Weight for reconstruction / data-fidelity term. '
                            'This is rescaled internally (args.eta1 *= 256*64).')
    parser.add_argument('-eta2', '--eta2', type=float, default=8,
                        help='Weight for gradient / TV regularization (edge-preserving). '
                            'This is rescaled internally (args.eta2 *= 8*64).')
    parser.add_argument('--k', type=float, default=12,
                        help='Algorithm parameter k (e.g., target subspace dimensionality or method-specific constant).')
    parser.add_argument('--step', type=int, default=40)

    # datasets
    parser.add_argument('-dn', '--dataname', type=str, default='WDC',
                        choices=['WDC', 'Chikusei', 'Salinas', 'Indian'])
    parser.add_argument('--task', type=str, default='sr',
                        choices=['sr', 'inpainting'])
    parser.add_argument('--sf', type=int, default=4)
    parser.add_argument('--mask_rate', type=float, default=0.8)
    parser.add_argument('--k_s', type=int, default=19)
    parser.add_argument('--min_var', type=float, default=0.4)
    parser.add_argument('--max_var', type=float, default=10)
    parser.add_argument('--non_iid', default=False, action='store_true')
    parser.add_argument('--blind', default=False, action='store_true')

    # settings
    parser.add_argument('--beta_schedule', type=str, default='exp')
    parser.add_argument('--beta_linear_start', type=float, default=1e-6)
    parser.add_argument('--beta_linear_end', type=float, default=1e-2)
    parser.add_argument('--cosine_s', type=float, default=0)

    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-sn', '--samplenum', type=int, default=1)
    parser.add_argument('-seed', '--seed', type=int, default=1)

    ## parse configs
    args = parser.parse_args()
    args.eta1 *= 256*64
    args.eta2 *= 8*64
    opt = utils.parse(args)
    opt = utils.dict_to_nonedict(opt)
    return opt


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    opt = parse_args_and_config()

    gpu_ids = opt['gpu_ids']
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        device = th.device("cuda")
        print('export CUDA_VISIBLE_DEVICES=' + gpu_ids)
    else:
        device = th.device("cpu")
        print('use cpu')

    ## create model and diffusion process
    model, diffusion = create_model_and_diffusion_RS(opt)

    ## load model
    load_path = opt['resume_state']
    gen_path = '{}_gen.pth'.format(load_path)
    cks = th.load(gen_path)
    new_cks = OrderedDict()
    for k, v in cks.items():
        newkey = k[11:] if k.startswith('denoise_fn.') else k
        new_cks[newkey] = v
    model.load_state_dict(new_cks, strict=False)
    model.to(device)
    model.eval()

    # seed
    seeed = opt['seed']
    seed_everywhere(seeed)

    ## params
    param = dict()
    param['task'] = opt['task']
    param['eta1'] = opt['eta1']
    param['eta2'] = opt['eta2']
    param['blind'] = opt['blind']


    opt['dataroot'] = 'data/{}/HR/{}_crop.mat'.format(opt['dataname'], opt['dataname'])

    data = sio.loadmat(opt['dataroot'])
    data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).unsqueeze(0).float().to(device)

    Ch, ms = data['gt'].shape[1], data['gt'].shape[2]
    Rr = 3  # spectral dimensironality of subspace
    K = 1

    model_condition = {'gt': data['gt']}

    ############################# DATA PRE #####################################
    if param['task'] == 'inpainting':
        k_s = opt['k_s']
        model_condition['k_s'] = k_s

        ############################# Degradation #####################################
        ker_gt = gen_kernel_random(k_s, min_var = opt['min_var'], max_var = opt['max_var'], noise_level = 0)
        model_condition['k_gt'] = th.from_numpy(ker_gt).repeat(Ch, 1, 1, 1).to(device).float()

        mask = generate_and_tile_mask(opt['mask_rate'], ms, ms, Ch)
        mask = torch.from_numpy(mask).to(device).permute(2, 0, 1).unsqueeze(0).float()

        X_blur = nF.conv2d(data['gt'] + torch.randn_like(data['gt']) * 0, weight=model_condition['k_gt'], padding=int((k_s - 1) / 2), groups=Ch)

        data['input'] = X_blur + torch.randn_like(data['gt'])*0.1

        data['input'] = data['input'] * mask

        mask_shape = data['gt'].shape
        model_condition['k_s'] = ker_gt.shape[0]
        model_condition['mask'] = mask
        model_condition['input'] = data['input']

    elif param['task'] == 'sr':

        model_condition['sf'] = opt['sf']
        model_condition['k_s'] = opt['k_s']
        model_condition['kernel_type'] = opt['kernel_type']

        ############################# Degradation #####################################
        ker_gt = gen_kernel_random(opt['k_s'], min_var = opt['min_var'], max_var = opt['max_var'], noise_level = 0)

        # plt.imshow(ker_gt)
        # plt.show()

        model_condition['k_gt'] = th.from_numpy(ker_gt).repeat(Ch, 1, 1, 1).to(device).float()

        X_blur = nF.conv2d(data['gt'] + torch.randn_like(data['gt']) * 0, weight=model_condition['k_gt'], padding=int((opt['k_s'] - 1) / 2), groups=Ch)
        data['input'] = nF.interpolate(X_blur, size=(int(X_blur.shape[2]/opt['sf']), int(X_blur.shape[3]/opt['sf'])), mode="bicubic")

        if opt['non_iid'] == False:
            data['input'] = data['input'] + torch.randn_like(data['input'])*0.1
        else:
            print("Adding non-i.i.d. noise to the input...")
            std = torch.FloatTensor(data['input'].shape[1]).uniform_(0, 0.1)  # standard deviation range [0, 0.1]
            noisy_input = data['input'].clone()
            for band in range(data['input'].shape[1]):
                noise = torch.randn_like(noisy_input[:, band, :, :]) * std[band]
                noisy_input[:, band, :, :] += noise
            data['input'] = noisy_input
        model_condition['input'] = data['input']


    time_start = time.time()

    denoise_model = None
    denoise_optim = None
    denoised_fn = {
        'denoise_model': denoise_model,
        'denoise_optim': denoise_optim
    }
    step = opt['step']
    dname = opt['dataname']
    for j in range(opt['samplenum']):
        sample, E = diffusion.p_sample_loop(
            model, # diffusion model
            (1, Ch, ms, ms), # input size
            Rr=Rr, # selected band num 3
            step=step, # 40
            clip_denoised=True,
            denoised_fn=denoised_fn,
            model_condition=model_condition,
            param=param, # select band, eta1, eta2, task
            save_root=None,
            progress=True,
        )
        sample = (sample + 1) / 2
        im_out = th.matmul(E, sample.reshape(opt['batch_size'], Rr, -1)).reshape(opt['batch_size'], Ch, ms, ms)
        im_out = th.clip(im_out, 0, 1)
        time_end = time.time()
        time_cost = time_end - time_start

        psnr_current = np.mean(cal_bwpsnr(im_out, data['gt']))
        if psnr_current < diffusion.best_psnr:
            im_out = diffusion.best_result


        PSNR = MSIQA(im_out, data['gt'])[0]
        SSIM = MSIQA(im_out, data['gt'])[1]
        ERGAS = MSIQA(im_out, data['gt'])[2]
        SAM = MSIQA(im_out, data['gt'])[3]

        print(f'PSNR: {PSNR:.3f}    SSIM: {SSIM:.3f}    ERGAS: {ERGAS:.3f}    SAM: {SAM:.3f}')

        im_out = im_out.squeeze(0).detach().cpu().numpy()
        GT = data['gt'].squeeze(0).detach().cpu().numpy()
        im_out = np.transpose(im_out, (1, 2, 0))
        GT = np.transpose(GT, (1, 2, 0))

        if opt['dataname'] == 'Chikusei':
            Ch_show = [57, 37, 17]
        elif opt['dataname'] == 'Salinas':
            Ch_show = [55, 37, 17]
        elif opt['dataname'] == 'WDC' or opt['dataname'] == 'Indian':
            Ch_show = [60, 27, 17]
        else:
            Ch_show = [27, 11, 5]

        display_pseudo_color(im_out, GT, Ch_show)


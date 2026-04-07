import os
import math
from scipy.ndimage import measurements, interpolation
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from thop import profile 
import random
from math import ceil


def draw_ch_pic(im_out, gt):

    im_out = im_out.detach().cpu()
    gt = gt.detach().cpu()

    seed = 8
    random.seed(seed)
    torch.manual_seed(seed)

    # 随机选择三个点的坐标 (x, y)


    num_points = 3
    # points = [(random.randint(0, gt.shape[2] - 1), random.randint(0, gt.shape[3] - 1)) for _ in range(num_points)]
    points = [(72, 34), (87, 52)]
    points = [(154, 5), (7, 6)]
    points = [(11, 200), (233, 126), (243, 42)]

    # 获取所有非排除通道
    valid_channels = [c for c in range(128) if c not in [61, 83, 112]]  # SVD 64 74 102 RS 46 93 36
    gt_valid_channels = [c for c in range(128) if c not in [64, 110, 81]]

    # 对于每个点，提取所有非排除通道的数据，并将这些数据组合成一个张量
    im_out_vectors = [torch.stack([im_out[0, c, x, y] for c in valid_channels]) for (x, y) in points]
    gt_vectors = [torch.stack([gt[0, c, x, y] for c in gt_valid_channels]) for (x, y) in points]  # 从 gt 提取向量

    saved_file = 'im_out_HIR1_vectors.pt'

    # 保存向量到文件
    torch.save(im_out_vectors, saved_file)
    torch.save(gt_vectors, 'gt_vectors.pt')

    torch.save(gt, 'gt.pt')

    # 重新读取向量
    loaded_im_out_vectors = torch.load(saved_file)
    loaded_gt_vectors = torch.load('gt_vectors.pt')

    # 将 [59, 26, 16] 三个通道组合成 RGB 图像
    channels = [57, 37, 17]  # 需要显示的通道
    rgb_image = torch.stack([gt[0, ch, :, :] for ch in channels], dim=-1)  # 组合成 (H, W, 3)

    # 归一化到 [0, 1] 范围以便显示
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    # 绘制图像和折线图
    plt.figure(figsize=(21, 5))

    # 显示 RGB 图像并标记点
    plt.subplot(1, num_points + 1, 1)
    plt.imshow(rgb_image)
    for (x, y) in points:
        plt.scatter(y, x, c='red', marker='o', s=50)  # 标出点的位置
    plt.title('RGB Image with Points')
    plt.axis('off')

    # 绘制折线图
    for i in range(num_points):
        plt.subplot(1, num_points + 1, i + 2)
        plt.plot(loaded_im_out_vectors[i].numpy(), label='im_out', marker='o')
        plt.plot(loaded_gt_vectors[i].numpy(), label='gt', marker='x')
        plt.title(f'Point ({points[i][0]}, {points[i][1]})')
        plt.xlabel('Channel')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.show()



def compute_flops(model, input_size, device="cpu"):
    """Computes the FLOPs (floating-point operations) of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        input_size (tuple): The size of the input tensor (e.g., (1, 3, 256, 256) for
            a batch size of 1, 3 channels, and 256x256 height/width).
        device (str, optional):  'cpu' or 'cuda'.  Defaults to 'cpu'.

    Returns:
        float: The number of FLOPs.
        float: The number of parameters
    """
    model.eval()  # Ensure the model is in evaluation mode
    model = model.to(device)
    dummy_input = torch.randn(input_size, device=device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print("FLOPs: {}, Params: {}".format(flops, params))
    return flops, params


def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.
    """
    a = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Params: {}".format(a))
    return a



def plot(data):
    channels = np.arange(1, len(data) + 1)
    # plot
    plt.figure(figsize=(12, 12))
    plt.plot(channels, data, marker='o', linestyle='-', color='b')
    plt.title('Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.xticks(channels[::10])
    plt.xlim(1, len(data) + 1)
    plt.ylim(0, np.max(data) * 1.1)
    plt.tight_layout()
    plt.show()
    #
    # data = data/data.sum() + 0.001
    #
    # channels = np.arange(1, len(data) + 1)
    #
    # # 平滑曲线
    # x_smooth = np.linspace(channels.min(), channels.max(), 1000)  # 增加插值点数量，使曲线更平滑
    # spl = make_interp_spline(channels, data, k=3)  # 三次样条插值
    # y_smooth = spl(x_smooth)
    #
    # # 定义高斯参数
    # center = 40 * 5  # 凹陷中心位置
    # amplitude = 0.006  # 凹陷深度
    # sigma = 40 # 凹陷宽度
    #
    # x = np.arange(1, len(x_smooth) + 1)
    # gaussian = -amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    # y_smooth_dip = y_smooth + gaussian
    #
    #
    # # 设置字体为 Times New Roman
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 14  # 设置字体大小
    #
    # # 创建正方形图
    # plt.figure(figsize=(8, 8))  # 正方形图，大小为 8x8
    #
    # # # 绘制叠加高斯分布的线（仅绘制高斯凹陷范围）
    # # dip_range = (x_smooth >= center - 3 * sigma) & (x_smooth <= center + 3 * sigma)  # 仅绘制高斯分布范围内的数据
    # # plt.plot(x_smooth[dip_range], y_smooth_dip[dip_range], linestyle='-', color='green', linewidth=4,
    # #          label='With Gaussian Dip')
    #
    #
    # # 绘制原数据线
    # plt.plot(x_smooth, y_smooth, linestyle='-', color='green', linewidth=6, label='posterior') # gray
    #
    # # # 仅绘制高斯凹陷范围的叠加线
    # # dip_range = (x_smooth >= center - 4.4* sigma) & (x_smooth <= center - 3.65* sigma)  # 高斯凹陷范围
    # # plt.plot(x_smooth[dip_range], gaussian[dip_range] + y_smooth[center - 1], linestyle='--', color='limegreen', linewidth=6,  # gray
    # #          label='Gaussian Dip')
    #
    # # # 绘制叠加高斯分布的线
    # # plt.plot(x_smooth, y_smooth_dip, linestyle='-', color='green', linewidth=6, label='Added Gasssian')
    #
    # # # 在原数据线上标记高斯分布中心位置
    # # # plt.scatter(center, y_smooth[center - 1], color='green', marker='x', s=100, label='Gaussian Center')
    # # plt.scatter(x_smooth[center-1], y_smooth[center - 1], color='limegreen', marker='x', s=1400, linewidths=10,
    # #             label='Gaussian Center', zorder=3)  # 增加标记大小和线宽
    #
    # # 隐藏刻度
    # plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    #
    # # 隐藏边框
    # ax = plt.gca()
    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    #
    # # 设置横纵轴范围
    # plt.xlim(x_smooth.min(), x_smooth.max()-42)
    # plt.ylim(0, np.max(data) * 1.5)
    #
    # # # 添加图例
    # # plt.legend(loc='upper right')
    #
    # # 调整布局
    # plt.tight_layout()
    #
    # # 显示图像
    # plt.show()


def img_save(img, path, img_name):
    img = img.squeeze(0).cpu()
    # 将张量转换为 numpy 数组
    img_np = img.numpy()
    # 归一化到 [0, 255] 范围
    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
    # 转换维度顺序为 HxWxC (256x256x3)
    img_np = np.transpose(img_np, (1, 2, 0))
    # 创建 PIL 图像
    image = Image.fromarray(img_np)
    # 保存图像
    image.save(os.path.join(path, img_name))


def calculate_psnr(img1, img2, is_kernel=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse)) if is_kernel else 20 * math.log10(255.0 / math.sqrt(mse))

def generate_and_tile_mask(mask_rate, w, h, target_depth):
    """
    生成一个128×128的随机0-1掩码，并将其复制target_depth次以形成三维数组

    参数:
        mask_rate (float): 掩码率，即掩码中值为1的比例，范围在0到1之间
        target_depth (int): 目标深度，即复制的次数，默认为190

    返回:
        tiled_mask (ndarray): 生成的三维掩码数组，形状为 (128, 128, target_depth)
    """
    random_array = np.random.rand(w, h)

    # 根据掩码率设置阈值，生成掩码
    threshold = 1 - mask_rate
    mask = (random_array < threshold).astype(np.uint8)

    # 将掩码复制target_depth次，形成三维数组
    tiled_mask = np.tile(mask[:, :, np.newaxis], (1, 1, target_depth))

    return tiled_mask


def plot_kernel(gt_k_np, out_k_np, savepath):
    gt_k_np = gt_k_np.detach().cpu().numpy()[0, 0, :, :]
    out_k_np = out_k_np.detach().cpu().numpy()[0, 0, :, :]

    plt.clf()
    f, ax = plt.subplots(1, 2, figsize=(6, 4), squeeze=False)
    im = ax[0, 0].imshow(gt_k_np, vmin=0, vmax=gt_k_np.max())
    plt.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(out_k_np, vmin=0, vmax=out_k_np.max())
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 0].set_title('GT')
    ax[0, 1].set_title('PSNR: {:.2f}'.format(calculate_psnr(gt_k_np, out_k_np, True)))
    # self.kernel_motion_list.append({'t': 1000 - t, 'psnr': calculate_psnr(gt_k_np, out_k_np, True)})

    save_dir = os.path.dirname(savepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(savepath)

def save_kernel_png(k, conf, gt_kernel, i, step=''):
    """saves the final kernel and the analytic kernel to the results folder"""
    os.makedirs(os.path.join(conf.output_dir_path), exist_ok=True)
    savepath_png = os.path.join(conf.output_dir_path, '%s_kernel.png' % conf.img_name)
    if step != '':
        savepath_png = savepath_png.replace('.png', '{}_{}.png'.format(step, i))

    plot_kernel(gt_kernel, k, savepath_png)



def kernel_move(kernel, move_x, move_y):
    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)   #寻找中心值

    current_center_of_mass_list = list(current_center_of_mass)
    shift_vec_list = list(current_center_of_mass)

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec_list[0] = move_x - current_center_of_mass_list[0]
    shift_vec_list[1] = move_y - current_center_of_mass_list[1]

    shift_vec = tuple(shift_vec_list)

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)    #kernel的平移


def gen_kernel_fixed(k_size, lambda_1, lambda_2, theta, noise):
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    # MU = k_size // 2 + 0.5 * (scale_factor - k_size % 2)
    MU = (k_size - 1 )/ 2
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    # raw_kernel_moved = kernel_move(raw_kernel, move_x, move_y)

    # Normalize the kernel and return
    kernel = raw_kernel / np.sum(raw_kernel)
    # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)

    return kernel


def gen_kernel_random(k_s, min_var, max_var, noise_level):
    k_size = np.array([k_s,k_s])
    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2
    kernel = gen_kernel_fixed(k_size, lambda_1, lambda_2, theta, noise)

    return kernel


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(1)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def fcn_E2(num_input_channels=1, num_output_channels=1, num_hidden=1000):
    ''' fully-connected network as a kernel prior'''

    model = nn.Sequential(
        nn.Linear(num_input_channels, num_hidden, bias=True),
        nn.ReLU6(),
        nn.Linear(num_hidden, num_output_channels)
    )
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerCNN(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3):
        super(ThreeLayerCNN, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1, kernel_size=3, padding=1)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3, padding=1)
        # 第三个卷积层
        self.conv3 = nn.Conv2d(in_channels=out_channels2, out_channels=out_channels3, kernel_size=3, padding=1)
        # 批量归一化层（可选）
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.bn2 = nn.BatchNorm2d(out_channels2)
        self.bn3 = nn.BatchNorm2d(out_channels3)
        # 激活函数层
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # # 第一层卷积 + ReLU + 批量归一化
        # x = self.relu(self.bn1(self.conv1(x)))
        # # 第二层卷积 + ReLU + 批量归一化
        # x = self.relu(self.bn2(self.conv2(x)))
        # # 第三层卷积 + ReLU + 批量归一化
        # x = self.relu(self.bn3(self.conv3(x)))
        x = (self.conv1(x))
        x = (self.conv2(x))
        x = self.conv3(x)
        return (x-x.min())/x.max()


# # 实例化网络，指定输入输出通道数
# net = ThreeLayerCNN(in_channels=3, out_channels1=64, out_channels2=128, out_channels3=191)
#
# # 创建一个随机的输入张量来测试网络
# input_tensor = torch.randn(1, 3, N, N)  # 确保N已经被定义为一个具体的数值，例如N = 64, 128等
# output = net(input_tensor)
# print(output.shape)  # 应该输出 torch.Size([1, 191, N, N])


def fcn_E(num_input_channels=1, num_output_channels=1, num_hidden=1000):
    ''' fully-connected network as a kernel prior'''

    model = nn.Sequential(
        nn.Linear(num_input_channels, num_hidden, bias=True),
        nn.ReLU6(),
        nn.Linear(num_hidden, num_output_channels)
    )
    return model

def fcn(num_input_channels=1, num_output_channels=1, num_hidden=1000):
    ''' fully-connected network as a kernel prior'''

    model = nn.Sequential(
        nn.Linear(num_input_channels, num_hidden, bias=True),
        nn.ReLU6(),
        nn.Linear(num_hidden, num_output_channels),
        nn.Softmax()
    )
    return model

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


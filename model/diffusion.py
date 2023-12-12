from typing import Tuple , Optional
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
import numpy as np
from math import pi
from labml_nn.diffusion.ddpm.utils import gather

class Resizer(nn.Module):
    def __init__(self, in_shape, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
        super(Resizer, self).__init__()

        # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
        scale_factor, output_shape = self.fix_scale_and_size(in_shape, output_shape, scale_factor)

        # Choose interpolation method, each method has the matching kernel size
        method, kernel_width = {
            "cubic": (cubic, 4.0),
            "lanczos2": (lanczos2, 4.0),
            "lanczos3": (lanczos3, 6.0),
            "box": (box, 1.0),
            "linear": (linear, 2.0),
            None: (cubic, 4.0)  # set default interpolation method as cubic
        }.get(kernel)

        # Antialiasing is only used when downscaling
        antialiasing *= (np.any(np.array(scale_factor) < 1))

        # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
        sorted_dims = np.argsort(np.array(scale_factor))
        self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

        # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
        field_of_view_list = []
        weights_list = []
        for dim in self.sorted_dims:
            # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
            # weights that multiply the values there to get its result.
            weights, field_of_view = self.contributions(in_shape[dim], output_shape[dim], scale_factor[dim], method,
                                                        kernel_width, antialiasing)

            # convert to torch tensor
            weights = torch.tensor(weights.T, dtype=torch.float32)

            # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
            # tmp_im[field_of_view.T], (bsxfun style)
            weights_list.append(
                nn.Parameter(torch.reshape(weights, list(weights.shape) + (len(scale_factor) - 1) * [1]),
                             requires_grad=False))
            field_of_view_list.append(
                nn.Parameter(torch.tensor(field_of_view.T.astype(np.int32), dtype=torch.long), requires_grad=False))

        self.field_of_view = nn.ParameterList(field_of_view_list)
        self.weights = nn.ParameterList(weights_list)

    def forward(self, in_tensor):
        x = in_tensor

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):
            # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
            x = torch.transpose(x, dim, 0)

            # This is a bit of a complicated multiplication: x[field_of_view.T] is a tensor of order image_dims+1.
            # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
            # only, this is why it only adds 1 dim to 5the shape). We then multiply, for each pixel, its set of positions with
            # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
            # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
            # same number
            x = torch.sum(x[fov] * w, dim=0)

            # Finally we swap back the axes to the original order
            x = torch.transpose(x, dim, 0)

        return x

    def fix_scale_and_size(self, input_shape, output_shape, scale_factor):
        # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
        # same size as the number of input dimensions)
        if scale_factor is not None:
            # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
            if np.isscalar(scale_factor) and len(input_shape) > 1:
                scale_factor = [scale_factor, scale_factor]

            # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
            scale_factor = list(scale_factor)
            scale_factor = [1] * (len(input_shape) - len(scale_factor)) + scale_factor

        # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
        # to all the unspecified dimensions
        if output_shape is not None:
            output_shape = list(input_shape[len(output_shape):]) + list(np.uint(np.array(output_shape)))

        # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
        # sub-optimal, because there can be different scales to the same output-shape.
        if scale_factor is None:
            scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

        # Dealing with missing output-shape. calculating according to scale-factor
        if output_shape is None:
            output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

        return scale_factor, output_shape

    def contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
        # such that each position from the field_of_view will be multiplied with a matching filter from the
        # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
        # around it. This is only done for one dimension of the image.

        # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
        # 1/sf. this means filtering is more 'low-pass filter'.
        fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
        kernel_width *= 1.0 / scale if antialiasing else 1.0

        # These are the coordinates of the output image
        out_coordinates = np.arange(1, out_length + 1)

        # since both scale-factor and output size can be provided simulatneously, perserving the center of the image requires shifting
        # the output coordinates. the deviation is because out_length doesn't necesary equal in_length*scale.
        # to keep the center we need to subtract half of this deivation so that we get equal margins for boths sides and center is preserved.
        shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2

        # These are the matching positions of the output-coordinates on the input image coordinates.
        # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
        # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
        # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
        # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
        # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
        # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
        # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
        # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
        match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)

        # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
        left_boundary = np.floor(match_coordinates - kernel_width / 2)

        # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
        # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
        expanded_kernel_width = np.ceil(kernel_width) + 2

        # Determine a set of field_of_view for each each output position, these are the pixels in the input image
        # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
        # vertical dim is the pixels it 'sees' (kernel_size + 2)
        field_of_view = np.squeeze(
            np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

        # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
        # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
        # 'field_of_view')
        weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

        # Normalize weights to sum up to 1. be careful from dividing by 0
        sum_weights = np.sum(weights, axis=1)
        sum_weights[sum_weights == 0] = 1.0
        weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

        # We use this mirror structure as a trick for reflection padding at the boundaries
        mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

        # Get rid of  weights and pixel positions that are of zero weight
        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        weights = np.squeeze(weights[:, non_zero_out_pixels])
        field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

        # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
        return weights, field_of_view


# These next functions are all interpolation methods. x is the distance from the left pixel center


def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


class DenoiseDiffusion:
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)  # 移动到指定设备
        self.sigma2 = self.beta.to(device)  # 移动到指定设备
        self.n_steps = n_steps
        self.device = device

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_long = t.long().to(x0.device)  # 确保t_long与x0在同一个设备上
        mean = gather(self.alpha_bar.to(x0.device), t_long) ** 0.5 * x0  # 确保self.alpha_bar也在x0的设备上
        var = 1 - gather(self.alpha_bar.to(x0.device), t_long)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    '''
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        t_float = t.float()  # 确保t是浮点类型
        eps_theta = self.eps_model(xt, t_float)

        t_long = t.long()  # 确保t是整数类型
        alpha_bar = gather(self.alpha_bar, t_long)
        alpha = gather(self.alpha, t_long)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t_long)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps
    '''



    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eta=0.):
        """
        DDIM反向采样步骤
        :param xt: 当前的样本
        :param t: 当前的时间步
        :param eta: 方差缩放参数，0 <= eta <= 1. 0为确定性采样。
        :return: 生成的样本
        """
        t_float = t.float()  # 确保t是浮点类型
        eps_theta = self.eps_model(xt, t_float)

        t_long = t.long()  # 确保t是整数类型
        alpha_t = gather(self.alpha, t_long)
        alpha_bar_t = gather(self.alpha_bar, t_long)
        sigma_t = gather(self.sigma2, t_long).sqrt()

        # 使用DDIM方程进行采样
        mean = (1 / alpha_t.sqrt()) * (xt - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * eps_theta)

        # 在DDIM中，添加噪声是可选的。eta为0表示确定性过程。
        if eta > 0:
            noise = torch.randn_like(xt)
            mean = mean + eta * sigma_t * noise

        return mean

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        t = t.float()
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)

    def generate_image(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Generate an image by first adding noise to x0 and then
        denoising it through the reverse diffusion process.

        :param x0: The original input image tensor.
        :return: The generated image tensor.
        """
        # 选择一个随机时间步
        t = torch.randint(0, self.n_steps, (1,), device=self.device).item()
        t_tensor = torch.tensor([t], device=self.device)

        # 在时间步t添加噪声
        xt = self.q_sample(x0, t_tensor)

        # 逆向扩散过程
        for t in reversed(range(t)):
            t_tensor = torch.tensor([t], device=self.device)
            xt = self.p_sample(xt, t_tensor)

        return xt

    def generate_refined_image(self, x0: torch.Tensor, w: float, D: float) -> torch.Tensor:
        # 使用固定的时间步t
        t = self.n_steps
        t_tensor = torch.tensor([t], device=self.device)

        # 生成xN
        xN = self.q_sample(x0, t_tensor)
        xg = xN.clone()

        for t in reversed(range(1, self.n_steps + 1)):
            t_tensor = torch.tensor([t], device=self.device)

            # 预测逆向扩散
            xg_t_1 = self.p_sample(xg, t_tensor)

            # 计算x0_g
            alpha_bar_t = gather(self.alpha_bar, t_tensor)
            epsilon_theta = self.eps_model(xg, t_tensor)
            alpha_t = gather(self.alpha, t_tensor)
            x0_g = (1 / alpha_bar_t.sqrt()) * xg - ((1 - alpha_t) / alpha_bar_t.sqrt()) * epsilon_theta

            # 应用低通滤波
            #x0_filtered = self.low_pass_filter(x0, D)
            #x0_g_filtered = self.low_pass_filter(x0_g, D)

            # 计算梯度
            grad = self.calculate_gradient(x0, x0_g, xg, D)

            # 更新生成的图像
            xg = xg_t_1 - w * grad

        return xg

    def low_pass_filter(self, x: torch.Tensor, D: float) -> torch.Tensor:
        """
        实现低通滤波
        :param x: 输入图像张量
        :param D: 低通滤波的尺度因子
        :return: 经过低通滤波的图像张量
        """
        # 获取输入图像的尺寸
        in_shape = x.shape[2:]

        # 计算输出尺寸（使用D作为缩放因子）
        out_shape = np.floor(np.array(in_shape) / D).astype(int)

        # 创建Resizer实例，指定线性插值方法
        resizer = Resizer(in_shape=in_shape, scale_factor=1/D, output_shape=out_shape, kernel='linear')

        # 对图像进行降采样和上采样（低通滤波）
        x_down = resizer(x)
        x_up = resizer(x_down, output_shape=in_shape)

        return x_up

    def calculate_gradient(self, x0: torch.Tensor, x0_pred: torch.Tensor, xg: torch.Tensor, D: float) -> torch.Tensor:
        # 确保xg允许计算梯度
        xg.requires_grad_(True)

        # 对x0和x0_pred应用低通滤波
        x0_filtered = self.low_pass_filter(x0, D)
        x0_pred_filtered = self.low_pass_filter(x0_pred, D)

        # 计算过滤后的图像间的距离
        distance = torch.norm(x0_filtered - x0_pred_filtered, p=2)

        # 清除现有梯度
        if xg.grad is not None:
            xg.grad.zero_()

        # 计算梯度
        distance.backward()

        # 返回xg的梯度
        return xg.grad

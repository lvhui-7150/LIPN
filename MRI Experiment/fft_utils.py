import torch

def fft2c(x):
    """
    标准矩阵/MRI中心化2D正向傅里叶变换 (Matches fastMRI convention)
    输入图像 x 的中心是低频，输出 k 空间的中心也是低频
    """
    # 1. 先将图像中心的低频分量移到四周（符合标准FFT左上角为原点的定义）
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    # 2. 进行正交归一化的二维快速傅里叶变换
    x = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')
    # 3. 再将四周的低频分量移回中心，生成标准的中心化 k 空间
    return torch.fft.fftshift(x, dim=(-2, -1))

def ifft2c(x):
    """
    标准矩阵/MRI中心化2D逆向傅里叶变换 (Matches fastMRI convention)
    输入 k 空间的中心是低频，输出图像的中心也是低频
    """
    # 1. 先将 k 空间中心的低频分量移到四周
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    # 2. 进行正交归一化的逆二维快速傅里叶变换
    x = torch.fft.ifftn(x, dim=(-2, -1), norm='ortho')
    # 3. 再将四周的低频分量移回中心，恢复出中心化的图像
    return torch.fft.fftshift(x, dim=(-2, -1))
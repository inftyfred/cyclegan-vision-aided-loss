import pickle
import lpips
import torch
import numpy as np
from PIL import Image
import zipfile
import io
import torchvision.transforms as transforms

# ============ 1. 加载模型 ============
with open('training-runs2/teacher/00000-stylegan3-t-obama_data-gpus1-batch16-gamma2-clip-cv_loss_multilevel_sigmoid_s/network-snapshot-004600.pkl', 'rb') as f:
    data = pickle.load(f)
    G = data['G_ema'].cuda().eval()

# ============ 2. 加载 LPIPS 模型（推荐 VGG）============
lpips_model = lpips.LPIPS(net='vgg').cuda()

# ============ 3. 加载 ZIP 数据集 ============
dataset_path = 'datasets/obama_data.zip'
zf = zipfile.ZipFile(dataset_path, 'r')
image_files = [f for f in zf.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ============ 4. 图像预处理 ============
img_resolution = G.img_resolution  # 模型的图像分辨率

transform = transforms.Compose([
    transforms.Resize((img_resolution, img_resolution)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 转换到 [-1, 1]
])

# ============ 5. 计算 LPIPS ============
num_samples = 1000  # 计算样本数
lpips_scores = []

for i in range(num_samples):
    # 生成图像
    z = torch.randn(1, G.z_dim).cuda()
    c = torch.zeros(1, G.c_dim).cuda()

    with torch.no_grad():
        gen_img = G(z, c)  # 已经是 [-1, 1] 范围

    # 从数据集随机选择真实图像
    idx = np.random.randint(0, len(image_files))
    img_data = zf.read(image_files[idx])
    real_img = Image.open(io.BytesIO(img_data)).convert('RGB')
    real_img = transform(real_img).unsqueeze(0).cuda()

    # 如果图像大于 256，下采样以节省显存
    if img_resolution > 256:
        gen_img_resized = torch.nn.functional.interpolate(
            gen_img, size=(256, 256), mode='bilinear', align_corners=False
        )
        real_img_resized = torch.nn.functional.interpolate(
            real_img, size=(256, 256), mode='bilinear', align_corners=False
        )
        dist = lpips_model(gen_img_resized, real_img_resized)
    else:
        dist = lpips_model(gen_img, real_img)

    lpips_scores.append(dist.item())

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{num_samples} samples")

zf.close()

# ============ 6. 统计结果 ============
avg_lpips = np.mean(lpips_scores)
std_lpips = np.std(lpips_scores)
min_lpips = np.min(lpips_scores)
max_lpips = np.max(lpips_scores)

print(f"== = LPIPS Results == = ")
print(f"Average LPIPS: {avg_lpips:.4f}")
print(f"Std LPIPS: {std_lpips:.4f}")
print(f"Min LPIPS: {min_lpips:.4f}")
print(f"Max LPIPS: {max_lpips:.4f}")

# 修改您的代码中的循环部分：
lpips_scores = []

for i in range(num_samples // 2):  # 注意：除以2
# 生成两张不同的图像
    z1 = torch.randn(1, G.z_dim).cuda()
z2 = torch.randn(1, G.z_dim).cuda()
c = torch.zeros(1, G.c_dim).cuda()

with torch.no_grad():
    gen_img1 = G(z1, c)
gen_img2 = G(z2, c)

# 计算LPIPS（生成图像之间的感知距离）
dist = lpips_model(gen_img1, gen_img2)
lpips_scores.append(dist.item())

# 结果：LPIPS越大 = 多样性越好
print(f"Diversity (LPIPS): {np.mean(lpips_scores):.4f}")
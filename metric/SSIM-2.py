from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse


def cal_SSIM(x, y):
    """计算两个图像之间的SSIM"""
    # 注意：新版本skimage中multichannel参数已改为channel_axis
    # 根据skimage版本调整
    try:
        return ssim(x, y, win_size=3, channel_axis=2)  # 新版本
    except TypeError:
        return ssim(x, y, win_size=3, multichannel=True)  # 旧版本


def get_paired_images(dir_path):
    """获取同一文件夹中成对的real和fake图像路径"""
    # 获取所有_real和_fake文件
    real_files = sorted(glob(os.path.join(dir_path, "*_real.*")))
    fake_files = sorted(glob(os.path.join(dir_path, "*_fake.*")))
    
    # 构建配对字典：基于共同的前缀
    real_dict = {}
    fake_dict = {}
    
    # 处理real文件
    for real_path in real_files:
        filename = os.path.basename(real_path)
        # 提取前缀（去掉_real后缀）
        if "_real" in filename:
            prefix = filename.split("_real")[0]
            real_dict[prefix] = real_path
    
    # 处理fake文件
    for fake_path in fake_files:
        filename = os.path.basename(fake_path)
        # 提取前缀（去掉_fake后缀）
        if "_fake" in filename:
            prefix = filename.split("_fake")[0]
            fake_dict[prefix] = fake_path
    
    # 找出共同的前缀（配对成功）
    common_prefixes = sorted(set(real_dict.keys()) & set(fake_dict.keys()))
    
    # 按共同前缀配对
    real_paths = []
    fake_paths = []
    
    for prefix in common_prefixes:
        real_paths.append(real_dict[prefix])
        fake_paths.append(fake_dict[prefix])
    
    return real_paths, fake_paths


def cal_batch_SSIM(real_paths, fake_paths):
    """批量计算SSIM"""
    assert len(real_paths) == len(fake_paths)
    
    SSIMs = []
    ssim_info = []  # 存储SSIM值和对应的图像文件名
    
    for i in tqdm(range(len(real_paths))):
        img1 = Image.open(real_paths[i])
        img2 = Image.open(fake_paths[i])
        
        # 确保图像模式一致，如果是RGBA转换为RGB
        if img1.mode in ('RGBA', 'LA') or (img1.mode == 'P' and 'transparency' in img1.info):
            img1 = img1.convert('RGB')
        if img2.mode in ('RGBA', 'LA') or (img2.mode == 'P' and 'transparency' in img2.info):
            img2 = img2.convert('RGB')
        
        # 转换为numpy数组
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        
        # 确保图像尺寸相同
        if img1_np.shape != img2_np.shape:
            # 调整尺寸到较小的一方
            h = min(img1_np.shape[0], img2_np.shape[0])
            w = min(img1_np.shape[1], img2_np.shape[1])
            img1_np = img1_np[:h, :w]
            img2_np = img2_np[:h, :w]
        
        tmp_ssim = cal_SSIM(img1_np, img2_np)
        SSIMs.append(tmp_ssim)
        # 存储SSIM值和对应的图像文件名
        ssim_info.append({
            'ssim': tmp_ssim,
            'real_path': real_paths[i],
            'fake_path': fake_paths[i],
            'filename': os.path.basename(real_paths[i])
        })
    
    return SSIMs, ssim_info


def get_top_n_ssim_average(ssim_info, n=100):
    """获取前N个较大SSIM值的平均值"""
    # 按SSIM值从大到小排序
    sorted_ssim_info = sorted(ssim_info, key=lambda x: x['ssim'], reverse=True)
    
    # 取前N个（如果不足N个，则取全部）
    top_n = min(n, len(sorted_ssim_info))
    top_ssim_values = [item['ssim'] for item in sorted_ssim_info[:top_n]]
    
    # 计算平均值
    if top_n > 0:
        return np.mean(top_ssim_values), top_n, sorted_ssim_info[:top_n]
    else:
        return 0, 0, []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算同一文件夹中_real和_fake图像的SSIM')
    parser.add_argument('--dir_path', type=str, required=True, 
                       help='包含_real和_fake图像的文件夹路径')
    parser.add_argument('--real_suffix', type=str, default='_real',
                       help='真实图像的后缀（默认：_real）')
    parser.add_argument('--fake_suffix', type=str, default='_fake',
                       help='生成图像的后缀（默认：_fake）')
    parser.add_argument('--top_n', type=int, default=100,
                       help='取前N个较大SSIM值的平均值（默认：100）')
    
    opt = parser.parse_args()
    
    print(f"搜索路径: {opt.dir_path}")
    print(f"真实图像后缀: {opt.real_suffix}")
    print(f"生成图像后缀: {opt.fake_suffix}")
    print(f"取前 {opt.top_n} 个较大SSIM值的平均值")
    
    # 获取配对图像
    real_paths, fake_paths = get_paired_images(opt.dir_path)
    
    if len(real_paths) == 0 or len(fake_paths) == 0:
        print(f"警告: 未找到配对图像")
        print(f"找到的_real文件: {len(glob(os.path.join(opt.dir_path, f'*{opt.real_suffix}.*')))}")
        print(f"找到的_fake文件: {len(glob(os.path.join(opt.dir_path, f'*{opt.fake_suffix}.*')))}")
        exit(1)
    
    print(f"成功配对: {len(real_paths)} 对图像")
    print(f"真实图像示例: {real_paths[0] if real_paths else '无'}")
    print(f"生成图像示例: {fake_paths[0] if fake_paths else '无'}")
    
    # 计算SSIM
    SSIMs, ssim_info = cal_batch_SSIM(real_paths, fake_paths)
    
    # 计算前N个较大SSIM值的平均值
    top_n_avg, actual_n, top_ssim_info = get_top_n_ssim_average(ssim_info, opt.top_n)
    
    # 输出统计信息
    print("\n===== SSIM 统计结果 =====")
    print(f"总图像对数: {len(SSIMs)}")
    print(f"均值 SSIM: {np.mean(SSIMs):.6f}")
    print(f"标准差: {np.std(SSIMs):.6f}")
    print(f"最小值: {np.min(SSIMs):.6f}")
    print(f"最大值: {np.max(SSIMs):.6f}")
    
    print(f"\n===== 前 {actual_n} 个较大SSIM值的统计结果 =====")
    print(f"平均值: {top_n_avg:.6f}")
    print(f"SSIM范围: [{top_ssim_info[-1]['ssim']:.6f} - {top_ssim_info[0]['ssim']:.6f}]")
    
    # 可选：保存结果
    save_path = os.path.join(opt.dir_path, "ssim_results.txt")
    with open(save_path, 'w') as f:
        f.write("===== SSIM 计算结果 =====\n")
        f.write(f"图像总数: {len(SSIMs)}\n")
        f.write(f"均值 SSIM: {np.mean(SSIMs):.6f}\n")
        f.write(f"标准差: {np.std(SSIMs):.6f}\n")
        f.write(f"最小值: {np.min(SSIMs):.6f}\n")
        f.write(f"最大值: {np.max(SSIMs):.6f}\n\n")
        
        f.write(f"===== 前 {actual_n} 个较大SSIM值的统计结果 =====\n")
        f.write(f"平均值: {top_n_avg:.6f}\n")
        f.write(f"SSIM范围: [{top_ssim_info[-1]['ssim']:.6f} - {top_ssim_info[0]['ssim']:.6f}]\n\n")
        
        f.write("各图像SSIM值 (从高到低排序):\n")
        for i, item in enumerate(ssim_info):
            f.write(f"{item['filename']}: {item['ssim']:.6f}\n")
        
        f.write(f"\n前 {actual_n} 个较大SSIM值的图像:\n")
        for i, item in enumerate(top_ssim_info):
            f.write(f"{i+1}. {item['filename']}: {item['ssim']:.6f}\n")
    
    print(f"\n详细结果已保存至: {save_path}")
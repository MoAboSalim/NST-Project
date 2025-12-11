# Evaluation_metrics.py
import torch
import lpips
from PIL import Image
from torchvision import transforms
from math import log10
from skimage.metrics import structural_similarity as ssim

def load_image(path):
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    return tf(img).unsqueeze(0)

def PSNR(img1, img2):
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 20 * log10(1.0 / torch.sqrt(mse))

def SSIM(img1, img2):
    img1_np = img1.squeeze(0).permute(1,2,0).numpy()
    img2_np = img2.squeeze(0).permute(1,2,0).numpy()
    return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)

def LPIPS(img1, img2):
    loss_fn = lpips.LPIPS(net='alex')
    return loss_fn(img1, img2).item()

def evaluate_images(content_path, stylized_path):
    img1 = load_image(content_path)
    img2 = load_image(stylized_path)
    psnr_val = PSNR(img1, img2)
    ssim_val = SSIM(img1, img2)
    lpips_val = LPIPS(img1, img2)
    print("Evaluation Results:")
    print(f"PSNR  : {psnr_val:.4f}")
    print(f"SSIM  : {ssim_val:.4f}")
    print(f"LPIPS : {lpips_val:.4f}  (lower is better)")
    return psnr_val, ssim_val, lpips_val

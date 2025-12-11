# eval_clean.py
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# ---------------- utils ----------------
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    return (feat - mean.expand(size)) / std.expand(size)

# ---------------- Color Preservation Logic ----------------
# هذه الدوال تقوم بتحويل نظام الألوان لفصل الإضاءة (Y) عن الألوان (UV)
def rgb2yuv(image):
    # image: batch x 3 x H x W
    y = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
    u = -0.147 * image[:, 0, :, :] - 0.289 * image[:, 1, :, :] + 0.436 * image[:, 2, :, :]
    v = 0.615 * image[:, 0, :, :] - 0.515 * image[:, 1, :, :] - 0.100 * image[:, 2, :, :]
    return torch.stack([y, u, v], dim=1)

def yuv2rgb(image):
    # image: batch x 3 x H x W
    y, u, v = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    return torch.stack([r, g, b], dim=1)

# ---------------- models ----------------
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, 3),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, 3),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, 3),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, 3),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, 1),
    nn.ReflectionPad2d(1),
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, ceil_mode=True),
    nn.ReflectionPad2d(1),
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(128, 128, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, ceil_mode=True),
    nn.ReflectionPad2d(1),
    nn.Conv2d(128, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(256, 256, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, ceil_mode=True),
    nn.ReflectionPad2d(1),
    nn.Conv2d(256, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2, ceil_mode=True),
    nn.ReflectionPad2d(1),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
    nn.ReflectionPad2d(1),
    nn.Conv2d(512, 512, 3),
    nn.ReLU(),
)

class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, 1)
        self.g = nn.Conv2d(in_planes, in_planes, 1)
        self.h = nn.Conv2d(in_planes, in_planes, 1)
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, 1)
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, 3)
    def forward(self, c4, s4, c5, s5):
        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(c4, s4) + self.upsample5_1(self.sanet5_1(c5, s5))
        ))

def make_encoders(vgg_net, device):
    enc_1 = nn.Sequential(*list(vgg_net.children())[:4]).to(device)
    enc_2 = nn.Sequential(*list(vgg_net.children())[4:11]).to(device)
    enc_3 = nn.Sequential(*list(vgg_net.children())[11:18]).to(device)
    enc_4 = nn.Sequential(*list(vgg_net.children())[18:31]).to(device)
    enc_5 = nn.Sequential(*list(vgg_net.children())[31:44]).to(device)
    return enc_1, enc_2, enc_3, enc_4, enc_5

# ---------------- Runner Logic ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Transform(in_planes=512).to(device)
decoder = decoder.to(device)
vgg = vgg.to(device)

transform.eval()
decoder.eval()
vgg.eval()

# Helper: Transform image to tensor with dynamic resizing
def to_tensor(image_path, size=512):
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((size, size)), # Resize to user-selected dimension
        transforms.ToTensor()
    ])
    return tf(img).unsqueeze(0)

# Main Inference Function
def run_sanet(content_path, style_path, output_path,
              alpha=1.0, preserve_color=False, image_size=512,
              vgg_path='vgg_normalised.pth',
              decoder_path='decoder.pth',
              transform_path='transformer.pth'):
    
    # Check for weights
    if not os.path.exists(vgg_path) or not os.path.exists(decoder_path) or not os.path.exists(transform_path):
        raise FileNotFoundError(f"Missing weights. Please check paths.")

    # Load weights (Cached)
    vgg.load_state_dict(torch.load(vgg_path, map_location=device))
    transform.load_state_dict(torch.load(transform_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    enc_1, enc_2, enc_3, enc_4, enc_5 = make_encoders(vgg, device)

    # 1. Resize inputs based on user selection
    content = to_tensor(content_path, size=image_size).to(device)
    style = to_tensor(style_path, size=image_size).to(device)

    with torch.no_grad():
        # Encode
        C4 = enc_4(enc_3(enc_2(enc_1(content))))
        C5 = enc_5(C4)
        S4 = enc_4(enc_3(enc_2(enc_1(style))))
        S5 = enc_5(S4)

        # Style Transfer
        out = decoder(transform(C4, S4, C5, S5)).clamp(0, 1)

    # 2. Alpha Blending (Style Strength Control)
    if alpha < 1.0:
        out = out * alpha + content * (1 - alpha)

    # 3. Color Preservation (Advanced Logic)
    if preserve_color:
        # Convert both to YUV space
        content_yuv = rgb2yuv(content)
        stylized_yuv = rgb2yuv(out)
        
        # Combine: 
        # Take Y (Luminance/Details) from the Stylized Image
        # Take U, V (Color) from the Original Content Image
        combined_yuv = torch.stack([stylized_yuv[:, 0], content_yuv[:, 1], content_yuv[:, 2]], dim=1)
        
        # Convert back to RGB
        out = yuv2rgb(combined_yuv).clamp(0, 1)

    # Save output
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    save_image(out.cpu(), output_path)
    return output_path


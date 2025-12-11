# NST-Project
Arbitrary Style Transfer with Style-Attentional Networks

🖼️ SANet – Neural Style Transfer for Self-Attention
High-Quality Artistic Style Transfer Using Self-Attention Networks

This version of the project performs high-quality Neural Style Transfer using Self-Attention Networks (SANet) with support for:

Object Control (Alpha Mix Control)

Preservation of Original Image Colors (Color Preservation)

Image Size Selection During Inference

Accelerated Training Using Mixed Precision (AMP)

TensorBoard Support

VGG Model Configured to Work with SANet

✔️ Self-Attention Style Transfer (SANet)

✔️ Adaptive Instance Normalization (AdaIN)

✔️ Color Preservation (YUV-based logic)

✔️ Fast GPU Inference

✔️ Supports Large Images

✔️ Training + Evaluation scripts

✔️ Clean, optimized PyTorch implementation

📁 Project Structure
project/
│── eval_clean.py          # Inference / style transfer on images
│── train.py               # Training SANet using content & style datasets
│── vgg_normalised.pth     # Pretrained VGG weights
│── decoder.pth            # Decoder trained weights
│── transformer.pth        # Transform/SANet weights
│── datasets/
│     ├── content/
│     └── style/
└── output/

The project supports Style Transfer via a Flask API, allowing you to send the Content image and Style image via HTTP request and receive the resulting image ready for quality evaluation.

⚙️ How It Works

The core file for running the API is:

app.py

It contains 3 main operations:

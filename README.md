# NST Project
Random Pattern Transfer Using Attentional Pattern Networks

🌟 Project Overview

Name: NST Project

Focus: Random Pattern Transfer Using Attentional Pattern Networks (SANet).

Goal: To implement high-quality neural pattern transfer.

✨ Supported Features
The project supports several advanced features:

Object control (Alpha blending control)

Preservation of original image colors (Color preservation)

Image sizing during inference

Accelerated training using mixed resolution (AMP)

TensorBoard support

VGG model configured for SANet

Self-attentional pattern rendering (SANet)

Adaptive state normalization (AdaIN)

Color preservation (YUV-based logic)

Fast GPU inference

Training and evaluation programs

Clean and optimized PyTorch implementation

📁 Project Structure and Files
The README file contains a list of several important files and folders:

Project Structure:

eval_clean.py: Image pattern rendering/rendering

train.py: SANet training using content and pattern datasets

vgg_normalised.pth: Pre-trained VGG Weights

decoder.pth: Trained weights for the decoder

transformer.pth: Transformer/SANet weights

content/: Content images folder

style/: Style images folder

output/: Output images folder

💾 Datasets Used
For Content Images: COCO Train 2014

For Style Images: Painter By Numbers

🚀 How it Works (API)

Functions: The project supports transferring styles via the Flask API.

Usage: Allows sending a content image and a style image via an HTTP request and receiving the resulting image.

Basic API file: app.py

👥 Team Contributions
This project was a collaborative effort. The table below outlines the primary responsibilities and key contributions of each team member:

Fatima,Data Preparation & Processing,Preparing datasets and handling their conversion into tensors for training.

Yustina,Network Architecture Definition,"Defining all network components: VGG Encoder, SANet, Transformer, and Decoder."

Mohamed,Training & Core Logic,"Network training process, calculating loss functions, and managing checkpoint saving."

Jamal,Inference & Output Handling,"Executing the network on single or multiple images (inference), saving the final output, and handling color/alpha blending."

Ahmed,Evaluation Metrics,Implementation and calculation of the Evaluation Metrics.

Nour ,(Frontend):Built the interactive Web UI to visualize style transfer results and metrics in real-time.

Zeyad , (Backend): Developed the Flask API to handle image processing and server-side model integration.



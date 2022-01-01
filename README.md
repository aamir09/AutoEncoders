# AutoEncoders
This repository contains codes for AutoEncoders and their applications implemented using Convolutional Neural Networks in Python.

## Simple Denoiser Results
A simple autoencoder based denoiser was implemented on the MNIST data with 60k training samples. The models and weights are also available in the repository for reference and perfect recreation of the reconstruction results. Random noise was added from the uniform distribution, mean and std as 10 and 20 respectively.

### Loss Plots
<img src="https://user-images.githubusercontent.com/62461730/147847096-84f7d984-1cef-4561-bfcf-274370798b24.jpeg" alt="Simple Denoiser Losss" width="500" height="500">


### Reconstruction
<img src="https://user-images.githubusercontent.com/62461730/147847237-4ed0f287-bde9-43d8-85f4-923f075f980f.png" alt="Simple Denoiser Reconstruction" width="600" height="300">

## Sparese Denoiser Results
The idea behind sparse autoencoder is to make the latent space sparse and we do it by adding a penalty in the form of L1 norm in and only in the latent space(CNN layer).

### Loss Plots
<img src="https://user-images.githubusercontent.com/62461730/147847278-1dd2cd5e-1da4-454d-97a4-35051949035d.jpeg" alt="Sparse Denoiser Losss" width="500" height="500">


### Reconstruction


<img src="https://user-images.githubusercontent.com/62461730/147847294-255da273-db47-4f80-ad4e-1046e64e6bdb.png" alt="Sparse Denoiser Reconstructions" width="600" height="300">

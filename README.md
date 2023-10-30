# apt.github.io

# Assessing Neural Network Robustness via Adversarial Pivotal Tuning (WACV'2024)


This is the repository that contains source code for the [APT website](https://captaine.github.io/apt/).



> Peter Ebert Christensen<sup>1</sup>, Vésteinn Snæbjarnarson<sup>1</sup>, Andrea Dittadi<sup>2</sup>, Serge Belongie<sup>1</sup>, Sagie Benaim<sup>3</sup>
> <sup>1</sup>University of Copenhagen, <sup>2</sup>Helmholtz AI, <sup>3</sup>Hebrew University
>
> The robustness of image classifiers is essential to their
deployment in the real world. The ability to assess this re-
silience to manipulations or deviations from the training
data is thus crucial. These modifications have tradition-
ally consisted of minimal changes that still manage to fool
classifiers, and modern approaches are increasingly robust
to them. Semantic manipulations that modify elements of
an image in meaningful ways have thus gained traction for
this purpose. However, they have primarily been limited to
style, color, or attribute changes. While expressive, these
manipulations do not make use of the full capabilities of a
pretrained generative model. In this work, we aim to bridge
this gap. We show how a pretrained image generator can
be used to semantically manipulate images in a detailed,
diverse, and photorealistic way while still preserving the
class of the original image. Inspired by recent GAN-based
image inversion methods, we propose a method called Ad-
versarial Pivotal Tuning (APT). Given an image, APT first
finds a pivot latent space input that reconstructs the image
using a pretrained generator. It then adjusts the genera-
tor’s weights to create small yet semantic manipulations in
order to fool a pretrained classifier. APT preserves the full
expressive editing capabilities of the generative model. We
demonstrate that APT is capable of a wide range of class-
preserving semantic image manipulations that fool a variety
of pretrained classifiers. Finally, we show that classifiers
that are robust to other benchmarks are not robust to APT
manipulations and suggest a method to improve them.
> 
<a href="https://arxiv.org/abs/2211.09782"><img src="https://img.shields.io/badge/arXiv-2211.09782-b31b1b.svg" height=30.5></a> <a href="https://captaine.github.io/apt/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=30.5></a>

## Requirements ##
- 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
- CUDA toolkit 11.1 or later.
- GCC 7 or later compilers. The recommended GCC version depends on your CUDA version; see for example, CUDA 11.4 system requirements.
- If you run into problems when setting up the custom CUDA kernels, we refer to the [Troubleshooting docs](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary) of the original StyleGAN3 repo and the following issues: https://github.com/autonomousvision/stylegan_xl/issues/23.
- Windows user struggling installing the env might find https://github.com/autonomousvision/stylegan_xl/issues/10
  helpful.
- Use the following commands with Miniconda3 to create and activate your PG Python environment:
  - ```conda env create -f environment.yml```
  - ```conda activate APT```

# Running APT

We modify the [PTI](https://github.com/autonomousvision/stylegan_xl/blob/main/run_inversion.py) inversion script from StyleGAN-XL into APT (see APT.py)

You will need to download the weights of a classifer. For your refence we used the PRIME-Resnet50 [weights](https://zenodo.org/record/5801872#.YcSPahPP08M)

Similarly you will need to get the [imagenet dataset](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)


## Generating Fooling Examples ##
To generate fooling examples for all classes in imagenet, run
```
python3 APT.py --outdir="" --target "" --inv-steps 1000 --run-pti --pti-steps 350  --startidx 0 --endidx 1000 --device cuda --network="https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
```

To generate fooling examples using another classifier you can pass the classifier object to 
python3 APT.py --perceptor=perceptor_object

alternatively you can check out our notebook in the notebook folder to see how you can pass your custom classifier to the function.


# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Pretrained Models ##

Pretrained StyleGAN-XL models can be found here (pass the url as `PATH_TO_NETWORK_PKL`):

|Dataset| Res | FID | PATH
 :---  |  ---:  |  ---:  | :---
ImageNet| 16<sup>2</sup>   |0.73|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet16.pkl`</sub><br>
ImageNet| 32<sup>2</sup>   |1.11|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet32.pkl`</sub><br>
ImageNet| 64<sup>2</sup>   |1.52|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet64.pkl`</sub><br>
ImageNet| 128<sup>2</sup>  |1.77|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet128.pkl`</sub><br>
ImageNet| 256<sup>2</sup>  |2.26|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl`</sub><br>
ImageNet| 512<sup>2</sup>  |2.42|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet512.pkl`</sub><br>
ImageNet| 1024<sup>2</sup> |2.51|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet1024.pkl`</sub><br>
CIFAR10 | 32<sup>2</sup>   |1.85|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl`</sub><br>
FFHQ    | 256<sup>2</sup>  |2.19|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq256.pkl`</sub><br>
FFHQ    | 512<sup>2</sup>  |2.23|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq512.pkl`</sub><br>
FFHQ    | 1024<sup>2</sup> |2.02|  <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq1024.pkl`</sub><br>
Pokemon | 256<sup>2</sup>  |23.97| <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon256.pkl`</sub><br>
Pokemon | 512<sup>2</sup>  |23.82| <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon512.pkl`</sub><br>
Pokemon | 1024<sup>2</sup> |25.47| <sub>`https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl`</sub><br>


## Citation
If you find Adversarial Pivotal Tuning useful for your work please cite:
```
@misc{christensen2022apt,
  doi = {10.48550/ARXIV.2211.09782},
  
  url = {https://arxiv.org/abs/2211.09782},
  
  author = {Christensen, Peter Ebert and Snæbjarnarson, Vésteinn and Dittadi, Andrea and Belongie, Serge and Benaim, Sagie},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Cryptography and Security (cs.CR), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Assessing Neural Network Robustness via Adversarial Pivotal Tuning},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

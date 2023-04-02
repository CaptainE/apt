# apt.github.io

# Adversarial Pivotal Tuning (APT)

This is the repository that contains source code for the [APT website](https://captaine.github.io/apt/).

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

# Running APT
To run apt fist install the [dependancies](https://github.com/autonomousvision/stylegan_xl/blob/main/environment.yml) 

We modify the [PTI](https://github.com/autonomousvision/stylegan_xl/blob/main/run_inversion.py) inversion script from StyleGAN-XL into APT (see APT.py)

You will need to download the weights of a classifer. For your refence we used the PRIME-Resnet50 [weights](https://zenodo.org/record/5801872#.YcSPahPP08M)

Similarly you will need to get the [imagenet dataset](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

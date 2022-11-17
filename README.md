# Hyperspectral-Image-Classification-with-IFormer-Network-Feature-Extraction

##Abstract:
Convolutional neural networks (CNNs) are widely used for hyperspectral image (HSI) classification due to their better ability to model the local details of HSI. However, CNNs tends to ignore the global information of HSI, and thus lack the ability to establish remote dependencies, which leads to computational cost consumption and remains challenging. To address this problem, we propose an end-to-end Inception Transformer network (IFormer) that can efficiently generate rich feature maps from HSI data and extract high- and low-frequency information from the feature maps. First, spectral features are extracted using batch normalization (BN) and 1D-CNN, while the Ghost Module generates more feature maps via low-cost operations to fully exploit the intrinsic information in HSI features, thus improving the computational speed. Second, the feature maps are transferred to Inception Transformer through a channel splitting mechanism, which effectively learns the combined features of high- and low-frequency information in the feature maps and allows for the flexible modeling of discriminative information scattered in different frequency ranges. Finally, the HSI features are classified via pooling and linear layers. The IFormer algorithm is compared with other mainstream algorithms in experiments on four publicly available hyperspectral datasets, and the results demonstrate that the proposed method algorithm is significantly competitive among the HSI classification algorithms.

If you are interested in our work, please quote:

@article{ren2022hyperspectral,
  title={Hyperspectral Image Classification with IFormer Network Feature Extraction},
  author={Ren, Qi and Tu, Bing and Liao, Sha and Chen, Siyuan},
  journal={Remote Sensing},
  volume={14},
  number={19},
  pages={4866},
  year={2022},
  publisher={MDPI}
}

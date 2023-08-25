# RemovalNet

This repository is the official implementation of paper: **"RemovalNet: DNN model fingerprinting removal attack"**.
The RemovalNet is a min-max bi-level optimization-based DNN fingerprint removal attack that is used to *delete fingerprint-specific knowledge* while *maintaining general semantic knowledge* of the target model.


## Overview

![Overview of REMOVALNET against DNN ownership verification. The removal process is conducted on the latent-level and logits-level to alter the behavior patterns in the latent representation and decision boundary, respectively.](https://raw.githubusercontent.com/grasses/RemovalNet/master/figure/fig1_framework.png)


<div style="display: block; margin-left: auto; margin-right: auto; width: 60%;">
    <img src="https://raw.githubusercontent.com/grasses/RemovalNet/master/figure/exp_vis_attention.png" alt="Examples of saliency maps for Layer#2 of the victim model and surrogate model generated by LayerCAM [46].">
</div>


<div style="display: block; margin-left: auto; margin-right: auto; width: 60%;">
<img src="https://raw.githubusercontent.com/grasses/RemovalNet/master/figure/exp_vis_decision.png" alt="Decision boundary changes of the surrogate model on CIFAR10.">
</div>


# Citation
```text
@article{yao2023removalnet,
  title={RemovalNet: DNN Fingerprint Removal Attacks},
  author={Yao, Hongwei Li, Zheng and KunZhe, Huang and Lou, Jian and Qin, Zhan and Ren, Kui},
  journal={arXiv preprint arXiv:2308.12319},
  year={2023}
}
```

# License
This library is under the MIT license. For the full copyright and license information, please view the LICENSE file that was distributed with this source code.
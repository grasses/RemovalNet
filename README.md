# RemovalNet

This repository is the official implementation of paper: **["RemovalNet: DNN model fingerprinting removal attack"](https://ieeexplore.ieee.org/abstract/document/10251039/)**.
The RemovalNet is a min-max bi-level optimization-based DNN fingerprint removal attack that is used to *delete fingerprint-specific knowledge* while *maintaining general semantic knowledge* of the target model.


![Overview of REMOVALNET against DNN ownership verification. The removal process is conducted on the latent-level and logits-level to alter the behavior patterns in the latent representation and decision boundary, respectively.](https://raw.githubusercontent.com/grasses/RemovalNet/master/figure/fig1_framework.png)
Overview of REMOVALNET against DNN ownership verification. 
The removal process is conducted on the latent-level and logits-level 
    to alter the behavior patterns in the latent representation and decision boundary, respectively.

<br>

![](https://raw.githubusercontent.com/grasses/RemovalNet/master/figure/exp_vis_attention.png)
Examples of saliency maps for Layer#2 of the victim model and surrogate model generated by LayerCAM [46].

<br>

![](https://raw.githubusercontent.com/grasses/RemovalNet/master/figure/exp_vis_decision.png)
Decision boundary changes of the surrogate model on CIFAR10.

<hr>

# Running!
Example command for model "train(vgg19_bn,CIFAR10)-":
```shell
python -m attack.RemovalNet.removalnet -model1 "train(vgg19_bn,CIFAR10)-" -subset CIFAR10 -subset_ratio 1.0 -layer 2 -batch_size 128 -device 0 -tag ''
```

> Please note, put target model in path: model/ckpt/train(vgg19_bn,CIFAR10)-/final_ckpt_s1000.pth !!

The pretrained target model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1rRZDmPUPrSLjCgqwsn9rJsKmLxGZFMWK?usp=drive_link)


# Citation
```text
@article{yao2023removalnet,
  title={RemovalNet: DNN Fingerprint Removal Attacks},
  author={Yao, Hongwei and Li, Zheng and Huang, Kunzhe and Lou, Jian and Qin, Zhan and Ren, Kui},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2023},
  publisher={IEEE}
}
```

# License
This library is under the MIT license. For the full copyright and license information, please view the LICENSE file that was distributed with this source code.
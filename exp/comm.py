task_dict = {
    "CIFAR10": [
        "train(resnet50,CIFAR10)-removalnet(CIFAR10,1.0,0.11,1.0,10,l3)-",
        "train(densenet121,CIFAR10)-removalnet(CIFAR10,1.0,0.1,0.5,20,l2)-",
        "train(vgg16_bn,CIFAR10)-removalnet(CIFAR10,1.0,0.1,0.5,20,l3)-",
        "train(vgg19_bn,CIFAR10)-removalnet(CIFAR10,1.0,0.1,1.0,20,l3)-",
        "train(mobilenet_v2,CIFAR10)-removalnet(CIFAR10,1.0,0.05,1.0,10,l3)-"
    ],
    "CINIC10": [
        "train(resnet50,CINIC10)-removalnet(CINIC10,1.0,0.1,2.0,20,l2)-",
        #"train(resnet50,CINIC10)-removalnet(CINIC10,1.0,0.2,2.0,20,l2)-",
        "train(densenet121,CINIC10)-removalnet(CINIC10,1.0,0.2,1.0,20,l3)-",
        "train(vgg16_bn,CINIC10)-removalnet(CINIC10,1.0,0.2,1.0,20,l3)-",
        "train(vgg19_bn,CINIC10)-removalnet(CINIC10,1.0,0.2,1.0,20,l3)-",
        "train(mobilenet_v2,CINIC10)-removalnet(CINIC10,1.0,0.2,1.0,20,l3)-"
    ],
    "CelebA32+20": [
        #"train(resnet50,CelebA32+20)-removalnet(CelebA32+20,1.0,0.5,1.0,20,l3)-",
        "train(resnet50,CelebA32+20)-removalnet(CelebA32+20,1.0,0.2,2.0,20,l2)-",
        "train(densenet121,CelebA32+20)-removalnet(CelebA32+20,1.0,0.2,1.0,20,l3)-",
        "train(vgg16_bn,CelebA32+20)-removalnet(CelebA32+20,1.0,0.2,2.0,20,l2)-",
        "train(vgg19_bn,CelebA32+20)-removalnet(CelebA32+20,1.0,0.2,2.0,20,l2)-",
        "train(mobilenet_v2,CelebA32+20)-removalnet(CelebA32+20,1.0,0.2,1.0,20,l3)-"
    ],
    "CelebA32+31": [
        "train(resnet50,CelebA32+31)-removalnet(CelebA32+31,1.0,0.2,2.0,20,l2)-",
        "train(densenet121,CelebA32+31)-removalnet(CelebA32+31,1.0,0.2,1.0,20,l3)-",
        "train(vgg16_bn,CelebA32+31)-removalnet(CelebA32+31,1.0,0.2,2.0,20,l2)-",
        "train(vgg19_bn,CelebA32+31)-removalnet(CelebA32+31,1.0,0.2,2.0,20,l2)-",
        "train(mobilenet_v2,CelebA32+31)-removalnet(CelebA32+31,1.0,0.2,1.0,20,l3)-"
    ],
    "ImageNet": []
}

params_dict = {
    "CIFAR10": {
        "resnet50": [0.1],
        "densenet121": [0.1],
        "vgg16_bn": [0.1],
        "vgg19_bn": [0.1],
        "mobilenet_v2": [0.1]
    },
    "CINIC10": {
        "resnet50": [0.1],
        "densenet121": [0.1],
        "vgg16_bn": [0.1],
        "vgg19_bn": [0.1],
        "mobilenet_v2": [0.1]
    },
    "CelebA32+20": {
        "resnet50": [0.1],
        "densenet121": [0.1],
        "vgg16_bn": [0.1],
        "vgg19_bn": [0.1],
        "mobilenet_v2": [0.1]
    },
    "CelebA32+31": {
        "resnet50": [0.1],
        "densenet121": [0.1],
        "vgg16_bn": [0.1],
        "vgg19_bn": [0.1],
        "mobilenet_v2": [0.1]
    },
}
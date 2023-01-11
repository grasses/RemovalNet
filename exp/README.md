
# 规范说明

**1. exp11 文件存储规范**

结果说明：1组数据集存储1个pt文件，包含这个防御算法的所有攻击

结果文件：*output/{防御算法名}/exp/exp11_{数据集名}_{模型名}_{防御算法参数}.pt*

结果格式：
```
{
  "params": {"key": "value"},  % 防御方法参数
  "result": {
    "$model.task": []
  } 
}
```

## RemovalNet模型选择

### exp2.1 accuracy vs dist

CIFAR10: 
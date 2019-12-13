# 实验记录

| 模型               | 方法                                                         | 分数  | 本地模型效果                           |
| ------------------ | ------------------------------------------------------------ | ----- | -------------------------------------- |
| vgg19              | tageted cost + non tageted cost，一比一，迭代10次            | 0.082 | 基本识别错误，但没有识别到taget类      |
| vgg19              | tageted cost，迭代20次                                       | 0.079 | 大部分能识别到taget类                  |
| resnet50           |                                                              |       | 没加噪声的原图识别率极低，暂时不明原因 |
| vgg19+inception_v3 | tageted cost + non tageted cost，一比一，先vgg19迭代20次，再inceptionV3迭代20次 | 0.101 | 基本识别错误，但没有识别到taget类      |
|inception_v3 | 先用IFGSM迭代10次，然后ILA迭代10次                                     | 0.156| 迁移到resnet18上测试，分类准确率30%最左右|
|resnet18 | 先用IFGSM迭代10次，然后ILA迭代10次                                         | 0.100|迁移到InceptionV3上测试，分类准确率到80%|
|Densenet121         | 同上                                                          |0.1339|迁移到InceptionV3上测试，分类准确率76%，而仅使用IFGSM的方法，分类准确率为93%|
|Alexnet             |同上                                                           | 0.1102 |迁移到InceptionV3上测试，分类准确率91%，而仅使用IFGSM的方法，分类准确率为93%|
|SqueezeNet1.0 L3   | 同上                                                            |0.1200|迁移到InceptionV3上测试，分类准确率74%，而仅使用IFGSM的方法，分类准确率为90%|
|inception_v3, inception_v4, inception_resnet_v2, resnet_V2, prob=0.7, momentum=1.0, num_iter=16 | **[Translation-Invariant-Attacks](https://github.com/dongyp13/Translation-Invariant-Attacks)**, untargeted |0.49||
|inception_v3, inception_v4, inception_resnet_v2, resnet_V2,  prob=0.7, momentum=1.0, num_iter=16 | **[Translation-Invariant-Attacks](https://github.com/dongyp13/Translation-Invariant-Attacks)**, targeted |0.44||
|inception_v3, inception_v4, inception_resnet_v2, resnet_V2, prob=0.5, momentum=1.0, num_iter=50 | **[Translation-Invariant-Attacks](https://github.com/dongyp13/Translation-Invariant-Attacks)**, untargeted |0.66||
|adv_inception_v3, ens_adv_inception_resnet_v2, nception_v3, inception_v4, inception_resnet_v2, resnet_V2, prob=0.7, momentum=1.0, num_iter=10 | **[Translation-Invariant-Attacks](https://github.com/dongyp13/Translation-Invariant-Attacks)**, targeted |0.48||
|adv_inception_v3, ens_adv_inception_resnet_v2, nception_v3, inception_v4, inception_resnet_v2, resnet_V2, prob=0.7, momentum=1.0, num_iter=50 | **[Translation-Invariant-Attacks](https://github.com/dongyp13/Translation-Invariant-Attacks)**, targeted |0.51||
|adv_inception_v3, ens3_adv_inception_v3, ens3_adv_inception_v3, ens_adv_inception_resnet_v2, inception_v4, adv_inception_resnet_v2, resnet_V2, prob=0.0, momentum=1.0, num_iter=50 | **[Translation-Invariant-Attacks](https://github.com/dongyp13/Translation-Invariant-Attacks)**, untargeted |1.02||


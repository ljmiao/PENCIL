# PENCIL
Third-party implement of paper ['Probabilistic End-to-end Noise Correction for Learning with Noisy Labels'](https://arxiv.org/abs/1903.07788)

Attention! I find that using the hyper parameters in the paper can not get good result, so I try some other parameters and the result is not bad.

# Requirements
pytorch >= 1.0

# hyper parameters
![](https://github.com/ljmiao/PENCIL/raw/master/hyper_parameters.jpg)

# experiment result:
![](https://github.com/ljmiao/PENCIL/raw/master/symmetric_noise_result.jpg)

![](https://github.com/ljmiao/PENCIL/raw/master/Asymmetric_Noise_result.jpg)

# Run
for example:
1. python backbone_train.py --lr 0.03 
2. python pencil_train.py --alpha 0.01 --beta 0.1 --lamda 500 --lr 0.03 --percent 0.3
3. python fine_tune.py --lr 0.1 --percent 0.3

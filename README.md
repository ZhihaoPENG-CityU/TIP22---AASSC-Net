# AASSC-Net
Adaptive Attribute and Structure Subspace Clustering Network

URL_arXiv: https://arxiv.org/abs/2109.13742

URL_IEEE: https://doi.org/10.1109/TIP.2022.3171421

We have added remarks in the code, where the specific details can correspond to the explanation in the paper. A better clustering performance can be obtained by fine-tuning the hyperparameters. For example, the hyperparameters of UMIST is: [reg1: 1000 reg2: 0.001 reg3: 0.1], where regs 1-3 denotes \lambda_1, \lambda_2, and \beta respectively in this paper.

We appreciate it if you use this code and cite our paper, which can be cited as follows,
> @article{peng2021adaptive, <br>
>   title={Adaptive Attribute and Structure Subspace Clustering Network}, <br>
>   author={Peng, Zhihao and Liu, Hui and Jia, Yuheng and Hou, Junhui},  <br>
>   journal={IEEE Transactions on Image Processing},  <br>
>   year={2021}, <br>
>   doi={10.1109/TIP.2022.3171421} <br>
> } <br>

# Environment
+ Tensorflow [2.8.0]
+ Python [3.7.7]

# FAQ
+ How to solve the error [ModuleNotFoundError: No module named 'tensorflow.contrib']?
  +   As the contrib module doesn't exist in TF2.0, it is advised to use "tf.compat.v1.keras.initializers.he_normal()" as the initializer.
+ How to solve the issue that [TensorFlow 1.x migrated to 2.x]?
  +   It is advised to use the "tf.compat.v1.XXX" for code compatibility processing.
+ How to solve the error [RuntimeError: tf.placeholder() is not compatible with eager execution]?
  +   It is advised to use the "tf.compat.v1.disable_eager_execution()".

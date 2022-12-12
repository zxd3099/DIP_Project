# $DIP$ 大作业

## 1.资料获取

- 计算机视觉论文：http://www.cvpapers.com/
- 历年$CVPR/ICCV/ECCV$论文：https://openaccess.thecvf.com
- 查论文与代码：https://paperswithcode.com/
- https://arxiv.org/
- 历年论文汇总：https://github.com/52CV/CV-Surveys

## 2.图像增强(Image Enhancement)

### 2.1 Toward Fast, Flexible, and Robust Low-Light Image Enhancement

论文地址：https://arxiv.org/abs/2204.10137

论文解读：https://zhuanlan.zhihu.com/p/505001373

> Low-light image enhancement aims at **making information hidden in the dark visible to improve image quality**, it has drawn much attention in multiple emerging computer vision areas recently. 

#### 2.1.1 Model-based Methods

受限于定义的正则化，它们大多生成不令人满意的结果，并且需要针对现实场景手动调整许多参数。

#### 2.1.2 Network-based Methods

$KinD$：通过引入一些训练损失和调整网络架构来改善 $RetinexNet$ 中出现的问题。

$DeepUPE$: 定义了一个用于增强低光输入的光照估计网络。提出了一个递归带网络，并通过半监督策略对其进行训练。

$EnGAN$: 设计了一个生成器，注意在不成对的监督下进行增强。

$SSIENet$: 建立了一个分解型架构来同时估计光照和反射率。

然而，它们并不稳定，而且很难实现始终如一的卓越性能，特别是在未知的现实场景中，不清楚的细节和不适当的暴露是普遍存在的。

#### 2.1.3 This Paper's Contribution

- 开发了一个具有权重共享的**自校准照明学习模块**，以协商每个阶段的结果之间的收敛，提高**曝光稳定性**，并大幅度**减少计算负担**。据我们所知，这是第一个利用学习过程来加速微光图像增强算法的工作
- 定义了**无监督训练损失**来约束各阶段在自校准模块作用下的输出，赋予对不同场景的适应能力。属性分析表明，SCI具有**操作不敏感的适应性和模型无关的通用性**，这是现有研究中未发现的。
- 进行了大量的实验，以说明我们相对于其他先进方法的优越性。并在**黑脸检测和夜间语义分割**方面进行了应用，显示了本文的实用价值。简而言之，SCI在基于网络的微光图像增强领域重新定义了**视觉质量、计算效率和下游任务性能**的峰值点。

#### 2.1.4 Illumination Learning with Weight Sharing

提出了一种学习照明量的新模型：
$$
F(x^t):\begin{cases}u^t=H_{\theta}(x^t),x^0=y\\x^{t+1}=x^t+u^t\end{cases}
$$
公式中$y$是low-light observation，$z$是desired clear image, $x$是illumination component.

$u^t和x^t$分别是第$k$个阶段的residual term和illumination，在每一个阶段，权重$\theta$和架构$H$是不变的。

算法$SCI$的流程：

![](https://picd.zhimg.com/v2-00c11b1a8cafaed5284465afbd57f1ef_1440w.jpg?source=172ae18b)

需要注意的是，H采用了了权重共享机制，即在每个阶段使用相同的架构H和权重。

照明U和微光观测X在大多数地区是相似的或存在的线性连接。与采用弱光观测和照明之间的直接映射（现有工作中常用的模式相比，学习残差表示大大降低了计算难度（即公式的第二行的来源），同时也能保证性能、提高稳定性，尤其是曝光控制。

具有多个权重共享块的级联机制不可避免地会增加可预见的推理成本。**理想的情况是第一个块可以输出所需的结果，从而满足任务需求。同时，后一个块输出与第一个块相似甚至完全相同的结果。这样，在测试阶段，我们只需要一个块来加快推理速度。**

#### 2.1.5 Self-Calibrated Module(自校准模块)

在这里，我们的目标是 **定义一个模块，使每个阶段的结果收敛到同一个状态**。我们知道，每个阶段的输入源于前一阶段，第一阶段的输入明确定义为低光观测。一个直观的想法是，我们是否可以将每个阶段（第一阶段除外）的输入与弱光观测（即第一阶段的输入）联系起来，从而间接探索每个阶段之间的收敛行为。为此，我们引入了一个自校准地图$s$，并将其添加到弱光观测中，以显示每个阶段和第一阶段输入之间的差异。具体而言，自校准模块可以表示为
$$
G(x^t):\begin{cases}z^t=y\oslash x^t\\s^t=K_{\theta}(z^t)\\v^t=y+s^t\end{cases}
$$
当$t\ge1,v^t$是每个阶段的转换输入，$K_{\theta}$是引入的参数化运算符，其中的$\theta$是学习率。所以第$k$阶段

#### 2.1.6 Unsupervised Training Loss

考虑到现有配对数据的不精确性，我们采用无监督学习来扩大网络容量。我们将总损耗定义为：
$$
L_{total}=\alpha L_f+\beta L_s
$$
其中$L_f$和$L_s$分别表示保真度和平滑损失。$\alpha$和$\beta$是正平衡参数。

保真度损失是为了保证**估计照度**和**每个阶段输入之间的像素级**一致性，公式如下：
$$
L_f=\sum_{t=1}^T||x^t-(y+s^{t-1})||^2
$$
其中$T$是总共的阶段数量。这个函数使用重定义的输入$y+s^{t-1}$来限制 illumination output  $x^t$

实际上，该函数使用重新定义的输入来约束输出照明，而不是手工制作的GT或普通的微光输入。

照明的平滑特性在这项任务中是一个广泛的共识。在这里，我们采用了一个具有**空间变化$l1$范数**的平滑项，表示为：
$$
L_s=\sum_{i=1}^N\sum_{j\in N(i)}w_{i,j}|x_i^t-x_j^t|
$$
其中，$N$是总像素数量，$i$是第$i$个像素值，$N(i)$代表$5\times5$窗口的$i$的毗邻值。
$$
w_{i,j}=exp(-\frac{\sum_c((y_{i,c}+s_{i,c}^{t-1})-(y_{i,c}+s_{j,c}^{t-1}))^2}{2\sigma^2})
$$
其中，$c$是$YUV$颜色空间的通道，$\sigma=0.1$

本质上，自校准模块在学习更好的基本块（本工作中的光照估计块）时起辅助作用，该基本块通过权重共享机制级联生成整体光照学习过程。更重要的是，自校准模块使每个阶段的结果趋于一致，但在现有工作中尚未对其进行探索。此外，SCI的核心思想实际上是引入额外的网络模块来辅助培训，而不是测试。它改进了模型表征，实现了仅使用单个块进行测试。也就是说，机制“重量分担+任务相关自校准模块”可以转移到处理其他加速任务。

#### 2.1.7 Operation-Insensitive 

一般来说，基于网络的方法中使用的操作应该是固定的，不能随意更改，但我们提出的算法在不同且简单的$H$设置下表现出惊人的适应性。方法在不同的设置（块$3\times3$卷积+$ReLU$的数量）中获得了稳定的性能。

重新审视我们设计的框架，这个属性可以被获得，因为**SCI不仅转换了照明共识（即剩余学习），还集成了物理原理（即元素分割操作）。**

#### 2.1.8 Dark Face Detection

我们利用著名的人脸检测算法$S3FD$来评估暗人脸检测性能。
请注意，$S3FD$是使用原始$S3FD$中显示的更宽面部数据集进行训练的，我们使用$S3FD$的预训练模型来**微调**通过各种方法增强的图像。
**同时，我们执行了一种名为$SCI+$的新方法，该方法将我们的$SCI$作为一个基本模块嵌入到$S3FD$前端，以便在任务损失和增强相结合的情况下进行联合训练**。



### 2.2 URetinex-Net: Retinex-based Deep Unfolding Network for Low-light Image Enhancement

论文代码：https://github.com/AndersonYong/URetinex-Net

> we propose a Retinex-based deep unfolding network (URetinex-Net), which unfolds an optimization problem into a learnable network to decompose a low-light image into   reflectance and illumination layers.

#### 2.2.1 Contributions

- Based on traditional model-based methods, we propose a novel deep unfolding network for LLIE(URetinex-Net), **consisting of three functionally clear modules corresponding to initialization, optimization, and illumination adjustment, respectively**, which inherits the flexibility and interpretability from model-based methods.
- The optimization module in our proposed URetinex-Net **unfolds optimization procedure into a deep network**, which leverages the powerful model ability of learning-based methods to adaptively fit data dependent priors.
- Extensive experiments on real-world datasets are conducted to demonstrate high efficiency and superiority of our URetinex-Net, which can **realize noise suppression and details preservation for the final enhanced results.**

#### 2.2.2 Framework

![](https://github.com/AndersonYong/URetinex-Net/raw/main/figure/framework.png)

Model-based LLIE methods are highly interpretable and flexible, while learning-based LLIE methods show superiority in learning complicated mapping in a data-driven manner. In addition, deep neural networks perform fast during inference, which is particularly computationally efficient. 

The reflectance and illumination can be obtained by minimizing the following regularized energy function:
$$
E(R,L)=||I-R·L||^2_F+\alpha\Phi(R)+\beta\Psi(L)
$$
where $||·||_F$ denotes Frobenius norm, $||I-R·L||^2_F$ is the fidelity term.

The problem is rewritten as:
$$
min_{P,Q,R,L}=||I-P·Q||^2_F+\alpha\Phi(R)+\beta\Psi(L)+\gamma||P-R||^2_F+\lambda||Q-F||^2_F
$$
The loss function for initializing two components:
$$
min_{R_0,L_0}||I-R_0·L_0||_1+\mu||L_0-max_{c\in\{R,G,B\}}I^{(c)}||_F^2
$$
the loss function for decomposing a normal-light image:
$$

$$

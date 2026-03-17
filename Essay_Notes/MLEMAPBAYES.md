
# Inferring pattern-driving intercellular flows from single-cell and spatial transcriptomics

Almet, A.A., Tsai, YC., Watanabe, M. _et al._ Inferring pattern-driving intercellular flows from single-cell and spatial transcriptomics. _Nat Methods_ **21**, 1806–1817 (2024). https://doi.org/10.1038/s41592-024-02380-w[^0]

[^0]: Almet, A.A., Tsai, YC., Watanabe, M. _et al._ Inferring pattern-driving intercellular flows from single-cell and spatial transcriptomics. _Nat Methods_ **21**, 1806–1817 (2024). https://doi.org/10.1038/s41592-024-02380-w

今天来介绍一下聂老师组的8月分出版的这篇推断细胞间流动网络的文章《从单细胞和空间转录组学推断模式驱动的细胞间流动》，这篇文章在Github上也上传了相关python包（Flowsig）可以下载使用。

# 方法适用数据

Flowsig在scRNA-seq和ST数据中都可以使用，不过对于scRNA-seq因为缺少空间信息，在学习因果网络的时候需要有扰动组数组（可以是疾病组，不同时间等）

# 基本假设

如下图所示，本方法基于如下生物学过程：
认为细胞上的受体收到配体信号后，信号影响细胞中的基因表达模块（GEMs），使得一些转录因子的生物作用发生改变，表现为影响下流的配体产生。
\
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/697f0e0b94184df18a2d484c99771379.png#pic_center)

# 结果概述：
## Fig. 1 | Description of the FlowSig model
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/369b95b6824645b9a0cddd430d87c15d.webp#pic_center)
\
**Fig.1a** 本图就是是上面所讲的生物学过程假设。
**Fig.1b** 基于scRNA-seq数据的流程图：
	（1）首先根据scRNA-seq数据利用pyLIGER构建GEMs，具体的矩阵分解内容可以参考这篇文章(Townes, F.W., Engelhardt, B.E. Nonnegative spatial factorization applied to spatial genomics. _Nat Methods_ **20**, 229–238 (2023). https://doi.org/10.1038/s41592-022-01687-w), 或者我之前的文章（[矩阵分解](https://blog.csdn.net/weixin_63470844/article/details/142991849?fromshare=blogdetail&sharetype=blogdetail&sharerId=142991849&sharerefer=PC&sharesource=weixin_63470844&sharefrom=from_link)）
	（2）随后利用Wilcoxon秩和检验来识别控制组和扰动组的差异信号。
	（3）根据条件独立测试和条件不变测试以及图学习模型得到了流入信号， GEMs，流出信号之间的因果网络。
		流入信号的定义：根据现有库，把每个受体和它所对应的调控因子作为先验，计算 $R_1 \times TF_1$ ，因为一个受体可能对应$m$个配体，所以$TF_1=\frac{TF_1^{(1)}+TF_1^{(2)}+...+TF_1^{(m)}}{m}$
		流出信号的定义：直接配体的表达量作为流出信号
**Fig.1c** 基于ST数据的流程图：
	1）首先根据ST数据利用非负空间分解（NSF）构建GEMs，具体的矩阵分解的内容可以参考讲解Fig1 b中提到的文章。
	（2）检测空间差异信号
	（3）根据条件独立测试和条件不变测试以及图学习模型得到了流入信号， GEMs，流出信号之间的因果网络。
		流入信号的定义：利用之前聂老师组发的COMMOT [^1] 的方法，可以直接计算出每个细胞接受的信号。
		流出信号的定义：直接配体的表达量作为流出信号

[^1]: Cang, Z., Zhao, Y., Almet, A.A. _et al._ Screening cell–cell communication in spatial transcriptomics via collective optimal transport. _Nat Methods_ **20**, 218–228 (2023). https://doi.org/10.1038/s41592-022-01728-4


## Fig. 2 | Synthetic validation of FlowSig.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43160c6e4e314899bce996461c369a00.webp#pic_center)

\
**Fig.2a-c** 介绍了三种细胞间信号流动模式，分别是 a.单一信号流入流出 b.信号分流模式 c.竞争性信号流
**Fig.2d-f** 比较了三种不同的信号流动模式，浅蓝色框表示使用总受体表达（游离加结合受体）作为流入变量的情况，而深蓝色框表示使用结合受体表达作为流入变量的情况。从图中可以看出来考虑扰动组和使用结合受体表达量作为流入信号变量的真阳率（TPR）和真阴率（TNR）会更高


## Fig. 3 | Experimental validation of FlowSig using a cortical organoid model.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a7261158e9fe44adbf00dc5f9ed433e2.webp#pic_center)
\
这张图是在皮质类器官模型中使用FlowSig的实验结果：
**Fig.3a-b** 利用Wilcoxon秩和检验，把不同天数当成控制组和扰动组，识别出来的差异性的流入信号和流出信号。
**Fig.3c** 对于得到的基因模组，观察了在不同时间点每个基因模组的表达量差异，从中识别出差异性的基因模组。
**Fig.3d-e** 根据FlowSig构建出的细胞间流动网络。其中信号流入选择的是FGF通路和BMP通路，根据Cellchat可以知道这个通路的具体受体基因。而GEMs中的TF是根据因子矩阵中从高到低对基因进行排序，从中选择的表达量最高的基因。
**Fig.3f-g** 实验证实了推断结果的正确性：分别添加FGF和BMP刺激后EOMES，PAX6和NR2F1的表达量都有显著变化。


## Fig. 4 | Application of FlowSig to perturbed non-spatial scRNA-seq of  pancreatic islets.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/45b73cb7b8774c51a1cd0de7721e466a.webp#pic_center)
\
这是在没有空间信息的胰岛数据scRNA-seq中的实验：
**Fig.4a** 和Fig.3c类似，得到在不同Cluster中的差异性GEMs。
**Fig.4b-c** 和Fig.3a-b类似，可以得到在控制组和扰动组（有IFN-$\gamma$ 刺激） 下的差异性流出信号和流出信号
**Fig.4d** 根据FlowSig构建的全局细胞间流网络，识别出了一些驱动因子（流入信号），基因模组（GEMs）和相应的转录因子（TFs），以及流出信号。
**Fig.4e-f** 分别根据差异性上调和下调信号绘制了细胞间流动图。


## Fig. 5 | Application of FlowSig to scRNA-seq of human BALF sampled from people with moderate or severe COVID-19.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5c91dc70a346496caefb3302a4c697a6.webp#pic_center)

这是在没有空间信息的COVID-19数据scRNA-seq中的实验。
**Fig.5a** 类似的找到在不同患病程度患者中的差异性GEMs
**Fig.5b** 不同细胞中的差异性GEMs
**Fig.5c** 不同患病程度患者的基因分布，可以找到差异性基因
**Fig.5d-f** 分别构建了健康，中毒患COVID-19和重度患COVID-19患者的细胞间流动网络
**Fig.5g-f** 可以发现不同患病程度患者中共有和独有的流入信号以及GEMs
## Fig. 6 | Application of FlowSig to spatial Stereo-seq data of E9.5 mouse embryo.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b63f1c339eff4ccd9afb30bd8891548d.webp#pic_center)
\
对有空间信息的E9.5期小鼠胚胎应用FlowSig进行分析:
**Fig.6a** 小鼠空间GEMs（基于NSF--非负空间分解，这考虑了空间信息）的分布图
**Fig.6b** 参考Cang et al.[^2]之前的方法,利用有向网络回溯来推断哪些空间 GEM 连接到信号流出节点。从而可以构建Shh信号的上游GEMs以及流入信号（一些受体）
**Fig.6c** 在全局细胞间流动网络中找到Shh受体接收的信号作为流入信号的下游GEMs以及流出信号。
**Fig.6d** 找到Shh上游TFs中比较重要的转录因子（重要性是利用随机森林特征（Gini）重要性衡量的）
**Fig.6e** 横纵坐标是流入信号强度，颜色代表转录因子表达量，从图中可以看出不同转录因子随着信号流入的表达量变化。从而得到流入信号和转录因子间的相关性。
**Fig.6f-g** 和Fig。6d-e类似
**Fig.6h** 根据FlowSig推断绘制的Shh 和 Wnt5a 之间建议的激活剂-抑制剂反馈

[^2]: Cang, Z. et al. Screening cell–cell communication in spatial transcriptomics via collective optimal transport. Nat. Methods 20, 218–228 (2023).

# 总结
FlowSig为我们推断数据中的信号流动提供了有力的工具。本方法应用范围很广，可以作为下游的疾病分析和病理研究的重要手段之一。并且和COMMOT相结合，可以直接计算出细胞所接受的信号量，使得ST数据中的信号流动更加合理。也会继续关注和期待聂老师组后续的文章哒( •̀ ω •́ )y

萌新写文，如果错误请多多包涵，欢迎各位大佬指正讨论~

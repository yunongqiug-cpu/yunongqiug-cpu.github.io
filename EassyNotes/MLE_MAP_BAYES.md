# MLE、MAP 与贝叶斯估计：
## 从二范数、先验到后验分布

**Author: Y.Qiu**

## 引言

在统计学习与机器学习中，参数估计常见的三种思想分别是：

- \textbf{MLE（Maximum Likelihood Estimation，极大似然估计）}：只利用观测数据，通过最大化似然函数来估计参数；
- \textbf{MAP（Maximum A Posteriori，最大后验估计）}：在似然的基础上进一步引入参数的先验信息；
- \textbf{Bayes Estimation（贝叶斯估计）}：不只给出单个参数点，而是给出参数的完整后验分布，并根据具体损失函数导出不同的最优点估计。

这三者之间并不是彼此割裂的，而是构成了一个逐层推广的统一框架：
\[
\text{MLE} \quad \longrightarrow \quad \text{MAP} \quad \longrightarrow \quad \text{Bayesian Estimation}.
\]

本文主要针对以下问题进行了回答：

1. 为什么 MLE 在高斯噪声假设下等价于最小化二范数（MSE）；
2. 为什么这里\textbf{正态噪声假设是关键}；
3. 在正态先验与 Laplace 先验下，为什么 MAP 分别等价于
   \[
   \text{MSE} + L_2 \text{正则}, \qquad \text{MSE} + L_1 \text{正则} ;
   \]
4. 贝叶斯估计为什么能给出完整分布，而不仅仅是一个点；
5. 为什么在不同损失函数下，后验均值、后验众数、后验中位数分别成为最优点估计。

## MLE：极大似然估计

### 基本定义

设观测数据为
\[
\mathcal{D} = \{x_1,\dots,x_n\},
\]
模型参数为 $ \theta $ 。若已知数据在给定参数 $ \theta $ 下的概率模型为 $ p(\mathcal{D}\mid \theta) $ ，则极大似然估计定义为
\[
\hat{\theta}_{\mathrm{MLE}}
=
\arg\max_{\theta} p(\mathcal{D}\mid \theta).
\]
由于对数函数单调递增，上式等价于
\[
\hat{\theta}_{\mathrm{MLE}}
=
\arg\max_{\theta} \log p(\mathcal{D}\mid \theta)
=
\arg\min_{\theta} \big(-\log p(\mathcal{D}\mid \theta)\big).
\]
因此，\textbf{MLE 的本质就是最小化负对数似然（negative log-likelihood）。换句话说，MLE的本质是找到一组参数，使得在这组参数下，观测到的数据出现的概率最大}

### 线性回归模型中的 MLE

考虑经典线性回归模型
\[
\bm{y} = X\bm{\theta} + \bm{\epsilon},
\qquad
\bm{\epsilon} \sim \mathcal{N}(\bm{0},\sigma^2 I),
\]
其中：

- $ \bm{y}\in\mathbb{R}^n $ 为观测向量；
- $ X\in\mathbb{R}^{n\times d} $ 为设计矩阵；
- $ \bm{\theta}\in\mathbb{R}^d $ 为待估参数；
- $ \bm{\epsilon} $ 为高斯噪声。

由
\[
\bm{y} = X\bm{\theta} + \bm{\epsilon},
\qquad
\bm{\epsilon} \sim \mathcal{N}(\bm{0},\sigma^2 I),
\]
可知条件分布为
\[
\bm{y}\mid \bm{\theta}
\sim
\mathcal{N}(X\bm{\theta},\,\sigma^2 I).
\]
因此其概率密度函数为
\[
p(\bm{y}\mid \bm{\theta})
=
\frac{1}{(2\pi \sigma^2)^{n/2}}
\exp\left(
-\frac{1}{2\sigma^2}
(\bm{y}-X\bm{\theta})^T(\bm{y}-X\bm{\theta})
\right).
\]

取对数得
\[
\log p(\bm{y}\mid \bm{\theta})
=
-\frac{n}{2}\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}
(\bm{y}-X\bm{\theta})^T(\bm{y}-X\bm{\theta}).
\]

由于第一项与 $ \bm{\theta} $ 无关，因此最大化对数似然等价于最小化
\[
(\bm{y}-X\bm{\theta})^T(\bm{y}-X\bm{\theta})
=
\|\bm{y}-X\bm{\theta}\|_2^2.
\]

于是得到
\[
\hat{\bm{\theta}}_{\mathrm{MLE}}
=
\arg\min_{\bm{\theta}}
\|\bm{y}-X\bm{\theta}\|_2^2.
\]

### 为什么 MLE 等价于二范数？

这里必须强调：

> \textbf{MLE 等价于二范数,是因为噪声被假设为高斯分布，进而可以把由设计矩阵到观测向量的映射吃到高斯噪声里面。}

事实上，只要模型写成
\[
y_i = f_{\theta}(x_i) + \epsilon_i,
\qquad
\epsilon_i \sim \mathcal{N}(0,\sigma^2),
\]
无论 $ f_{\theta}(x) $ 是线性函数还是非线性函数，都有
\[
y_i \mid x_i,\theta \sim \mathcal{N}(f_{\theta}(x_i),\sigma^2),
\]
从而
\[
p(y_i\mid x_i,\theta)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(y_i-f_{\theta}(x_i))^2}{2\sigma^2}
\right).
\]
对独立样本求联合似然并取负对数，得到
\[
-\log p(\mathcal{D}\mid \theta)
=
\text{常数项}
+
\frac{1}{2\sigma^2}
\sum_{i=1}^n
(y_i-f_{\theta}(x_i))^2.
\]
因此
\[
\hat{\theta}_{\mathrm{MLE}}
=
\arg\min_{\theta}
\sum_{i=1}^n
(y_i-f_{\theta}(x_i))^2.
\]

所以二范数（或均方误差，MSE）对应的根源是：
\[
\boxed{\text{高斯噪声} \;\Longrightarrow\; \text{平方误差损失}}.
\]

### 为什么正态噪声如此重要？

正态噪声的重要性体现在以下三点：

1. \textbf{负对数似然恰好是平方误差}。正如上文的推导，正态噪声使得优化目标具有非常清晰的解析形式；
2. \textbf{数学处理方便}。高斯分布在线性变换、加和、条件分布等运算下封闭，推导非常整洁；
3. \textbf{统计学意义深刻}。许多独立微小随机扰动的叠加在中心极限定理下近似服从正态分布，因此高斯噪声在很多场景下是合理的一阶近似。

\textbf{但也要指出：若噪声并非高斯，则 MLE 通常不再对应平方误差。}例如：

- 若 $ \epsilon $ 服从 Laplace 分布，则对应 $ L_1 $ 损失；
- 若观测服从 Bernoulli 分布，则对应交叉熵损失；
- 若观测为计数且服从 Poisson 或 Negative Binomial，则对应 Poisson/NB 的负对数似然，而不是 MSE。

因此，\textbf{损失函数的选择本质上是概率模型选择的结果}。

## MAP：最大后验估计

### MAP 的定义

与 MLE 只利用似然不同，MAP 还引入参数先验 $ p(\theta) $ 。根据贝叶斯公式，
\[
p(\theta\mid \mathcal{D})
=
\frac{p(\mathcal{D}\mid \theta)p(\theta)}{p(\mathcal{D})}.
\]
因此最大后验估计定义为
\[
\hat{\theta}_{\mathrm{MAP}}
=
\arg\max_{\theta} p(\theta\mid \mathcal{D}).
\]

由于 $ p(\mathcal{D}) $ 与 $ \theta $ 无关，可写为
\[
\hat{\theta}_{\mathrm{MAP}}
=
\arg\max_{\theta} p(\mathcal{D}\mid \theta)p(\theta).
\]
再取对数：
\[
\hat{\theta}_{\mathrm{MAP}}
=
\arg\max_{\theta}
\big[
\log p(\mathcal{D}\mid \theta) + \log p(\theta)
\big].
\]
等价地，
\[
\hat{\theta}_{\mathrm{MAP}}
=
\arg\min_{\theta}
\big[
-\log p(\mathcal{D}\mid \theta) - \log p(\theta)
\big].
\]
于是得到 MAP 的损失函数：
\[
\boxed{
L_{\mathrm{MAP}}(\theta)
=
-\log p(\mathcal{D}\mid \theta)
-\log p(\theta)
}
\]
这说明：

> \textbf{MAP = 负对数似然 + 先验诱导的正则项}

### MAP 比 MLE 更稳健的原因

MLE 只依赖观测数据。当样本很少时，似然函数往往不稳定，参数估计容易出现高方差、过拟合等问题。

而 MAP 在优化中额外加入了先验项 $ -\log p(\theta) $ ，因此会把参数限制在一个先验认为更合理的区域。这意味着：

- \textbf{小样本时}：先验影响显著，能提高稳定性；
- \textbf{大样本时}：似然逐渐主导，MAP 往往趋近于 MLE。

## MAP 与正则化的完全推导

下面在同样的高斯观测模型下，分别考虑正态先验与 Laplace 先验。

### 观测模型：高斯噪声对应 MSE
\[
\boxed{
L_{\mathrm{MAP}}(\theta)
=
-\log p(\mathcal{D}\mid \theta)
-\log p(\theta)
}
\]
对于上式中的前半部分，实际上就是我们前面推导的MLE的部分，我们已经证明这部分在高斯噪声的假设下等价于二范数。所以这个MAP的损失函数可以看为：
\[
L_{\mathrm{MAP}}(\theta)=\hat{\theta}_{\mathrm{MLE}}-\log p(\theta)
\]
其中
\[
\hat{\bm{\theta}}_{\mathrm{MLE}}
=
\arg\min_{\bm{\theta}}
\|\bm{y}-X\bm{\theta}\|_2^2.
\]
为了方便，后面就都以上面的线性回归情况为例来说明。事实上对于下面的非线性情况本节结论也成立。
\[
\hat{\theta}_{\mathrm{MLE}}
=
\arg\min_{\theta}
\sum_{i=1}^n
(y_i-f_{\theta}(x_i))^2.
\]
下面我们将目光集中在后面关于参数的这一项上。

### 情形一：正态先验 $ \Longrightarrow $ $ L_2 $ 正则

假设参数先验为
\[
\bm{\theta} \sim \mathcal{N}(\bm{0},\tau^2 I).
\]
则先验密度为
\[
p(\bm{\theta})
=
\frac{1}{(2\pi\tau^2)^{d/2}}
\exp\left(
-\frac{1}{2\tau^2}
\bm{\theta}^T\bm{\theta}
\right).
\]
取负对数：
\[
-\log p(\bm{\theta})
=
\frac{d}{2}\log(2\pi\tau^2)
+
\frac{1}{2\tau^2}
\|\bm{\theta}\|_2^2.
\]
忽略常数项：
\[
-\log p(\bm{\theta})
\equiv
\frac{1}{2\tau^2}
\|\bm{\theta}\|_2^2.
\]
因此 MAP 目标函数为
\[
L_{\mathrm{MAP}}(\bm{\theta})
=
\frac{1}{2\sigma^2}
\|\bm{y}-X\bm{\theta}\|_2^2
+
\frac{1}{2\tau^2}
\|\bm{\theta}\|_2^2.
\]
两边同乘 $ 2\sigma^2 $ ，不改变最优解，得
\[
\hat{\bm{\theta}}_{\mathrm{MAP}}
=
\arg\min_{\bm{\theta}}
\left[
\|\bm{y}-X\bm{\theta}\|_2^2
+
\lambda \|\bm{\theta}\|_2^2
\right],
\]
其中
\[
\lambda = \frac{\sigma^2}{\tau^2}.
\]
这正是 Ridge 回归的形式。\\
因此有
\[
\boxed{
\text{Gaussian likelihood} + \text{Gaussian prior}
\Longrightarrow
\text{MSE} + L_2 \text{ regularization}
}
\]

### 情形二：Laplace 先验 $ \Longrightarrow $ $ L_1 $ 正则

现在假设各个参数分量独立，且
\[
p(\theta_j)
=
\frac{\lambda}{2}
\exp(-\lambda |\theta_j|),
\qquad j=1,\dots,d.
\]
则联合先验为
\[
p(\bm{\theta})
=
\prod_{j=1}^d \frac{\lambda}{2}\exp(-\lambda |\theta_j|)
=
\left(\frac{\lambda}{2}\right)^d
\exp\left(
-\lambda \sum_{j=1}^d |\theta_j|
\right).
\]
注意
\[
\sum_{j=1}^d |\theta_j| = \|\bm{\theta}\|_1.
\]
因此
\[
p(\bm{\theta})
=
\left(\frac{\lambda}{2}\right)^d
\exp(-\lambda \|\bm{\theta}\|_1).
\]
取负对数得
\[
-\log p(\bm{\theta})
=
-d\log\frac{\lambda}{2}
+
\lambda \|\bm{\theta}\|_1.
\]
忽略常数项：
\[
-\log p(\bm{\theta})
\equiv
\lambda \|\bm{\theta}\|_1.
\]

于是 MAP 目标函数变为
\[
L_{\mathrm{MAP}}(\bm{\theta})
=
\frac{1}{2\sigma^2}
\|\bm{y}-X\bm{\theta}\|_2^2
+
\lambda \|\bm{\theta}\|_1.
\]
乘以 $ 2\sigma^2 $ 后可重写为
\[
\hat{\bm{\theta}}_{\mathrm{MAP}}
=
\arg\min_{\bm{\theta}}
\left[
\|\bm{y}-X\bm{\theta}\|_2^2
+
\lambda' \|\bm{\theta}\|_1
\right],
\]
其中 $ \lambda' = 2\sigma^2\lambda $ 。

因此有
\[
\boxed{
\text{Gaussian likelihood} + \text{Laplace prior}
\Longrightarrow
\text{MSE} + L_1 \text{ regularization}
}
\]

### 小结：先验与正则化的一一对应

| \textbf{观测噪声} | \textbf{参数先验} | \textbf{MAP 对应损失} |
|---|---|---|
| Gaussian | Gaussian | MSE $ +\, L_2 $ 正则 |
| Gaussian | Laplace | MSE $ +\, L_1 $ 正则 |

因此可以说：
\[
\boxed{
\text{正则化项本质上就是先验分布的负对数}
}
\]

## Bayes：完整后验分布而不是单个点

### MLE 与 MAP 都是点估计

MLE 和 MAP 最终都给出一个参数值：
\[
\hat{\theta}_{\mathrm{MLE}}, \qquad \hat{\theta}_{\mathrm{MAP}}.
\]
因此它们属于\textbf{点估计（point estimation）}。

点估计的特点是：

- 输出简单；
- 便于优化与部署；
- 但无法直接反映不确定性。

例如，若只给出一个参数值 $ \hat{\theta}=2.5 $ ，我们并不知道这个估计是“非常确定”还是“其实可能在 $ [1.0,4.0] $ 内都合理”。

### 贝叶斯估计给出的是什么？

贝叶斯方法不满足于只求一个最优参数，而是通过贝叶斯公式求整个后验分布：
\[
p(\theta\mid \mathcal{D})
=
\frac{p(\mathcal{D}\mid \theta)p(\theta)}{p(\mathcal{D})}.
\]
这里：

- $ p(\theta) $ 表示在看到数据前，对参数的先验认识；
- $ p(\mathcal{D}\mid \theta) $ 表示数据在参数 $ \theta $ 下出现的可能性；
- $ p(\theta\mid \mathcal{D}) $ 表示看到数据后，对参数不确定性的更新。

因此，贝叶斯估计输出的是：
\[
\boxed{
\text{参数的整个后验分布 } p(\theta\mid \mathcal{D})
}
\]
而不仅仅是一个点。
\\
例如，若
\[
\theta\mid \mathcal{D} \sim \mathcal{N}(2.5,0.1^2),
\]
说明参数高度集中在 $ 2.5 $ 附近，不确定性较小；
而若
\[
\theta\mid \mathcal{D} \sim \mathcal{N}(2.5,2^2),
\]
则说明虽然均值同样是 $ 2.5 $ ，但不确定性远大得多。

### 为什么贝叶斯方法能“给出分布”？

这是因为在贝叶斯框架中，参数本身被看作随机变量。
先验 $ p(\theta) $ 表示参数在观测数据之前可能取哪些值；观测到数据后，用似然对其加权，得到后验：
\[
p(\theta\mid \mathcal{D})
\propto
p(\mathcal{D}\mid \theta)p(\theta).
\]
$ p(\theta) $ 称为先验权重，表示“在看数据之前，我有多相信这个参数值”。
\\ $ p(\mathcal{D}\mid \theta) $ 称为似然权重,表示“如果参数真的是这个值，那么当前数据出现的可能性有多大”。
\\两者相乘就得到“看完数据之后，这个参数值该有多可信”。也就是说，所有可能的参数值都被保留下来，只是其“可信程度”根据数据重新分配了权重。\\\\
需要进一步说明的是，后验分布 $ p(\theta|\mathcal D) $ 的计算来源于贝叶斯公式：
\[
p(\theta|\mathcal D)
=
\frac{p(\mathcal D|\theta)p(\theta)}{p(\mathcal D)},
\quad
p(\mathcal D)
=
\int p(\mathcal D|\theta)p(\theta)\,d\theta.
\]
其中分母 $ p(\mathcal D) $ 是对参数空间的积分：
\[
p(\mathcal D) = \int p(\mathcal D|\theta)p(\theta)\,d\theta,
\]
通常难以解析计算。因此，在大多数实际问题中，后验分布 $ p(\theta|\mathcal D) $ 往往无法得到闭式表达。

不过，在某些任务中并不需要显式计算该归一化常数。例如，在求解 MAP（最大后验估计）或进行基于比值的采样方法（如 MCMC）时，由于 $ p(\mathcal D) $ 与参数 $ \theta $ 无关，可以在优化或计算中忽略，从而仅使用未归一化形式
\[
p(\theta|\mathcal D) \propto p(\mathcal D|\theta)p(\theta).
\]

然而，在需要进行后验预测（详见5.5节）或计算期望时，例如
\[
p(y^\ast|x^\ast,\mathcal D)
=
\int p(y^\ast|x^\ast,\theta)p(\theta|\mathcal D)\,d\theta,
\]
必须使用归一化后的后验分布。否则，由于权重未正确归一化，会导致结果失去概率解释，并影响预测的准确性与不确定性刻画。

在一些特殊情形下（例如高斯似然与高斯先验的组合），后验分布仍属于同一分布族，从而可以得到解析解；而在更一般的情况下，则需要借助近似方法（如 MAP、Laplace 近似、变分推断或 MCMC 采样）来刻画后验分布。

### 后验分布如何表达不确定性？

一旦得到后验分布，就可以计算：

- \textbf{后验均值}：反映平均估计；
- \textbf{后验方差}：反映不确定性大小；
- \textbf{可信区间}（credible interval）：反映参数落在某一区间内的后验概率；
- \textbf{后验预测分布}：反映未来观测的不确定性。

### 后验预测分布

在实际问题中，我们往往关心新样本 $ y^\ast $ 的预测。贝叶斯方法不把参数固定成单一点，而是对所有参数值进行积分平均：
\[
p(y^\ast\mid x^\ast,\mathcal{D})
=
\int p(y^\ast\mid x^\ast,\theta)\,p(\theta\mid \mathcal{D})\,d\theta.
\]
这说明预测结果同时考虑了：

- 数据噪声；
- 参数不确定性。

这也是贝叶斯方法相比普通点估计更完整的地方。

## Bayes估计框架下不同损失函数对应不同点估计的数学推导

### 贝叶斯决策论框架

假设我们已经得到了参数的后验分布 $ p(\theta\mid \mathcal{D}) $ 。但需要强调的是，在贝叶斯框架中，后验分布 $ p(\theta|\mathcal D) $ 已经包含了关于参数的不确定性信息，因此理论上可以直接用于预测（通过后验预测分布）。然而，\textbf{在许多实际任务中（例如回归输出、分类决策或控制问题），仍然需要输出一个具体的数值或决策。}因此，需要在后验分布基础上进一步引入损失函数，并通过最小化后验期望损失来得到点估计。换言之，点估计并不是贝叶斯推断的目标，而是决策过程的结果。
如果现在想从中提取一个点估计 $ a$ ，则应该最小化该点估计在后验下的\textbf{后验期望损失}：
\[
a^\ast
=
\arg\min_{a}
\mathbb{E}_{\theta\mid \mathcal{D}}
\big[
L(a,\theta)
\big]
=
\arg\min_a
\int L(a,\theta)\,p(\theta\mid \mathcal{D})\,d\theta.
\]
不同的损失函数 $ L(a,\theta) $ 会给出不同的最优点估计。

### 平方损失对应后验均值

取平方损失
\[
L(a,\theta) = (a-\theta)^2.
\]
则目标函数为
\[
R(a)
=
\int (a-\theta)^2 p(\theta\mid \mathcal{D})\,d\theta.
\]
对 $ a$ 求导：
\[
\frac{dR(a)}{da}
=
\int 2(a-\theta)\,p(\theta\mid \mathcal{D})\,d\theta
=
2a\int p(\theta\mid \mathcal{D})\,d\theta
-
2\int \theta p(\theta\mid \mathcal{D})\,d\theta.
\]
由于
\[
\int p(\theta\mid \mathcal{D})\,d\theta = 1,
\]
所以
\[
\frac{dR(a)}{da}
=
2a - 2\mathbb{E}[\theta\mid \mathcal{D}].
\]
令导数为零，得到
\[
a^\ast = \mathbb{E}[\theta\mid \mathcal{D}].
\]
因此，
\[
\boxed{
\text{平方损失} \Longrightarrow \text{后验均值}
}
\]

### 绝对损失对应后验中位数

取绝对损失
\[
L(a,\theta) = |a-\theta|.
\]
则目标函数为
\[
R(a)
=
\int |a-\theta|\,p(\theta\mid \mathcal{D})\,d\theta.
\]
将积分拆为两部分：
\[
R(a)
=
\int_{-\infty}^{a} (a-\theta)\,p(\theta\mid \mathcal{D})\,d\theta
+
\int_{a}^{\infty} (\theta-a)\,p(\theta\mid \mathcal{D})\,d\theta.
\]
对 $ a$ 求导：
\[
\frac{dR(a)}{da}
=
\int_{-\infty}^{a} p(\theta\mid \mathcal{D})\,d\theta
-
\int_{a}^{\infty} p(\theta\mid \mathcal{D})\,d\theta.
\]
令导数为零，得
\[
\int_{-\infty}^{a^\ast} p(\theta\mid \mathcal{D})\,d\theta
=
\int_{a^\ast}^{\infty} p(\theta\mid \mathcal{D})\,d\theta
=
\frac{1}{2}.
\]
这正说明 $ a^\ast $ 是后验分布的中位数。因此
\[
\boxed{
\text{绝对损失} \Longrightarrow \text{后验中位数}
}
\]

### $ 0$ -$ 1$ 损失对应后验众数（MAP）

设损失函数为
\[
L(a,\theta)
=
\begin{cases}
0, & a=\theta,\\
1, & a\neq \theta.
\end{cases}
\]
则后验期望损失为
\[
R(a)
=
\int L(a,\theta)p(\theta\mid \mathcal{D})\,d\theta
=
1 - p(a\mid \mathcal{D})
\]
（在离散情形下严格成立；连续情形中对应“选取使后验密度最大的点”这一极限思想）。

因此最小化 $ R(a) $ 等价于最大化 $ p(a\mid \mathcal{D}) $ ，从而
\[
a^\ast
=
\arg\max_a p(a\mid \mathcal{D}).
\]
这就是后验众数，即 MAP：
\[
\boxed{
\text{ $ 0$ -$ 1$ 损失} \Longrightarrow \text{后验众数（MAP）}
}
\]

## 点估计与完整分布的对比

| \textbf{方法} | \textbf{输出} | \textbf{特点} |
|---|---|---|
| MLE | 一个点 $ \hat{\theta}_{\mathrm{MLE}} $ | 仅依赖数据；实现简单；不直接表达不确定性 |
| MAP | 一个点 $ \hat{\theta}_{\mathrm{MAP}} $ | 在 MLE 基础上结合先验；小样本更稳健；对应正则化 |
| Bayesian | 整个后验分布 $ p(\theta\mid \mathcal{D}) $ | 能表达参数不确定性；可构造可信区间与后验预测分布；更完整但计算更复杂 |

## 三者关系的统一总结

### 从优化角度看

- \textbf{MLE}
  \[
  \hat{\theta}_{\mathrm{MLE}}
  =
  \arg\min_{\theta}
  \big[-\log p(\mathcal{D}\mid \theta)\big].
  \]

- \textbf{MAP}
  \[
  \hat{\theta}_{\mathrm{MAP}}
  =
  \arg\min_{\theta}
  \big[-\log p(\mathcal{D}\mid \theta)-\log p(\theta)\big].
  \]

- \textbf{Bayesian}
  \[
  p(\theta\mid \mathcal{D})
  =
  \frac{p(\mathcal{D}\mid \theta)p(\theta)}{p(\mathcal{D})}.
  \]

### 从“损失函数来源”看

- 似然决定数据拟合项；
- 先验决定正则项；
- 后验结合二者，给出参数不确定性的完整分布。

### 从“最终输出”看

- MLE：输出一个由数据驱动的最优点；
- MAP：输出一个结合了先验约束的最优点；
- Bayes：输出一个完整分布，再根据具体任务与损失函数导出点估计。

## 结论

本文的核心结论可以概括为以下几点：

1. \textbf{MLE 等价于二范数（MSE）的根本原因不是线性，而是高斯噪声假设。}
   \[
   \text{Gaussian noise} \Longrightarrow -\log\text{likelihood} \propto \text{squared error}.
   \]

2. \textbf{MAP 本质上是在 MLE 上加入先验。}
   \[
   L_{\mathrm{MAP}}(\theta)= -\log p(\mathcal{D}\mid \theta)-\log p(\theta).
   \]
   其中\textbf{正态先验对应 $ L_2 $ 正则，Laplace 先验对应 $ L_1 $ 正则。}
   \[
   \text{Gaussian prior} \Longrightarrow L_2,
   \qquad
   \text{Laplace prior} \Longrightarrow L_1.
   \]

3. \textbf{贝叶斯估计并不只给一个点，而是给整个后验分布。}
   这使得我们能够量化不确定性，并进行后验预测。

4. \textbf{点估计并不是唯一的，而是取决于所采用的损失函数。}
   \[
   \text{平方损失} \Longrightarrow \text{后验均值},\qquad
   \text{绝对损失} \Longrightarrow \text{后验中位数},\qquad
   \text{ $ 0$ -$ 1$ 损失} \Longrightarrow \text{后验众数（MAP）}.
   \]

总之，MLE、MAP 与 Bayesian estimation 并不是互相竞争的三套体系，而是一个逐步扩展的统一概率学习框架：
\[
\boxed{
\text{MLE: 数据拟合}
\quad\subset\quad
\text{MAP: 数据拟合 + 先验约束}
\quad\subset\quad
\text{Bayes: 完整后验分布与决策}
}
\]

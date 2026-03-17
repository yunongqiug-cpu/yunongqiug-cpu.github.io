# MLE、MAP 与贝叶斯估计  
## 从二范数、先验到后验分布

**Author: Y.Qiu**

---

## 引言

在统计学习与机器学习中，参数估计常见的三种思想分别是：

- **MLE（Maximum Likelihood Estimation，极大似然估计）**：只利用观测数据，通过最大化似然函数来估计参数  
- **MAP（Maximum A Posteriori，最大后验估计）**：在似然的基础上进一步引入参数的先验信息  
- **Bayes Estimation（贝叶斯估计）**：不只给出单个参数点，而是给出参数的完整后验分布，并根据具体损失函数导出不同的最优点估计  

这三者之间并不是彼此割裂的，而是构成了一个逐层推广的统一框架：

$$
\text{MLE} \quad \longrightarrow \quad \text{MAP} \quad \longrightarrow \quad \text{Bayesian Estimation}
$$

---

## 本文回答的问题

1. 为什么 MLE 在高斯噪声假设下等价于最小化二范数（MSE）  
2. 为什么 **正态噪声假设是关键**  
3. 在正态先验与 Laplace 先验下，为什么 MAP 分别等价于  

$$
\text{MSE} + L_2 \text{正则}, \qquad \text{MSE} + L_1 \text{正则}
$$

4. 贝叶斯估计为什么能给出完整分布  
5. 为什么不同损失函数对应不同点估计  

---

# MLE：极大似然估计

## 基本定义

设观测数据为：

$$
\mathcal{D} = \{x_1,\dots,x_n\}
$$

模型参数为 $ \theta $，则：

$$
\hat{\theta}_{\mathrm{MLE}} =
\arg\max_{\theta} p(\mathcal{D}\mid \theta)
$$

等价于：

$$
\hat{\theta}_{\mathrm{MLE}} =
\arg\min_{\theta} \big(-\log p(\mathcal{D}\mid \theta)\big)
$$

👉 本质：

> **MLE = 最小化负对数似然**

---

## 线性回归中的 MLE

$$
\bm{y} = X\bm{\theta} + \bm{\epsilon}, \quad
\bm{\epsilon} \sim \mathcal{N}(0,\sigma^2 I)
$$

得到：

$$
\hat{\bm{\theta}} =
\arg\min_{\bm{\theta}}
\|\bm{y}-X\bm{\theta}\|_2^2
$$

---

## 为什么变成 MSE？

核心结论：

$$
\boxed{
\text{高斯噪声} \;\Longrightarrow\; \text{平方误差损失}
}
$$

---

## 正态噪声为什么重要？

1. 负对数似然 = MSE  
2. 数学闭合性好  
3. 中心极限定理支持  

但：

- Laplace → $ L_1 $
- Bernoulli → Cross-Entropy  
- Poisson → Poisson loss  

👉 **损失函数 = 概率模型**

---

# MAP：最大后验估计

## 定义

$$
p(\theta|\mathcal{D}) =
\frac{p(\mathcal{D}|\theta)p(\theta)}{p(\mathcal{D})}
$$

$$
\hat{\theta}_{MAP} =
\arg\min [-\log p(\mathcal{D}|\theta) - \log p(\theta)]
$$

---

## 本质

$$
\boxed{
\text{MAP} = \text{MLE} + \text{Regularization}
}
$$

---

## 正态先验 → L2

$$
\theta \sim \mathcal{N}(0,\tau^2)
$$

$$
\Rightarrow \|\theta\|_2^2
$$

---

## Laplace 先验 → L1

$$
p(\theta) \propto e^{-\lambda |\theta|}
$$

$$
\Rightarrow \|\theta\|_1
$$

---

## 对应关系

| 噪声 | 先验 | 损失 |
|------|------|------|
| Gaussian | Gaussian | MSE + L2 |
| Gaussian | Laplace | MSE + L1 |

---

# Bayes：完整分布

## 核心

$$
p(\theta|\mathcal{D})
$$

👉 不再是点，而是分布

---

## 后验预测

$$
p(y^*|x^*,\mathcal{D}) =
\int p(y^*|x^*,\theta)p(\theta|\mathcal{D})d\theta
$$

---

# 点估计 vs 贝叶斯

| 方法 | 输出 | 特点 |
|------|------|------|
| MLE | 点 | 无不确定性 |
| MAP | 点 | 有先验 |
| Bayes | 分布 | 表达不确定性 |

---

# 不同损失 → 不同估计

## 平方损失

$$
\Rightarrow \mathbb{E}[\theta|\mathcal{D}]
$$

👉 后验均值

---

## 绝对损失

👉 后验中位数

---

## 0-1 损失

👉 MAP（众数）

---

# 总结

$$
\boxed{
\text{MLE} \subset \text{MAP} \subset \text{Bayes}
}
$$

---

## 核心理解

- MLE：只看数据  
- MAP：数据 + 先验  
- Bayes：完整不确定性  

---

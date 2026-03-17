# MLE、MAP 与贝叶斯估计  
## 从二范数、先验到后验分布

**Author: Y.Qiu**

---

## 引言

在统计学习与机器学习中，参数估计常见的三种思想分别是：

- **MLE（Maximum Likelihood Estimation）**
- **MAP（Maximum A Posteriori）**
- **Bayesian Estimation（贝叶斯估计）**

统一框架：

$$
\text{MLE} \rightarrow \text{MAP} \rightarrow \text{Bayesian Estimation}
$$

---

## 1. MLE

### 定义

$$
\hat{\theta}_{MLE} = \arg\max p(\mathcal D | \theta)
$$

等价于：

$$
\arg\min -\log p(\mathcal D | \theta)
$$

---

### 高斯噪声 → MSE

$$
y = f_\theta(x) + \epsilon, \quad \epsilon \sim \mathcal N(0, \sigma^2)
$$

$$
\Rightarrow \text{MLE} = \arg\min \sum (y - f_\theta(x))^2
$$

$$
\boxed{\text{Gaussian noise} \Rightarrow \text{MSE}}
$$

---

## 2. MAP

$$
\hat{\theta}_{MAP} = \arg\max p(\mathcal D | \theta)p(\theta)
$$

$$
= \arg\min [-\log p(\mathcal D|\theta) - \log p(\theta)]
$$

---

### 正态先验 → L2

$$
\theta \sim \mathcal N(0, \tau^2 I)
$$

$$
\Rightarrow \|	heta\|_2^2
$$

---

### Laplace先验 → L1

$$
p(\theta) \propto e^{-\lambda \|\theta\|_1}
$$

---

## 3. Bayes

$$
p(\theta | \mathcal D) = \frac{p(\mathcal D|\theta)p(\theta)}{p(\mathcal D)}
$$

输出的是：

$$
\boxed{\text{posterior distribution}}
$$

---

## 4. 点估计 vs 分布

| 方法 | 输出 | 特点 |
|------|------|------|
| MLE | 点 | 无不确定性 |
| MAP | 点 | 带先验 |
| Bayes | 分布 | 可表达不确定性 |

---

## 5. 不同损失对应

- 平方损失 → 后验均值  
- 绝对损失 → 后验中位数  
- 0-1损失 → MAP  

---

## 总结

$$
\text{MLE} \subset \text{MAP} \subset \text{Bayes}
$$

# ML 专题 03：Bayesian Decision Theory（贝叶斯决策）与 Naive Bayes（朴素贝叶斯）详解

> 依据：课件 ML-Slides3（71 页）、week7 录音稿（整节课讲 Bayes、MLE、GMM 和三种朴素贝叶斯）、总复习 PPT Revision2026S1 p.2–6、week13 录音稿（老师逐页讲总复习）、四份考过这部分的期末卷参考答案。
> 约定：专业术语用英文，括号里附中文。例题数字都用程序核对过。图片在 `figs/nb/` 文件夹，每张图前面写了看哪里。公式用 LaTeX 书写。
> 决策树（Gini、entropy）在卷子里常和朴素贝叶斯放在同一题，但课件是单独一章（Slides6），放到下一个专题讲。

---

## 0. 这一章在考试里的分量（先看这个）

### 0.1 考频

六份卷子里有四份考了 Bayes 或朴素贝叶斯，共 66 分。

| 试卷 | 题号 | 考什么 | 分值 |
|---|---|---|---|
| 23-24 S1 | Q3(b) | **Gaussian NB**：给两类的均值和**散度矩阵**（样本数 600 和 400），分类一个样本 | 13 |
| 24-25 S1 | Q2(a) | **完整 Gaussian Bayes 决策规则**：给均值和协方差矩阵（$C_1 \ne C_2$），先验 500 : 300，写判别函数并分类 | 12 |
| 24-25 S1 | Q2(b) | 同一组数据改用 **Gaussian NB**，写出假设并分类 | 13 |
| 24-25 S2 | Q3(a) | **Bernoulli NB**：数据就是课件 p.58 的例子（Yes/No 换成 1/0） | 10 |
| 25-26 S1 | Q3(a) | **Bernoulli NB**：还是课件 p.58 的例子，换成文字标签 | 9 |
| 25-26 S1 | Q3(b) | **Multinomial NB**：文本分类，必须用 **Laplace smoothing（拉普拉斯平滑）** | 9 |

**规律**：

- 全是计算题，每次换一种变体：Gaussian NB → 完整 Bayes + Gaussian NB → Bernoulli → Bernoulli + Multinomial。
- **Bernoulli NB 连续两年考的都是课件原例**，答案都是 $4/45$ 对 $1/20$。课件例题必须会。
- 论述没有单独考过，但题目常要求 "state your assumptions"，这部分占分。
- 按"变体轮换"推测，下次可能是 Multinomial（换数据）、Gaussian NB，或者几种混在一起。GMM 在聚类部分也会出现。

### 0.2 老师在课上说过的话（week7 录音稿）

1. **只用先验分类没有意义**：班里 60% 男生、40% 女生，只看先验就会把所有人都判成男生，"40% of the samples will be classified wrongly"。必须加上新的证据 $x$，用 **posterior probability（后验概率）**决策。
2. **$p(x)$ 是公共因子**：分母 $p(x)$ 对每一类都一样，"we can ignore the p(x)"，直接比较 **joint probability（联合概率）** $p(x\mid\omega_j)P(\omega_j)$，或者它的对数。
3. **判别函数**是更一般的概念：每类一个 $g_j(x)$，取最大的那一类。在 Bayes 里，$g_j$ 可以是后验、联合概率或联合概率的对数。
4. **难点在类条件密度**："the estimation of the class-conditional density function is very challenging"，所以假设它是高斯分布，只需估计均值和协方差。
5. **MLE**：参数取使训练数据出现概率最大的值；取对数把连乘变成连加，"the natural logarithm and the exponential can be cancelled"。结果就是熟悉的样本均值和（除以 $n$ 的）协方差。
6. **Bayes 分类器的边界是曲线**：例子里的边界 "is a curve… so the classifier is a non-linear classifier"。
7. **GMM**：一类数据来自多个高斯时用混合模型；分量个数 $m$ 是 hyperparameter（超参数）。分量太多，有的 $\alpha$ 会很小，边界变复杂，"higher chance to overfit"；可以根据小的 $\alpha$ 减少分量。
8. **朴素贝叶斯的动机**（week13 又讲了一遍）：完整协方差矩阵的参数太多，"if the number of samples is not many, this estimation is not accurate"。假设特征相互独立后，多元高斯变成一元高斯的乘积。
9. **三种朴素贝叶斯**：连续特征用 Gaussian NB；二值特征用 Bernoulli NB；文本词频用 Multinomial NB。
10. **Bernoulli NB 是重点**："This is very important. **You must know this, very important.**" week13 又说 "another very important part"，并演示了怎么从计数表估计先验和条件概率（就是总复习 p.6 那张表）。
11. **Laplace smoothing**：某个词在某类里没出现过，概率是 0，乘起来整个结果就是 0。所以每个计数加 1，分母加上**训练数据中不同词的个数 $d$**。
12. **Multinomial NB 很实用**："very powerful… very fast, efficient and good for text classification"。

### 0.3 总复习 PPT（Revision2026S1）里的这一章：p.2–6

| 页 | 内容 | 对应本讲义 |
|---|---|---|
| p.2 | **Bayes theorem**：$p(\omega_j\mid x) = \dfrac{p(x\mid\omega_j)p(\omega_j)}{p(x)}$，$p(x) = \sum_j p(x\mid\omega_j)p(\omega_j)$；标出 prior、class-conditional probability density function、posterior 三个名词 | §1 |
| p.3 | **Gaussian class-conditional pdf**：$p(x\mid\omega_j) = \dfrac{1}{(2\pi)^{d/2}\lvert C_j\rvert^{1/2}}\, e^{-\frac12(x - m_j)^TC_j^{-1}(x - m_j)}$ | §3 |
| p.4 | **Bayes decision rule**：$p(\omega_i\mid x) > p(\omega_j\mid x)$ 对所有 $j \ne i$ 就判 $\omega_i$；或用变体 $p(x\mid\omega_i)p(\omega_i) > p(x\mid\omega_j)p(\omega_j)$ | §2 |
| p.5 | **Naive Bayes**：假设特征独立，$p(\omega_j\mid x) = \dfrac{\prod_i p(x_i\mid\omega_j)\,p(\omega_j)}{p(x)}$；Gaussian NB 用于连续特征 | §5 |
| p.6 | **Bernoulli NB 用于二值特征**：一张计数表（$X_i = A$ 时 $\omega_1$ 100 个、$\omega_2$ 200 个；$X_i = B$ 时 $\omega_1$ 150 个、$\omega_2$ 100 个），"$p(\omega_j)$, $p(x_i\mid\omega_j)$ are estimated from the information in the table"；**Multinomial NB 用于文本分类** | §5、§6.5 |

> 注：总复习 p.4 的变体在 PPT 上印成了 "$p(x\mid\omega_i)p(\omega_i) > p(\omega_j\mid x)p(\omega_j)$"，右边应为 $p(x\mid\omega_j)p(\omega_j)$，是排版笔误。

![Revision p.2 和 p.3](figs/nb/revision_p2_p3.png)

![Revision p.4 和 p.5](figs/nb/revision_p4_p5.png)

![Revision p.6](figs/nb/revision_p6.png)

**由此得到的复习优先级**：

| 级别 | 内容 | 理由 |
|---|---|---|
| **第一级：必须能默写和会算** | 总复习 p.2–6：四种概率的名称和关系；高斯密度公式；Bayes 决策规则及其联合概率变体；朴素贝叶斯公式；三种 NB 各用于什么特征；**从计数表估计概率** | 总复习专门列出；Bernoulli 连考两年 |
| **第二级：必须会做** | 完整 Gaussian Bayes 的对数判别函数（24-25 S1，12 分）；Gaussian NB 及"散度 ≠ 协方差"陷阱；Multinomial NB + Laplace 平滑；写出假设 | 总复习只列了名字，计算全靠课件例题和真题 |
| **第三级：理解即可** | MLE 推导；边界形状（$C$ 不同→二次，$C$ 相同→直线）；Bayes 与 LDA 的关系；GMM 和 EM 的步骤；NB 的优缺点 | 用来回答"为什么"；GMM 会在聚类里再出现 |

---

## 1. 四种概率：prior、likelihood、posterior、joint（Slides3 p.1–8，总复习 p.2）

### 1.1 课件的鱼例子

把鱼分成 sea bass（鲈鱼，$\omega_1$）和 salmon（鲑鱼，$\omega_2$）。特征 $x$ 是鱼的 lightness（亮度）。

| 名称 | 符号 | 含义 | 怎么得到 |
|---|---|---|---|
| **Prior probability（先验概率）** | $P(\omega_j)$ | 看到 $x$ **之前**对类别的初始信念（"initial belief"） | 训练集中各类样本所占比例 |
| **Class-conditional probability density function（类条件概率密度）**，也叫 **likelihood（似然）** | $p(x\mid\omega_j)$ | 已知属于 $\omega_j$ 时，$x$ 的分布 | 从该类训练样本估计（例如假设高斯） |
| **Joint probability（联合概率）** | $p(x, \omega_j) = p(x\mid\omega_j)P(\omega_j)$ | $x$ 和 $\omega_j$ 同时发生的概率 | 上面两项相乘 |
| **Posterior probability（后验概率）** | $P(\omega_j\mid x)$ | 看到 $x$ **之后**属于 $\omega_j$ 的概率 | Bayes 定理 |
| **Evidence（证据）** | $p(x) = \sum_j p(x\mid\omega_j)P(\omega_j)$ | 归一化常数 | 对所有类求和 |

**Bayes theorem（贝叶斯定理）**：

$$P(\omega_j\mid x) = \frac{p(x\mid\omega_j)\,P(\omega_j)}{p(x)}$$

### 1.2 为什么不能只用先验（老师的男女生例子）

- 只有先验时，只能永远猜先验大的那一类（课件 p.3）。
- 60% 男生、40% 女生，就会把每个人都判成男生，女生全错，错误率 40%，分类就没有意义。
- 加上特征 $x$（身高、体重；鱼的亮度、长度），用后验决策，才能把两类区分开。

**看课件 Slides3 p.4 和 p.9**（下图左、右）：
- p.4 是两类的类条件密度曲线。横轴是亮度 $x$，两条曲线峰的位置不同，说明两类鱼的亮度分布不同，这就是可以用来分类的信息。
- p.9 是 discriminant function（判别函数）的结构：同一个 $x$ 送进 $g_1$、$g_2$、…、$g_c$，选出最大的那个。

![Slides3 p.4 和 p.9](figs/nb/slides_p4_p9.png)

下图演示先验怎样影响决策（课件没有这张图）：
- (a) 两类的类条件密度，和课件 p.4 类似。
- (b) 先验相等时，两条后验曲线在 $x = 11.66$ 处相交，这里就是决策边界。
- (c) 把 $\omega_1$ 的先验提高到 0.8，边界右移到 12.42，$\omega_1$ 的判决区域变大。

![图 1-1 先验对边界的影响](figs/nb/fig1_prior_shift.png)

---

## 2. Bayes decision rule（贝叶斯决策规则）（Slides3 p.6–10，总复习 p.4）

### 2.1 规则

**把 $x$ 判给后验概率最大的类**：

$$\text{若 } P(\omega_i\mid x) > P(\omega_j\mid x)\ \text{对所有 } j \ne i\ \text{成立，就判 } x \in \omega_i$$

### 2.2 等价的几种判别函数（要会写）

$p(x)$ 对每一类都一样，只起缩放作用（"just a scale factor"），所以以下三种写法给出**相同的决策**：

| 判别函数 $g_j(x)$ | 说明 |
|---|---|
| $P(\omega_j\mid x)$ | 后验概率本身 |
| $p(x\mid\omega_j)\,P(\omega_j)$ | 联合概率，**考试最常用**，不用算 $p(x)$ |
| $\ln p(x\mid\omega_j) + \ln P(\omega_j)$ | 对数形式；高斯密度取对数后指数消失，计算最方便 |

课件 p.10 的总结："A Bayes classifier is determined by the **conditional probability density function** as well as the **prior probability**."

### 2.3 补充：为什么叫"最优"（课件外）

把 $x$ 判给后验最大的类，在每个 $x$ 上犯错的概率都是 $1 - \max_j P(\omega_j\mid x)$，已经是最小的，所以 Bayes 决策规则使**总错误率最小**（minimum error rate）。前提是先验和类条件密度都准确。实际中它们要从数据估计，估计不准时就不再最优。

---

## 3. Gaussian class-conditional density（高斯类条件密度）与参数估计（Slides3 p.11–26，总复习 p.3）

### 3.1 一元和多元高斯

**Univariate（一元）**：

$$p(x\mid\omega) = \frac{1}{\sqrt{2\pi}\,\sigma}\, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

两个参数：mean（均值）$\mu$、variance（方差）$\sigma^2$。

**Multivariate（多元，$d$ 维）**：

$$p(x\mid\omega_j) = \frac{1}{(2\pi)^{d/2}\lvert C_j\rvert^{1/2}}\, e^{-\frac12(x - m_j)^TC_j^{-1}(x - m_j)}$$

- $m_j$：$d$ 维均值向量（mean vector）
- $C_j$：$d \times d$ 协方差矩阵（covariance matrix），$\lvert C_j\rvert$ 是行列式（determinant），$C_j^{-1}$ 是逆矩阵
- 参数个数：$d + \dfrac{d(d + 1)}{2}$（均值 $d$ 个，对称的协方差矩阵 $d(d + 1)/2$ 个）

### 3.2 Maximum-likelihood estimation（最大似然估计，MLE）（课件 p.16–21）

**思想**：训练样本 $D = \{x_1, \dots, x_n\}$ 独立抽取，likelihood（似然）是 $p(D\mid\theta) = \prod_k p(x_k\mid\theta)$。选使它最大的 $\theta$。

**步骤**：
1. 取对数：$l(\theta) = \sum_k \ln p(x_k\mid\theta)$。对数单调递增，最大化 $l$ 等价于最大化 $p(D\mid\theta)$，而且连乘变成连加。
2. 令 $\dfrac{\partial l}{\partial\theta} = 0$。

**结果**（每一类分别用自己的样本算）：

$$\hat\mu = \frac1n\sum_k x_k,\qquad \hat C = \frac1n\sum_k (x_k - \hat\mu)(x_k - \hat\mu)^T$$

注意 MLE 的协方差**除以 $n$**，不是 $n - 1$。课件按 MLE 算，考试也按除以 $n$ 算。如果你用 $n - 1$，在答卷上写明（参考答案说只影响第 3 位小数）。

**和 LDA 的散度矩阵的关系**：$S_i = \sum(x - m_i)(x - m_i)^T$ 没有除以样本数，所以 $\hat C_i = S_i/N_i$。**题目给的是散度矩阵时，要先除以样本数**，这是 23-24 S1 的核心陷阱（见 §6.4）。

### 3.3 对数判别函数（24-25 S1 Q2(a) 的标准写法）

对 $p(x\mid\omega_i)P(\omega_i)$ 取对数，去掉各类共有的常数 $-\tfrac d2\ln 2\pi$：

$$g_i(x) = -\frac12(x - m_i)^TC_i^{-1}(x - m_i) - \frac12\ln\lvert C_i\rvert + \ln P(\omega_i)$$

三项分别是：
- 马氏距离项（Mahalanobis distance）：离均值越远，越不像这一类。
- **$-\tfrac12\ln\lvert C_i\rvert$**：分布越分散，密度越低。**两类 $C$ 不同时这一项不能省**。
- **$\ln P(\omega_i)$**：先验。**先验不等时不能省**。

### 3.4 决策边界的形状

令 $g_1(x) = g_2(x)$：
- **$C_1 \ne C_2$**：二次项 $x^T(C_1^{-1} - C_2^{-1})x$ 不能抵消，边界是**二次曲线**（hyperquadric），分类器是 non-linear（非线性）的。课件 p.26 就是这种情况。
- **$C_1 = C_2 = C$**：二次项抵消，边界是**直线（超平面）**，法向量是 $C^{-1}(m_1 - m_2)$，和 LDA 的 $w = S_W^{-1}(m_1 - m_2)$ 方向一致。这就是"LDA 在两类高斯、协方差相同时最优"的原因（见 LDA 专题 E8）。

**看课件 Slides3 p.22 和 p.26**（下图左、右）：p.22 是 200 个训练样本；p.26 是用两类各自的高斯估计得到的决策区域，绿色判类 1、黄色判类 2，两色交界是一条**曲线**。

![Slides3 p.22 和 p.26](figs/nb/slides_p22_p26.png)

下图用 §6.3 例题的参数画三种边界（课件没有这张图）：
- (a) 完整 Bayes，$C_1 \ne C_2$：红、蓝椭圆都是斜的，边界是弯曲的二次曲线。
- (b) 两类共用协方差：边界变成直线。
- (c) 朴素贝叶斯：只保留对角元，椭圆轴和坐标轴平行；边界仍然是曲线，但方程里没有 $x_1x_2$ 交叉项。
- 黑三角是测试点 $(2, 2)$，三种情况下都落在粉色（$\omega_1$）区域。

![图 3-1 三种边界](figs/nb/fig3_boundaries.png)

---

## 4. Gaussian Mixture Model（高斯混合模型，GMM）（Slides3 p.27–52）

> 这一节没有单独考过，但聚类部分的 distribution-based clustering 就是 GMM，概念要清楚。

### 4.1 什么时候用

一类的数据如果来自**多个**高斯（multimodal，多峰），用一个高斯拟合不好。GMM 用 $m$ 个高斯的加权和表示这一类的类条件密度：

$$p(x\mid\omega) = \sum_{i=1}^m \alpha_i\,\mathcal N(x\mid\mu_i, C_i),\qquad \sum_i \alpha_i = 1$$

$\alpha_i$ 是第 $i$ 个分量的权重。

### 4.2 为什么要用 EM（课件 p.31–35）

- 定义 $\gamma_{ik} = P(\text{第 } i \text{ 个分量}\mid x_k)$，即样本 $k$ 属于分量 $i$ 的概率（soft assignment，软分配）。
- MLE 给出 $\mu_i = \dfrac{\sum_k\gamma_{ik}x_k}{\sum_k\gamma_{ik}}$，$C_i$、$\alpha_i$ 类似。也就是说，每个样本按 $\gamma$ 加权参与每个分量的计算。
- 但 $\gamma$ 本身又要用 $\mu$、$C$、$\alpha$ 来算，像"先有鸡还是先有蛋"，没有闭式解。
- **EM（Expectation-Maximization）算法**交替进行：
  1. **Initialization（初始化）**：随机给 $\alpha_i$、$\mu_i$、$C_i$。
  2. **E-step（期望步）**：用当前参数算每个 $\gamma_{ik}$。
  3. **M-step（最大化步）**：用 $\gamma$ 加权更新 $\mu_i$、$C_i$、$\alpha_i$。
  4. 重复 2–3，直到参数稳定或达到迭代次数（课件做了 1000 次）。

### 4.3 分量个数 m 怎么选（课件 p.46–52）

- $m$ 是超参数，高维数据看不到分布，只能试。
- 用的分量太多时，有些 $\alpha_i$ 会很小（课件里 3 个分量时 $\alpha = 0.0523$、0.5040、0.4437），说明那个分量不重要，可以减少。减少后要**重新**跑 EM，不能直接删掉。
- 分量太多，边界变得很复杂，训练集上好、测试集上差，即 **overfitting（过拟合）**。

**看课件 Slides3 p.43 和 p.45**（下图左、右）：类 2 本来由两团数据组成。p.43 用 2 个分量，训练错 14 个；p.45 只用 1 个高斯，错 15 个，边界拟合得更差。

![Slides3 p.43 和 p.45](figs/nb/slides_p43_p45.png)

**再看课件 Slides3 p.51 和 p.52**（下图左、右）：p.51 用 5 个分量，绿黄交界变得弯弯曲曲；p.52 是老师总结的两条 observations（分量过多会过拟合；小的 $\alpha$ 可以用来判断该用几个分量）。

![Slides3 p.51 和 p.52](figs/nb/slides_p51_p52.png)

---

## 5. Naive Bayes（朴素贝叶斯）（Slides3 p.53–71，总复习 p.5–6）

### 5.1 推导：从链式法则到独立性假设（课件 p.53–55）

链式法则（chain rule）：

$$p(x_1, \dots, x_d, \omega_j) = p(x_1\mid x_2, \dots, x_d, \omega_j)\,p(x_2\mid x_3, \dots, x_d, \omega_j)\cdots p(x_d\mid\omega_j)\,P(\omega_j)$$

**Naive assumption（朴素假设）**：**给定类别后，各特征 conditionally independent（条件独立）**，于是 $p(x_i\mid x_{i+1}, \dots, x_d, \omega_j) = p(x_i\mid\omega_j)$，得到

$$p(x\mid\omega_j)P(\omega_j) = \prod_{i=1}^d p(x_i\mid\omega_j)\cdot P(\omega_j)$$

$$P(\omega_j\mid x) = \frac{\prod_i p(x_i\mid\omega_j)\,P(\omega_j)}{p(x)}$$

决策时同样不用算 $p(x)$，比较分子即可。

### 5.2 为什么要这样简化

| | 完整 Gaussian Bayes | Gaussian NB |
|---|---|---|
| 每类要估计的参数 | $d + d(d + 1)/2$ | $2d$（每个特征一个均值、一个方差） |
| $d = 10$ | 65 | 20 |
| $d = 100$ | 5150 | 200 |
| 需要求逆吗 | 要求 $C_j^{-1}$ 和 $\lvert C_j\rvert$ | 不需要 |

参数少，所需样本就少，估计更稳定，也缓解了 curse of dimensionality（维数灾难）（课件 p.71）。

**代价**：如果特征实际上强相关，独立性假设就不成立，丢掉了相关信息。

下图演示这一点（课件没有这张图）：同一组强相关的数据（$\rho = 0.87$），(a) 完整高斯的等高线是斜椭圆，贴合数据；(b) 朴素贝叶斯只用两个方差，等高线变成轴对齐的圆，把左下到右上的相关结构完全丢了。

![图 5-1 朴素假设丢掉了什么](figs/nb/fig5_naive_assumption.png)

**为什么假设"天真"还好用**（课件 p.71）："In spite of the apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations such as **document classification and spam filtering**." 分类只需要**哪一类的后验最大**判对，不需要概率值本身准确；而且它训练极快，所需样本少。

### 5.3 三种朴素贝叶斯（课件 p.55，总复习 p.5–6）

| 类型 | 适用特征 | $p(x_i\mid\omega_j)$ 怎么估计 |
|---|---|---|
| **Gaussian NB** | **连续**特征 | 每个特征一个一元高斯：$\mu_{ij}$ = 该类第 $i$ 维的均值，$\sigma^2_{ij}$ = 该类第 $i$ 维的方差（除以 $N_j$） |
| **Bernoulli NB** | **二值**特征（yes/no、0/1、male/female） | 在 $\omega_j$ 的样本中，$x_i$ 取这个值的**比例** |
| **Multinomial NB** | **计数**特征，典型是文本的词频 | 词 $x_i$ 在 $\omega_j$ 所有文档中出现的次数 ÷ $\omega_j$ 的总词数（通常加 Laplace 平滑） |

### 5.4 Bernoulli NB：课件原例（**连考两年**）

**看课件 Slides3 p.58 和 p.62**（下图左、右）：p.58 是 5 条训练记录（Confident、Studied、Sick → Pass/Fail），要分类 Confident = Yes、Studied = Yes、Sick = No；p.62 是最终的两个乘积 $4/45 \approx 0.088$ 和 $1/20 = 0.05$。

![Slides3 p.58 和 p.62](figs/nb/slides_p58_p62.png)

训练数据（课件 p.58）：

| # | Confident | Studied | Sick | Result |
|---|---|---|---|---|
| 1 | Yes | No | No | Fail |
| 2 | Yes | No | Yes | Pass |
| 3 | No | Yes | Yes | Fail |
| 4 | No | Yes | No | Pass |
| 5 | Yes | Yes | Yes | Pass |

记 Pass = $\omega_1$、Fail = $\omega_2$，测试样本 $x$ = (Confident = Yes, Studied = Yes, Sick = No)。

**步骤 1：先验**
- $P(\omega_1) = 3/5 = 0.6$（Pass 有 3 条：#2、#4、#5）
- $P(\omega_2) = 2/5 = 0.4$（Fail 有 2 条：#1、#3）

**步骤 2：类条件概率**（只在该类的样本里数）

| | $\omega_1$ = Pass（#2、#4、#5） | $\omega_2$ = Fail（#1、#3） |
|---|---|---|
| $P(\text{Confident = Yes}\mid\cdot)$ | #2、#5 → $2/3$ | #1 → $1/2$ |
| $P(\text{Studied = Yes}\mid\cdot)$ | #4、#5 → $2/3$ | #3 → $1/2$ |
| $P(\text{Sick = No}\mid\cdot)$ | #4 → $1/3$ | #1 → $1/2$ |

**步骤 3：联合概率**

$$p(x\mid\omega_1)P(\omega_1) = \frac23\cdot\frac23\cdot\frac13\cdot\frac35 = \frac{4}{45} \approx 0.0889$$

$$p(x\mid\omega_2)P(\omega_2) = \frac12\cdot\frac12\cdot\frac12\cdot\frac25 = \frac{1}{20} = 0.05$$

**结论**：$0.0889 > 0.05$，判为 **Pass**。归一化后验 $P(\text{Pass}\mid x) = \dfrac{0.0889}{0.0889 + 0.05} = 0.64$。

> 易错点：Sick = **No** 要数 No 的个数（$\omega_1$ 里只有 #4），不要数成 Yes；不要漏乘先验。这道题课件和两份真题都**没有用平滑**。

### 5.5 Multinomial NB：课件原例（"A very close game"）

**看课件 Slides3 p.67 和 p.69**（下图左、右）：p.67 是 5 条带标签的训练句子；p.69 是加 1 平滑后每个词的条件概率表，以及 Sports 类的计算结果 $2.765 \times 10^{-5}$。

![Slides3 p.67 和 p.69](figs/nb/slides_p67_p69.png)

| 文本 | 类别 |
|---|---|
| "A great game" | Sports |
| "The election was over" | Not sports |
| "Very clean match" | Sports |
| "A clean but forgettable game" | Sports |
| "It was a close election" | Not sports |

测试句子："A very close game"。

**基本公式**（课件 p.66）：

$$p(x_i\mid\omega_j) = \frac{\mathrm{count}(x_i, \omega_j)}{\sum_k \mathrm{count}(x_k, \omega_j)}$$

分子是词 $x_i$ 在 $\omega_j$ 的所有文本里**一共出现几次**，分母是 $\omega_j$ 的**总词数**。

**零概率问题**：close 在 Sports 里出现 0 次，$p(\text{close}\mid\text{Sports}) = 0/11 = 0$，整个乘积就变成 0，不管其他词多像体育新闻。

**Laplace smoothing（拉普拉斯平滑，加 1 平滑）**（课件 p.68）：

$$p(x_i\mid\omega_j) = \frac{\mathrm{count}(x_i, \omega_j) + 1}{\sum_k \mathrm{count}(x_k, \omega_j) + d}$$

**$d$ 是全部训练数据中不同词（unique words）的个数**，两类共用。

本例：
- Sports 总词数 $= 3 + 3 + 5 = 11$；Not sports 总词数 $= 4 + 5 = 9$。
- 不同的词：a, great, game, the, election, was, over, very, clean, match, but, forgettable, it, close，共 $d = 14$。
- 所以分母分别是 $11 + 14 = 25$ 和 $9 + 14 = 23$。

| 词 | Sports 次数 | $P(\text{词}\mid\text{Sports})$ | Not sports 次数 | $P(\text{词}\mid\text{Not sports})$ |
|---|---|---|---|---|
| a | 2 | $3/25$ | 1 | $2/23$ |
| very | 1 | $2/25$ | 0 | $1/23$ |
| close | 0 | $1/25$ | 1 | $2/23$ |
| game | 2 | $3/25$ | 0 | $1/23$ |

- $P(\text{Sports}) = 3/5$，$P(\text{Not sports}) = 2/5$。
- Sports：$\dfrac{3}{25}\cdot\dfrac{2}{25}\cdot\dfrac{1}{25}\cdot\dfrac{3}{25}\cdot\dfrac35 = 2.765 \times 10^{-5}$
- Not sports：$\dfrac{2}{23}\cdot\dfrac{1}{23}\cdot\dfrac{2}{23}\cdot\dfrac{1}{23}\cdot\dfrac25 = 5.718 \times 10^{-6}$

**结论**：判为 **Sports**。

> 易错点：分母加的是 $d$，不是 1，也不是 2；$d$ 是**所有类合在一起**的不同词数；词出现几次就数几次（"a" 在 Sports 里出现 2 次）；测试句子里一个词出现两次，就要乘两次它的概率。

---

## 6. 计算题标准流程 + 四道例题

### 6.1 通用流程（写全了就有步骤分）

1. **先验**：$P(\omega_j) = N_j/N$（题目给了就直接用）。
2. **类条件概率**：按题型选 Gaussian / Bernoulli / Multinomial；**写出假设**（独立性、高斯、用什么估计）。
3. **乘起来**：$g_j = P(\omega_j)\prod_i p(x_i\mid\omega_j)$，或者用对数形式。
4. **比较**：取最大者；最好再算一下归一化后验 $P(\omega_j\mid x) = g_j/\sum_k g_k$，说明有多大把握。

### 6.2 常见失分点

- **漏乘先验**（尤其先验不等时）。
- **散度矩阵当方差用**：给 $S_i$ 就要除以 $N_i$（§6.4）。
- **$\sigma$ 和 $\sigma^2$ 混淆**：密度公式分母是 $\sqrt{2\pi}\,\sigma$，指数里是 $2\sigma^2$。
- **完整 Bayes 漏掉 $-\tfrac12\ln\lvert C_i\rvert$**：两类 $C$ 不同时这一项不能省。二次型展开时交叉项要乘 2。
- **朴素贝叶斯做成了完整协方差**：NB 只用对角元（方差），忽略协方差。
- **Bernoulli 数错值**：注意测试值是 Yes 还是 No，只在该类里数。
- **Multinomial 平滑的分母**：加 $d$（不同词的总个数），不是加 1。
- **没写假设**：题目说 "state the assumptions" 时有专门的分。

### 6.3 例题 A：完整 Gaussian Bayes（24-25 S1 Q2(a) 的格式，数值已用程序核对）

**题目**：两类服从二维高斯，$P(\omega_1) = 0.6$，$P(\omega_2) = 0.4$；
- $m_1 = (1, 2)^T$，$C_1 = \begin{bmatrix}1&0.5\\0.5&2\end{bmatrix}$
- $m_2 = (3, 1)^T$，$C_2 = \begin{bmatrix}2&-0.4\\-0.4&1\end{bmatrix}$

写出 Bayes 判别函数，并对 $x = (2, 2)^T$ 分类。

**判别函数**：$g_i(x) = -\tfrac12(x - m_i)^TC_i^{-1}(x - m_i) - \tfrac12\ln\lvert C_i\rvert + \ln P(\omega_i)$；$g_1 > g_2$ 判 $\omega_1$。因为 $C_1 \ne C_2$，边界是二次曲线（图 3-1(a)）。

**所需的量**

| | $\omega_1$ | $\omega_2$ |
|---|---|---|
| $\lvert C_i\rvert$ | $1\times 2 - 0.5^2 = 1.75$ | $2\times 1 - 0.4^2 = 1.84$ |
| $C_i^{-1}$ | $\dfrac{1}{1.75}\begin{bmatrix}2&-0.5\\-0.5&1\end{bmatrix} = \begin{bmatrix}1.142857&-0.285714\\-0.285714&0.571429\end{bmatrix}$ | $\dfrac{1}{1.84}\begin{bmatrix}1&0.4\\0.4&2\end{bmatrix} = \begin{bmatrix}0.543478&0.217391\\0.217391&1.086957\end{bmatrix}$ |
| $\ln\lvert C_i\rvert$ | 0.559616 | 0.609766 |
| $\ln P(\omega_i)$ | −0.510826 | −0.916291 |

**代入 $x = (2, 2)$**

- $\omega_1$：$x - m_1 = (1, 0)$，二次型 $= 1.142857$
  - $g_1 = -\tfrac12(1.142857) - \tfrac12(0.559616) - 0.510826 = -1.3621$
- $\omega_2$：$x - m_2 = (-1, 1)$，二次型 $= 0.543478 + 2(0.217391)(-1)(1) + 1.086957 = 1.195652$
  - $g_2 = -\tfrac12(1.195652) - \tfrac12(0.609766) - 0.916291 = -1.8190$

**结论**：$g_1 > g_2$，判为 $\omega_1$。换算成概率：$p(x\mid\omega_1)P(\omega_1) = 0.040765$，$p(x\mid\omega_2)P(\omega_2) = 0.025813$，后验 $P(\omega_1\mid x) = 0.612$。

### 6.4 例题 B：Gaussian NB + 散度矩阵陷阱（23-24 S1 的格式）

**题目**：$\omega_1$ 有 50 个样本，$\omega_2$ 有 30 个。均值和**散度矩阵**为
- $m_1 = (2.0, 5.0)^T$，$S_1 = \begin{bmatrix}45&10\\10&80\end{bmatrix}$
- $m_2 = (4.0, 3.5)^T$，$S_2 = \begin{bmatrix}54&-6\\-6&27\end{bmatrix}$

用 Gaussian NB 对 $x = (3.0, 4.5)^T$ 分类，写出假设。

**假设**
1. 给定类别后，两个特征条件独立：$p(x\mid\omega_j) = p(x_1\mid\omega_j)\,p(x_2\mid\omega_j)$。所以**只用散度矩阵的对角元**，非对角元 10 和 −6 被忽略。
2. **每个特征在每一类内服从一元高斯**，方差用 MLE：$\sigma^2_{jk} = S_j(k, k)/N_j$。
3. 先验按样本比例：$P(\omega_1) = 50/80 = 0.625$，$P(\omega_2) = 30/80 = 0.375$。

**参数**

| | $\mu_1$ | $\sigma_1^2$ | $\mu_2$ | $\sigma_2^2$ |
|---|---|---|---|---|
| $\omega_1$ | 2.0 | $45/50 = 0.9$ | 5.0 | $80/50 = 1.6$ |
| $\omega_2$ | 4.0 | $54/30 = 1.8$ | 3.5 | $27/30 = 0.9$ |

（表头的下标 1、2 指特征 $x_1$、$x_2$。）

**计算**

| | $p(x_1\mid\omega_j)$ | $p(x_2\mid\omega_j)$ | 乘积 | × 先验 |
|---|---|---|---|---|
| $\omega_1$ | $\dfrac{1}{\sqrt{2\pi\cdot 0.9}}e^{-(1.0)^2/1.8} = 0.2413$ | $\dfrac{1}{\sqrt{2\pi\cdot 1.6}}e^{-(0.5)^2/3.2} = 0.2917$ | 0.070378 | **0.04399** |
| $\omega_2$ | $\dfrac{1}{\sqrt{2\pi\cdot 1.8}}e^{-(1.0)^2/3.6} = 0.2252$ | $\dfrac{1}{\sqrt{2\pi\cdot 0.9}}e^{-(1.0)^2/1.8} = 0.2413$ | 0.054344 | **0.02038** |

**结论**：判为 $\omega_1$，后验 $P(\omega_1\mid x) = 0.683$。

> **陷阱演示**：如果直接把 45、80 当方差，$\sigma \approx 7$–9，密度几乎是平的：两类的结果分别是 0.00164 和 0.00152，几乎分不开。散度 ≠ 协方差，**一定要除以样本数**。

### 6.5 例题 C：Bernoulli NB，从计数表出发（总复习 p.6 的格式）

**题目**：两个二值特征。训练集的计数如下（$X_1$ 的表就是总复习 p.6 那张）：

| $X_1$ | $\omega_1$ | $\omega_2$ |
|---|---|---|
| A | 100 | 200 |
| B | 150 | 100 |

| $X_2$ | $\omega_1$ | $\omega_2$ |
|---|---|---|
| Y | 200 | 90 |
| N | 50 | 210 |

对 $x = (X_1 = A, X_2 = Y)$ 分类。

**先验**：$N_1 = 100 + 150 = 250$，$N_2 = 200 + 100 = 300$，$N = 550$。$P(\omega_1) = 250/550 = 0.4545$，$P(\omega_2) = 300/550 = 0.5455$。（两张表各自的列和应相等，可以用来检查读表是否正确。）

**类条件概率**

| | $\omega_1$ | $\omega_2$ |
|---|---|---|
| $P(X_1 = A\mid\cdot)$ | $100/250 = 0.4$ | $200/300 = 0.6667$ |
| $P(X_2 = Y\mid\cdot)$ | $200/250 = 0.8$ | $90/300 = 0.3$ |

**联合概率**
- $\omega_1$：$0.4545 \times 0.4 \times 0.8 = 0.1455$
- $\omega_2$：$0.5455 \times 0.6667 \times 0.3 = 0.1091$

**结论**：判为 $\omega_1$，后验 0.571。

> 注意：单看 $X_1 = A$，$\omega_2$ 更像（0.667 对 0.4）；但 $X_2 = Y$ 强烈支持 $\omega_1$（0.8 对 0.3），乘起来 $\omega_1$ 胜出。这就是"每个特征各投一票，票数相乘"。
> 小技巧：$P(\omega_j)\,P(A\mid\omega_j) = \dfrac{N_j}{N}\cdot\dfrac{\mathrm{count}(A, \omega_j)}{N_j} = \dfrac{\mathrm{count}(A, \omega_j)}{N}$，可以少算一步。

### 6.6 例题 D：Multinomial NB + Laplace 平滑（25-26 S1 Q3(b) 的格式）

**题目**：训练文本
- Spam："win cash now"、"win prize"
- Ham："meeting now"、"project meeting today"

用 Multinomial NB（Laplace 平滑）对 "cash meeting now" 分类。

**统计**
- 先验：$P(\text{Spam}) = P(\text{Ham}) = 2/4 = 0.5$。
- Spam 词频：win 2、cash 1、now 1、prize 1，共 5 个词。
- Ham 词频：meeting 2、now 1、project 1、today 1，共 5 个词。
- 不同的词：cash, meeting, now, prize, project, today, win，$d = 7$。所以两类的分母都是 $5 + 7 = 12$。

**为什么必须平滑**：meeting 在 Spam 里出现 0 次，cash 在 Ham 里出现 0 次，不平滑的话两类的乘积都是 0，无法比较。

| 词 | $P(\text{词}\mid\text{Spam})$ | $P(\text{词}\mid\text{Ham})$ |
|---|---|---|
| cash | $(1+1)/12 = 2/12$ | $(0+1)/12 = 1/12$ |
| meeting | $(0+1)/12 = 1/12$ | $(2+1)/12 = 3/12$ |
| now | $(1+1)/12 = 2/12$ | $(1+1)/12 = 2/12$ |

- Spam：$0.5 \times \dfrac{2\cdot 1\cdot 2}{12^3} = 0.5 \times \dfrac{4}{1728} = 1.157 \times 10^{-3}$
- Ham：$0.5 \times \dfrac{1\cdot 3\cdot 2}{12^3} = 0.5 \times \dfrac{6}{1728} = 1.736 \times 10^{-3}$

**结论**：判为 **Ham**，后验 $6/10 = 0.6$。

---

## 7. 三类方法的比较与 NB 的优缺点

### 7.1 Full Bayes vs Naive Bayes vs LDA

| | 完整 Gaussian Bayes | Gaussian Naive Bayes | Fisher LDA |
|---|---|---|---|
| 模型 | 每类一个完整高斯 $\mathcal N(m_j, C_j)$ | 每类每个特征一个一元高斯 | 不建概率模型，找投影方向 |
| 考虑特征相关吗 | 是 | **否**（独立假设） | 是（通过 $S_W$） |
| 参数（每类） | $d + d(d+1)/2$ | $2d$ | 两类共用 $S_W$ |
| 边界 | 二次曲线（$C$ 相同时为直线） | 二次曲线，但没有交叉项 | 直线 |
| 输出 | 后验概率 | 后验概率 | 判别值 $g(x)$（距离） |
| 先验 | 显式使用 | 显式使用 | 只在中点阈值里隐含"先验相等" |
| 适合 | 样本多、特征相关 | 样本少、维数高、特征近似独立；文本 | 两类近似高斯、协方差相近 |

### 7.2 Naive Bayes 的优点和缺点（论述题素材）

**优点**
1. 参数少（每类 $2d$ 个），所需训练样本少，缓解维数灾难（课件 p.71）。
2. 训练和预测都很快：每个特征的分布单独估计，只需计数或求均值、方差。
3. 不需要求协方差矩阵的逆，不会遇到奇异问题。
4. 可以处理不同类型的特征（连续用 Gaussian，二值用 Bernoulli，计数用 Multinomial）。
5. 在文本分类、垃圾邮件过滤中效果很好（课件 p.71）。
6. 输出后验概率，可以表示置信度。

**缺点**
1. 条件独立的假设通常不成立；特征强相关时，相关信息丢失，同一信息被重复计算，后验会过于极端（overconfident）。
2. 零概率问题：训练集里没出现过的取值会让整个乘积为 0，需要 Laplace 平滑。
3. Gaussian NB 假设每个特征服从正态分布；分布偏斜或多峰时效果差。
4. 得到的概率值本身常常不准，只适合用来比大小。

---

## 8. 论述题专项

答题框架同前：**Definition → Explanation → Example / Equation → Conclusion**，要点个数 ≈ 分值 ÷ 2。

### E1 ★★★ Explain the prior, class-conditional density, posterior and joint probability, and state the Bayes decision rule.
（解释先验概率、类条件密度、后验概率和联合概率，并写出 Bayes 决策规则。）

> - The **prior probability** $P(\omega_j)$ is the probability of class $\omega_j$ before observing the feature vector; it represents the initial belief and is estimated as the fraction of training samples in $\omega_j$.
> - The **class-conditional probability density** (likelihood) $p(x\mid\omega_j)$ describes how the feature vector is distributed within class $\omega_j$; it is estimated from the training samples of that class, e.g. by a Gaussian with mean $m_j$ and covariance $C_j$.
> - The **joint probability** $p(x, \omega_j) = p(x\mid\omega_j)P(\omega_j)$ is the probability that $x$ and $\omega_j$ occur together.
> - The **posterior probability** $P(\omega_j\mid x) = p(x\mid\omega_j)P(\omega_j)/p(x)$, with $p(x) = \sum_j p(x\mid\omega_j)P(\omega_j)$, is the probability of class $\omega_j$ after observing $x$ (Bayes theorem).
> - **Bayes decision rule:** assign $x$ to $\omega_i$ if $P(\omega_i\mid x) > P(\omega_j\mid x)$ for all $j \ne i$. Since $p(x)$ is common to all classes, the rule is equivalent to comparing $p(x\mid\omega_j)P(\omega_j)$, or $\ln p(x\mid\omega_j) + \ln P(\omega_j)$. This rule minimizes the probability of classification error.

> 【中文】
> - **先验概率** P(ωⱼ) 是观测特征向量之前类别 ωⱼ 的概率，代表最初的判断，用训练样本中属于 ωⱼ 的比例来估计。
> - **类条件概率密度**（似然）p(x|ωⱼ) 描述特征向量在 ωⱼ 类内部的分布，用该类的训练样本估计，例如用均值为 mⱼ、协方差为 Cⱼ 的高斯。
> - **联合概率** p(x, ωⱼ) = p(x|ωⱼ)P(ωⱼ) 是 x 和 ωⱼ 同时出现的概率。
> - **后验概率** P(ωⱼ|x) = p(x|ωⱼ)P(ωⱼ)/p(x)，其中 p(x) = Σⱼ p(x|ωⱼ)P(ωⱼ)，是观测到 x 之后类别 ωⱼ 的概率（Bayes 定理）。
> - **Bayes 决策规则**：若对所有 j ≠ i 都有 P(ωᵢ|x) > P(ωⱼ|x)，就把 x 判给 ωᵢ。p(x) 对各类相同，所以等价于比较 p(x|ωⱼ)P(ωⱼ)，或 ln p(x|ωⱼ) + ln P(ωⱼ)。这个规则使分类错误概率最小。

### E2 ★★ Why should a classifier use the posterior rather than the prior probability?
（为什么分类器应该用后验概率而不是先验概率？）

> The prior contains no information about the sample being classified. A decision based on priors alone assigns every sample to the class with the largest prior; for example, if 60% of the students are male, every student is classified as male and all female students (40%) are misclassified. The posterior combines the prior with the evidence from the measured features through the class-conditional density, so samples whose features are typical of the minority class can still be assigned to it. The prior still matters: it shifts the decision boundary towards the less probable class.

> 【中文】先验不包含待分类样本本身的任何信息。只按先验决策，会把每个样本都判给先验最大的类；例如 60% 的学生是男生，那么每个学生都被判为男生，全部女生（40%）都被分错。后验通过类条件密度把先验和测得的特征证据结合起来，所以特征像少数类的样本仍然可以判给少数类。先验依然有作用：它会把决策边界推向概率较小的那一类。

### E3 ★★★ What is the "naive" assumption in Naive Bayes? Why is it made, and why does the classifier still work well?
（朴素贝叶斯的"朴素"假设是什么？为什么要做这个假设？为什么分类器仍然好用？）

> **Assumption.** Given the class, the features are conditionally independent: $p(x_1, \dots, x_d\mid\omega_j) = \prod_i p(x_i\mid\omega_j)$. It is called naive because in practice features are often correlated.
>
> **Why it is made.** Estimating a full $d$-dimensional class-conditional density is difficult: a Gaussian needs $d + d(d+1)/2$ parameters per class and the inversion of a covariance matrix, which is inaccurate or impossible when the number of samples is small. Under independence, only $d$ one-dimensional distributions per class are needed ($2d$ parameters for Gaussian NB), each estimated separately. This reduces the required training data and alleviates the curse of dimensionality.
>
> **Why it still works.** Classification only requires the correct class to have the largest posterior, not accurate probability values; even when the independence assumption is violated, the ranking of the classes is often correct. Naive Bayes has worked well in document classification and spam filtering, and it is extremely fast to train.
>
> **Limitation.** When features are strongly correlated, the dependence information is lost and the same evidence may be counted several times, which can degrade accuracy.

> 【中文】
> **假设。** 给定类别后，各特征条件独立：p(x₁, …, x_d|ωⱼ) = Πᵢ p(xᵢ|ωⱼ)。之所以叫"朴素"，是因为实际中特征往往是相关的。
>
> **为什么这样假设。** 估计完整的 d 维类条件密度很困难：一个高斯每类需要 d + d(d+1)/2 个参数，还要对协方差矩阵求逆，样本少时不准甚至做不到。在独立假设下，每类只需 d 个一维分布（Gaussian NB 为 2d 个参数），各自单独估计。这减少了所需的训练数据，缓解了维数灾难。
>
> **为什么仍然好用。** 分类只要求正确的类后验最大，不要求概率值准确；即使独立假设不成立，各类的排序也常常是对的。朴素贝叶斯在文档分类和垃圾邮件过滤中效果很好，而且训练极快。
>
> **局限。** 特征强相关时，相关信息丢失，同一证据可能被重复计算多次，会降低准确率。

### E4 ★★ Describe the three types of Naive Bayes classifier and when each is used.
（描述三种朴素贝叶斯分类器及各自的适用场合。）

> 1. **Gaussian NB** for continuous features: each feature in each class is modelled by a univariate normal density with the class mean $\mu_{ij}$ and variance $\sigma^2_{ij}$ estimated by maximum likelihood.
> 2. **Bernoulli NB** for binary features (yes/no, 0/1, male/female): $p(x_i\mid\omega_j)$ is the fraction of training samples in $\omega_j$ that take the given value of $x_i$.
> 3. **Multinomial NB** for count features, typically word counts in text classification: $p(\text{word}_i\mid\omega_j)$ is the number of occurrences of the word in documents of class $\omega_j$ divided by the total number of words in that class, usually with Laplace smoothing.

> 【中文】
> 1. **Gaussian NB** 用于连续特征：每类的每个特征用一元正态密度建模，均值 μᵢⱼ 和方差 σᵢⱼ² 用最大似然估计。
> 2. **Bernoulli NB** 用于二值特征（是/否、0/1、男/女）：p(xᵢ|ωⱼ) 是 ωⱼ 的训练样本中 xᵢ 取该值的比例。
> 3. **Multinomial NB** 用于计数特征，典型的是文本分类中的词频：p(词ᵢ|ωⱼ) 是该词在 ωⱼ 类文档中出现的次数除以该类的总词数，通常加 Laplace 平滑。

### E5 ★★ Why is Laplace smoothing needed in Multinomial Naive Bayes?
（为什么多项式朴素贝叶斯需要 Laplace 平滑？）

> If a word in the test document never appears in the training documents of class $\omega_j$, its estimated conditional probability is 0, and the product $\prod_i p(x_i\mid\omega_j)P(\omega_j)$ becomes 0 regardless of all other words. The class is then excluded purely because of one unseen word, and if each class has some unseen word, all posteriors are zero and no decision can be made. Laplace (add-one) smoothing estimates $p(x_i\mid\omega_j) = \dfrac{\mathrm{count}(x_i, \omega_j) + 1}{\sum_k \mathrm{count}(x_k, \omega_j) + d}$, where $d$ is the number of unique words in all training data. Every word gets a small non-zero probability and the probabilities still sum to 1 over the vocabulary.

> 【中文】如果测试文档中某个词从没在 ωⱼ 类的训练文档中出现过，它的条件概率估计为 0，乘积 Πᵢ p(xᵢ|ωⱼ)P(ωⱼ) 就变成 0，不管其他词怎样。这个类仅仅因为一个没见过的词就被排除；如果每一类都有没见过的词，所有后验都是 0，无法决策。Laplace（加一）平滑用 p(xᵢ|ωⱼ) = (count(xᵢ, ωⱼ) + 1) / (Σₖ count(xₖ, ωⱼ) + d) 估计，其中 d 是全部训练数据中不同词的个数。每个词都得到一个小的非零概率，而且这些概率在词表上的和仍为 1。

### E6 ★★ Compare the full Gaussian Bayes classifier with Gaussian Naive Bayes.
（比较完整的高斯 Bayes 分类器和高斯朴素贝叶斯。）

> Both model each class with Gaussian densities and use the Bayes decision rule. The full classifier uses a $d$-dimensional Gaussian with a full covariance matrix $C_j$, so it captures correlations between features and gives a quadratic decision boundary with cross terms; it needs $d + d(d+1)/2$ parameters per class and the inverse and determinant of $C_j$. Gaussian NB assumes conditional independence, which is equivalent to using only the diagonal of $C_j$; it needs only $2d$ parameters per class, no matrix inversion, and is more reliable when the sample size is small relative to $d$. When the features are nearly uncorrelated, both give almost the same result (24-25 S1: −1.147 vs −1.141); when they are strongly correlated, the full model is more accurate if enough data are available.

> 【中文】两者都用高斯密度为每一类建模，都用 Bayes 决策规则。完整分类器用带完整协方差矩阵 Cⱼ 的 d 维高斯，能刻画特征间的相关性，决策边界是带交叉项的二次曲线；每类需要 d + d(d+1)/2 个参数，还要 Cⱼ 的逆和行列式。Gaussian NB 假设条件独立，相当于只用 Cⱼ 的对角线；每类只需 2d 个参数，不需要求逆，样本数相对 d 较少时更可靠。特征几乎不相关时，两者结果几乎相同（24-25 S1：−1.147 对 −1.141）；特征强相关且数据足够时，完整模型更准确。

### E7 ★★ How are the parameters of a Gaussian class-conditional density estimated?
（高斯类条件密度的参数是怎样估计的？）

> By maximum-likelihood estimation. Assuming the $n$ training samples of a class are drawn independently, the likelihood is $p(D\mid\theta) = \prod_k p(x_k\mid\theta)$. Because the logarithm is monotonic, we maximize the log-likelihood $l(\theta) = \sum_k \ln p(x_k\mid\theta)$, which turns the product into a sum and cancels the exponential of the Gaussian. Setting $\partial l/\partial\theta = 0$ gives $\hat\mu = \frac1n\sum_k x_k$ and $\hat C = \frac1n\sum_k(x_k - \hat\mu)(x_k - \hat\mu)^T$. The priors are estimated as $N_j/N$. Each class is estimated separately from its own samples.

> 【中文】用最大似然估计。假设某类的 n 个训练样本独立抽取，似然为 p(D|θ) = Πₖ p(xₖ|θ)。对数单调递增，所以最大化对数似然 l(θ) = Σₖ ln p(xₖ|θ)，这把连乘变成连加，并消掉高斯的指数。令 ∂l/∂θ = 0，得 μ̂ = (1/n)Σₖ xₖ，Ĉ = (1/n)Σₖ(xₖ − μ̂)(xₖ − μ̂)ᵀ。先验估计为 Nⱼ/N。每一类用自己的样本分别估计。

### E8 ★ When is a Gaussian mixture model used, and how is the number of components chosen?
（什么时候用高斯混合模型？分量个数怎么选？）

> A GMM is used when the samples of a class are multimodal, i.e. generated by several distributions, so that a single Gaussian cannot model them. The class-conditional density is $p(x\mid\omega) = \sum_i \alpha_i\mathcal N(x\mid\mu_i, C_i)$ with $\sum\alpha_i = 1$. The parameters have no closed-form ML solution because the component memberships $\gamma_{ik}$ and the parameters depend on each other, so the EM algorithm is used: initialize randomly, compute $\gamma_{ik}$ (E-step), update $\alpha_i$, $\mu_i$, $C_i$ by weighted ML (M-step), and repeat until convergence. The number of components $m$ is a hyperparameter. Too many components make the decision boundary overly complex and cause overfitting; they also produce some very small weights $\alpha_i$, which indicate unnecessary components. $m$ can be reduced accordingly (re-running EM) or selected by validation.

> 【中文】当一类的样本是多峰的，即由多个分布生成、单个高斯无法刻画时，用 GMM。类条件密度为 p(x|ω) = Σᵢ αᵢ N(x|μᵢ, Cᵢ)，Σαᵢ = 1。参数没有闭式的最大似然解，因为分量归属 γᵢₖ 和参数相互依赖，所以用 EM 算法：随机初始化；计算 γᵢₖ（E 步）；用加权最大似然更新 αᵢ、μᵢ、Cᵢ（M 步）；重复直到收敛。分量个数 m 是超参数。分量太多会使决策边界过于复杂、导致过拟合，还会出现一些很小的权重 αᵢ，说明这些分量是多余的。可以据此减小 m（重新跑 EM），或用验证集选。

### E9 ★ Relationship between the Bayes classifier and LDA.
（Bayes 分类器与 LDA 的关系。）

> If both classes are Gaussian with a common covariance matrix $C$, the quadratic terms in $g_1(x) - g_2(x)$ cancel and the Bayes decision boundary is a hyperplane with normal vector $C^{-1}(m_1 - m_2)$. This is the same direction as Fisher's $w = S_W^{-1}(m_1 - m_2)$, since $S_W$ is proportional to the pooled covariance. LDA is therefore Bayes-optimal under these conditions. The two differ in the threshold: Bayes adds $\ln[P(\omega_1)/P(\omega_2)]$, whereas LDA's midpoint threshold implicitly assumes equal priors. When the covariances differ, the Bayes boundary is quadratic and LDA is no longer optimal.

> 【中文】如果两类都是高斯分布且协方差矩阵 C 相同，g₁(x) − g₂(x) 中的二次项抵消，Bayes 决策边界是法向量为 C⁻¹(m₁ − m₂) 的超平面。由于 S_W 与合并协方差成正比，这与 Fisher 的 w = S_W⁻¹(m₁ − m₂) 方向相同。所以在这些条件下 LDA 是 Bayes 最优的。两者的区别在阈值：Bayes 要加 ln[P(ω₁)/P(ω₂)]，而 LDA 的中点阈值隐含先验相等。协方差不同时，Bayes 边界是二次的，LDA 不再最优。

---

## 9. 自测清单（能不看讲义说出来，才算过关）

**第一级：总复习 p.2–6**
- [ ] 默写 Bayes 定理和 $p(x)$ 的展开式，标出 prior、class-conditional density、posterior。
- [ ] 默写多元高斯密度公式。
- [ ] 默写 Bayes 决策规则和联合概率形式的变体，并说出为什么可以不算 $p(x)$。
- [ ] 默写朴素贝叶斯公式，说出独立性假设。
- [ ] 说出三种 NB 分别用于什么特征。
- [ ] 会从总复习 p.6 那样的计数表算出先验和条件概率。

**第二级：计算**
- [ ] 课件 Bernoulli 例题（$4/45$ 对 $1/20$）合上书算一遍。
- [ ] 课件 Multinomial 例题（$2.765 \times 10^{-5}$ 对 $5.718 \times 10^{-6}$）合上书算一遍，说清 $d = 14$ 怎么来的。
- [ ] 完整 Bayes 的对数判别函数三项都写出，并会算 2×2 的行列式和逆。
- [ ] Gaussian NB：知道给散度矩阵时要除以样本数，只用对角元。
- [ ] 会写 "assumptions"：独立性、高斯、MLE 估计、先验按比例。

**第三级：理解**
- [ ] MLE 的思路和结果（除以 $n$）。
- [ ] 边界什么时候是二次曲线、什么时候是直线。
- [ ] GMM 为什么用 EM，$m$ 太大会怎样。
- [ ] NB 的优缺点各说 3 条。

**配套真题**：23-24 S1 Q3(b)、24-25 S1 Q2、24-25 S2 Q3(a)、25-26 S1 Q3(a)(b)。按 §6.1 的流程各手算一遍，再和参考答案核对。

**综合模拟卷**：[ML综合模拟卷.md](ML综合模拟卷.md)（各专题练习题已删除，统一用这张卷子练）。

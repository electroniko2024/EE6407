# ML 专题 01：Fisher Linear Discriminant Analysis（Fisher 线性判别分析，LDA）

> 依据：课件 **ML-Slides4.pdf**（共 58 页）、**总复习 PPT Revision2026S1.pdf 第 7–9 页**、**week8 录音稿**（LDA 全部内容）、**week13 录音稿**（老师逐页讲解总复习 PPT）、week9 开头（与 SVM 衔接）、六份期末卷参考答案。
> 约定：正文用中文，专业术语写英文并在括号里附中文；**可以直接写上答卷的句子用英文**。
> 符号：粗体向量省略，$x$ 是 $d$ 维列向量，$^T$ 表示转置。公式用 LaTeX 书写，需要支持数学公式的 Markdown 阅读器（Typora、VS Code、Obsidian 等）。

---

## 0. 这一章在考试里的分量（先看这个）

### 0.1 考频

| 卷子 | LDA 怎么考的 | 分值 |
|---|---|---|
| 23-24 S1 | Q3(a) 设计 LDA 分类器 + 分类测试样本 | 12 |
| 23-24 S2 | Q2(c) **LDA 与 SVM 的区别**（论述） | 7 |
| 24-25 S1 | Q3(a) 设计 + 分类 | 12 |
| 24-25 S2 | Q3(b) 设计 + Q3(c) **距离与置信度** | 10 + 5 |
| 25-26 S1 | Q2(a) **把 LDA 写成优化问题并解释原理**（论述）+ Q2(b) 设计 + Q2(c) 分类与置信度 | 12 + 8 + 5 |
| 25-26 S2 | Q2(a) 设计 + **最佳适用条件** + Q2(b) 距离与置信度 + Q2(c) **高维小样本为何失效、怎么补救** | 12 + 6 + 7 |

**结论：六份卷子都考了 LDA，其中五份有计算题。最近两年整道 Q2（25 分）都是 LDA，而且论述占比越来越大。**

### 0.2 老师在课上说过的话（week8 录音稿）

1. **考什么**："this part is important… normally at least tested in the exam in the past years"。考试**不能用电脑**，所以"we should use this formula"，也就是 $w = S_W^{-1}(m_1 - m_2)$，不要指望用特征值法。
2. **公式要会解释**：$m_1 - m_2$ 管类间（between-class），$S_W$ 管类内（within-class）。类内散度应该放在"分母"，但矩阵不能做除法，所以写成逆矩阵 $S_W^{-1}$。老师特别强调**用 $S_B$ 的逆肯定错**，因为那样就完全没考虑类内。
3. **$w_0$ 怎么来**："the middle point of the two mean vectors should be on the decision boundary"。这背后有个假设：两类的散度相同或相近（same or similar scatter）。假设不成立时要在算出来的值附近调整 $w_0$。
4. **置信度**：$\lvert g(x)\rvert/\lVert w\rVert$ 是样本到决策面的距离，距离越大越有把握。老师打的比方：40 分及格，估分 90 很放心，估分 41～42 就很悬。
5. **正则化**：$S_W$ 奇异（singular）时用 $S_W + \beta I$。即使 $S_W$ 不奇异，加这一项也能缓解过拟合（overfitting）。
6. **特征值法**：要选**非零特征值**对应的特征向量，而且**要检查符号**。往年有学生做作业时没检查，结果把整组测试样本的类别全判反了。
7. **不考的内容**：多类判别分析 MDA（Slides4 p.38–58）。学生课后问会不会考，老师答 "Normally no"。
8. **概念会反复用**："between-class difference / within-class scatter" 这对概念在**特征选择**和**聚类评价**里还会出现，老师希望大家 "have a very deep understanding of this concept"。

### 0.3 总复习 PPT（Revision2026S1）里的 LDA：只有 3 页，但每一笔都要会

总复习 PPT 一共 23 页，覆盖 9 个板块，LDA 是第 3 板块，占 p.7–9：

| 页 | 内容 | 对应本讲义 |
|---|---|---|
| p.7 | $g(x) = w^Tx + w_0$，配一张**投影示意图**：两类样本（○ 和 ×）沿虚线投影到 $w$ 方向的直线上，投影后上下分成两段 | §1.1、§2.1、§7.5 画图 |
| p.8 | $J(w) = \dfrac{\lvert\tilde m_1 - \tilde m_2\rvert^2}{\tilde s_1^2 + \tilde s_2^2} = \dfrac{(\tilde m_1 - \tilde m_2)^2}{\tilde s_1^2 + \tilde s_2^2}$，即 $J(w) = \dfrac{w^TS_Bw}{w^TS_Ww}$；$S_B = (m_1 - m_2)(m_1 - m_2)^T$；$S_i = \sum_{x\in D_i}(x - m_i)(x - m_i)^T$；$S_W = S_1 + S_2$ | §3.1、§3.2 |
| p.9 | $w = S_W^{-1}(m_1 - m_2)$；$w_0 = -\tfrac12 w^T(m_1 + m_2)$；配一张图：$m_1$、$m_2$ 连线的**中点（×）落在决策边界上**，并写出 $w^T\frac{m_1 + m_2}{2} + w_0 = 0$ | §3.4、§5、§7.5 |

（$D_i$ 是第 $i$ 类的样本集合，课件里写作 $x \in D_i$。）

**总复习 p.7 原图**：投影示意图。画法见 §7.5。

![Revision p.7](figs/lda/revision_p7.png)

**总复习 p.9 原图**：中点落在决策边界上。注意图里边界恰好垂直于 $m_1m_2$ 连线，这只是示意，见 §7.5 和图 7-2。

![Revision p.9](figs/lda/revision_p9.png)

**week13 老师讲这三页时说的话**：
- 求 $w$ 有两种方法："one is through the eigen-decomposition（录音里说成了 singular value decomposition），another is through a formula"。
- 公式**不要死记，要理解**（"not memorized, understand"）：$m_1 - m_2$ 是 between-class，$S_W$ 是 within-class，"we combine these two parts"。"If you use $S_B$… certainly is wrong"，因为 $S_B$ 只含类间。
- $w_0$ 的原理是"the middle point should be on the decision boundary"。假设是 "the spread of the two classes should be similar"，不相似时 "change the value around $w_0$, try a few values"。
- 这三条和 week8 讲的完全一致，说明老师认为这些就是 LDA 的核心。

**由此得到的复习优先级**：

| 级别 | 内容 | 理由 |
|---|---|---|
| **第一级：必须能默写和解释** | 总复习 p.7–9 的全部内容：$g(x)$、$J(w)$ 的两种写法、$S_B$、$S_i$、$S_W$、$w$、$w_0$ 及其原理，还有两张示意图 | 总复习专门列出，老师逐页再讲一遍 |
| **第二级：必须会做** | 距离 $\lvert g(x)\rvert/\lVert w\rVert$ 和置信度；$S_W$ 奇异与正则化；LDA 的适用条件；LDA vs SVM | 总复习 PPT **没有**列出，但真题反复考：距离和置信度考过 3 次，高维小样本、适用条件、与 SVM 对比各考过 1 次 |
| **第三级：理解即可** | 特征值法及其符号检查；Rayleigh quotient 求导；与 Bayes 的关系；LDA vs PCA | 课件或延伸内容，可以作为论述题的加分点 |
| **不用看** | MDA | 老师说 normally no，总复习也没有 |

> 要特别注意第二级：**总复习 PPT 只是提纲，没列出的内容照样考**。比如 25-26 S2 的"高维小样本"那 7 分，总复习里一个字都没有。

---

## 1. 线性判别函数的几何意义（Slides4 p.3–12）

这一节是后面算**距离和置信度**的基础。SVM 也用同一套几何。

### 1.1 Linear discriminant function（线性判别函数）

$$g(x) = w^Tx + w_0$$

- $w$：weight vector（权向量）
- $w_0$：bias / threshold weight（偏置、阈值）

**Decision rule（决策规则）**：
- $g(x) > 0$ 判为 $\omega_1$
- $g(x) < 0$ 判为 $\omega_2$
- $g(x) = 0$ 时样本落在 **decision boundary / decision surface（决策边界 / 决策面）**上。二维时它是一条直线，高维时是 **hyperplane（超平面）$H$**。

### 1.2 w 与超平面垂直（w is normal to the hyperplane）

设 $x_1$、$x_2$ 都在决策面上，则 $w^Tx_1 + w_0 = w^Tx_2 + w_0 = 0$，所以 $w^T(x_1 - x_2) = 0$。也就是说，$w$ 和超平面里的任意向量都垂直。

由此可得：
- $w$ 决定超平面的 **orientation（方向）**。
- $w$ 指向 $\omega_1$ 那一侧，称为 **positive side（正侧）**。
- $w_0$ 决定超平面的 **position（位置）**。

**看课件 Slides4 p.4 和 p.6**（下图左、右）：
- p.4：直线一侧 $g(x) > 0$，另一侧 $g(x) < 0$，线上 $g(x) = 0$。
- p.6：$x_1$、$x_2$ 是边界上的两个绿点，红色箭头 $w$ 和这条线垂直，指向 $\omega_1$（红点）一侧。

![Slides4 p.4 和 p.6](figs/lda/slides_p4_p6.png)

### 1.3 点到超平面的距离（Slides4 p.9–11，必背）

把 $x$ 分解成两部分：在超平面上的投影点 $x_p$，加上沿法向走的一段 $r\,\dfrac{w}{\lVert w\rVert}$：

$$x = x_p + r\,\frac{w}{\lVert w\rVert}$$

代入 $g$，因为 $g(x_p) = 0$，得到

$$g(x) = r\lVert w\rVert,\qquad r = \frac{g(x)}{\lVert w\rVert}$$

- $r$ 是 **signed distance（有符号距离）**：正值在正侧，负值在负侧。
- 原点到超平面的距离是 $r_0 = \dfrac{w_0}{\lVert w\rVert}$。
  - $w_0 > 0$：原点在正侧
  - $w_0 < 0$：原点在负侧
  - $w_0 = 0$：超平面过原点

下面这张图把这一节的几个量画在一起：
- 黑线是决策边界，紫色箭头 $w$ 垂直于它，指向 $g > 0$ 的粉色区域。
- 绿色双箭头是 $x$ 到边界的距离 $r$，$x_p$ 是 $x$ 在边界上的投影点（垂足）。
- 橙色双箭头是原点到边界的距离 $\lvert w_0\rvert/\lVert w\rVert$。这里 $w_0 < 0$，所以原点落在蓝色的负侧。

![图 1-1 几何意义](figs/lda/fig1_geometry.png)

**对照课件 Slides4 p.9–10**（下图）：p.9 是二维版，把 $x$ 分解成 $x_p$ 加一段 $r\,w/\lVert w\rVert$；p.10 是三维版，灰色平面就是超平面 $H$。

![Slides4 p.9 和 p.10](figs/lda/slides_p9_p10.png)

### 1.4 距离就是置信度（confidence）

老师原话的意思：样本离边界近，就算分对了，置信度也低；离边界远，置信度高。他用的比方是 40 分及格线，估分 90 和估分 41 的区别。

**答题时怎么写"置信度"**：

- 只写一个数字（比如 $r = 0.16$）拿不全分，因为距离要有**参照**才有意义。
- LDA 的参照：**类均值到决策面的距离 $\lvert g(m_i)\rvert/\lVert w\rVert$**。测试样本的 $r$ 远小于它，置信度就低；接近或超过它，置信度就高。
- SVM 的参照：间隔边界（margin），也就是比较 $\lvert g(x)\rvert$ 和 1。两者别混用。
- 画出来看最直观：§7.4 的图 7-2 里，橙色的 $x_a$ 几乎贴着边界（低置信），绿色的 $x_b$ 比 $m_1$ 离边界还远（高置信）。

可以直接套用的句子：

> The distance from $x$ to the decision boundary is $r = \lvert g(x)\rvert/\lVert w\rVert$ = ___. For comparison, the class mean $m_1$ lies at $\lvert g(m_1)\rvert/\lVert w\rVert$ = ___ from the boundary. Since $r$ is much smaller, $x$ is close to the boundary, so the classification is made with **low confidence**: a small perturbation of $x$ or an estimation error in $w$ could flip the decision.

---

## 2. Fisher 的思想：把高维样本投影到一条线上（Slides4 p.13–16）

### 2.1 Projection（投影）

$y = w^Tx$，把 $d$ 维样本 $x$ 投影成一个标量 $y$。

- **$w$ 的大小（magnitude）没有意义**，它只会把 $y$ 整体缩放；**只有方向（direction）重要**（Slides4 p.13）。
- 目标：找一个方向，使投影后两类 **well separated（分得开）**，而不是 **intermingled / mixed（混在一起）**（p.14）。

**看课件 Slides4 p.14**（下图）：同样两团样本，左图投影到竖直方向的 $w$ 上，两类的投影点各占一段；右图投影到另一个方向，两类的投影点混在一起。Fisher LDA 要做的就是**自动找出左图那样的方向**。

![Slides4 p.14](figs/lda/slides_p14.png)

### 2.2 两个要求（老师在 week8 讲得很细）

| 要求 | 衡量方式 | 希望 |
|---|---|---|
| **Between-class separation（类间分离）** | 投影后两类均值之差 $\lvert\tilde m_1 - \tilde m_2\rvert$ | 越**大**越好 |
| **Within-class scatter（类内散度）** | 投影后每类的 scatter $\tilde s_i^2$ | 越**小**越好 |

- 投影后的类均值：$\tilde m_i = \dfrac{1}{N_i}\sum w^Tx = w^Tm_i$，即投影后的均值等于均值的投影。
- 投影后的类内散度：$\tilde s_i^2 = \sum_{x\in\omega_i} (w^Tx - \tilde m_i)^2$

老师特别说明：**scatter（散度）就是不除以样本数的方差**。除以 $N_i$ 才是 variance（方差）或 covariance matrix（协方差矩阵）。

**为什么只看均值差不够？** 如果某个方向上均值差很大，但类内方差也很大，投影后两类照样重叠。所以要把"均值差"放到"类内散度"的尺度下来比较。这就是下面这个比值的来历。

下图用程序生成的数据演示这一点（课件没有这张图，是帮助理解的；论述题画草图时可以照着画）：
- **(a)** 两类都是斜着拉长的椭圆。灰线是 $m_1 - m_2$ 的方向，绿线是 Fisher 方向 $S_W^{-1}(m_1 - m_2)$。
- **(b)** 投影到灰线上：两个均值离得不近，但每类沿这个方向都铺得很开（类内散度大），直方图大面积重叠，$J = 0.0062$。
- **(c)** 投影到绿线上：均值差变化不大，但每类都挤成窄窄一条，两类基本分开，$J = 0.0394$，约为 (b) 的 6 倍。

![图 2-1 均值差方向 vs Fisher 方向](figs/lda/fig2_fisher_vs_mean.png)

**看这张图要记住一句话**：Fisher 方向通常**不是**两个均值的连线方向，它会往"类内散度小"的方向偏。

---

## 3. Fisher criterion（Fisher 准则）的推导（Slides4 p.17–22）

> **25-26 S1 Q2(a)（12 分）原题考的就是这一节**："formulate LDA as an optimization problem and explain the rationale"。推导主线要能默写。

### 3.1 准则函数

$$\max_w\; J(w) = \frac{(\tilde m_1 - \tilde m_2)^2}{\tilde s_1^2 + \tilde s_2^2}$$

分子越大越好（类间），分母越小越好（类内），老师的话是 "combine these two requirements into one single loss function"。

### 3.2 用矩阵写出分子和分母

**Scatter matrix（散度矩阵）**：

$$S_i = \sum_{x\in\omega_i} (x - m_i)(x - m_i)^T$$

**Within-class scatter matrix（类内散度矩阵）**：

$$S_W = S_1 + S_2$$

**分母**：
- $\tilde s_i^2 = \sum (w^Tx - w^Tm_i)^2 = \sum w^T(x - m_i)(x - m_i)^Tw = w^TS_iw$
- 所以 $\tilde s_1^2 + \tilde s_2^2 = w^TS_W w$

**分子**：
- $(\tilde m_1 - \tilde m_2)^2 = (w^Tm_1 - w^Tm_2)^2 = w^T(m_1 - m_2)(m_1 - m_2)^Tw = w^TS_B w$
- 其中 **between-class scatter matrix（类间散度矩阵）** $S_B = (m_1 - m_2)(m_1 - m_2)^T$

代回得到：

$$J(w) = \frac{w^TS_B w}{w^TS_W w}$$

这个式子叫 **generalized Rayleigh quotient（广义瑞利商）**。

### 3.3 求最大值：广义特征值问题

对 $w$ 求导并令导数为 0：

$$(w^TS_W w)\,S_B w - (w^TS_B w)\,S_W w = 0 \;\Longrightarrow\; S_B w = J(w)\,S_W w$$

所以最优的 $w$ 满足

$$S_B w = \lambda S_W w$$

这是 **generalized eigenvalue problem（广义特征值问题）**。若 $S_W$ **non-singular（非奇异）**，两边左乘 $S_W^{-1}$，就变成普通的 **eigenvalue problem（特征值问题）**：

$$S_W^{-1}S_B w = \lambda w$$

> 课件只说"must satisfy"，没写求导过程。答卷上写出上面这一行求导，就能说明为什么是特征值问题，论述题能多拿分。
>
> 由这一行还能看出两件事：
> 1. **$\lambda$ 就等于 $J(w)$ 本身**，所以应该取**最大特征值**对应的特征向量。
> 2. 两类问题里 $S_B$ 的秩（rank）为 1，所以只有一个非零特征值，另一个是 0。课件 p.33 的例子正是这样：特征值为 0.000 的那个特征向量 "has no discriminative power"。

### 3.4 不解特征值也能直接得到 w（Slides4 p.22，考试用这个）

$S_B w = (m_1 - m_2)\,[(m_1 - m_2)^Tw]$。方括号里是一个标量，记为 $\rho$，所以 **$S_B w$ 永远与 $(m_1 - m_2)$ 同方向**。

代入 $S_B w = \lambda S_W w$ 得 $(m_1 - m_2)\rho = \lambda S_W w$。$\rho$ 和 $\lambda$ 都只是比例因子（scaling factor），而我们只关心方向，所以去掉它们：

$$\boxed{\,w = S_W^{-1}(m_1 - m_2)\,}$$

老师对这个公式的直观解释（论述题可以直接用）：

- **$m_1 - m_2$**：类间差异，希望它大，所以放"分子"。
- **$S_W^{-1}$**：类内散度希望小，本该放"分母"，但矩阵不能做除法，所以用逆矩阵。
- **用 $S_B^{-1}$ 是错的**：那样只考虑了类间，类内完全没进公式。何况 $S_B$ 的秩为 1，本身就不可逆。

**一个小算例，说明 $S_W^{-1}$ 起了什么作用。** 设 $m_1 - m_2 = [1, 1]^T$，$S_W = \mathrm{diag}(10, 100)$，也就是第 2 个特征的类内散度大得多。

| 方向 | 分子 $(w^T(m_1-m_2))^2$ | 分母 $w^TS_W w$ | $J(w)$ |
|---|---|---|---|
| 只看均值差：$w = [1, 1]$ | 4 | 110 | 0.036 |
| Fisher：$w = S_W^{-1}(m_1-m_2) = [0.1, 0.01]$ | 0.0121 | 0.11 | **0.110** |

Fisher 方向的 $J$ 大约是前者的 3 倍。它把权重压到"类内噪声大"的第 2 个特征上，更依赖类内更紧凑的第 1 个特征。**这就是"为什么要除以类内散度"最好的例证**，论述题可以用。

---

## 4. Regularization（正则化）：S_W 奇异怎么办（Slides4 p.23）

### 4.1 什么时候会奇异

$S_W$ 是 $N$ 个外积（outer product）之和再减去均值的影响，所以 $\mathrm{rank}(S_W) \le N - 2$（两类情形）。

- 维数 $d$ 大、样本数 $N$ 少（**high-dimensional, small sample size**）时，若 $d > N - 2$，$S_W$ 一定是 **singular（奇异）**，求不了逆。
- 即使 $d$ 略小于 $N$，$S_W$ 也会接近奇异（ill-conditioned，病态）。求逆会把小特征值的估计误差成倍放大，$w$ 对噪声非常敏感，结果是**训练集上好、测试集上差**，也就是过拟合。

### 4.2 补救

$$S_W \leftarrow S_W + \beta I$$

其中 $\beta$ 是一个小正数（课件举例 0.001），$I$ 是单位矩阵（identity matrix）。

- 作用：相当于给对角线每个元素都加上 $\beta$，矩阵变成正定、可逆。
- 老师补充：即使 $S_W$ 不奇异也可以加这一项，因为它是一种 regularization，能**缓解过拟合**，提高测试集上的表现。他把它和"在损失函数里加惩罚项、约束 $w$ 的大小"类比。
- $\beta$ 的选法（课件外补充）：用交叉验证（cross-validation）选。
- 其他补救（课件外补充，论述题可以列出）：
  - 先降维，例如用 PCA 或特征选择（feature selection）。
  - 用伪逆（pseudo-inverse）。
  - 假设 $S_W$ 是对角阵，忽略特征间的相关性。
  - 增加训练样本。

---

## 5. Threshold w₀（阈值）怎么定（Slides4 p.24–25）

### 5.1 公式

原理：**两类均值的中点 $\frac{m_1 + m_2}{2}$ 应该落在决策面上**，即 $g\!\left(\frac{m_1 + m_2}{2}\right) = 0$，所以

$$w_0 = -\frac12\, w^T(m_1 + m_2)$$

**看课件 Slides4 p.24**（下图）：红点 $m_1$、蓝点 $m_2$ 的连线中点（×）落在决策边界上，代入 $g = 0$ 就解出 $w_0$。总复习 p.9 的图就是这一张。

![Slides4 p.24](figs/lda/slides_p24.png)

### 5.2 隐含的假设（论述题高频）

课件 p.25 和老师都强调：这个做法默认**两类投影后的散度相同或相近（same or similar scatter）**。

- 如果一类很紧凑、另一类很分散，边界就不该放在正中间，而应该往紧凑那一类靠。
- 老师给的做法：以公式算出的 $w_0$ 为基准，在附近试几个值（比如算出 1.2，就试 1.1、1.15、1.25、1.3），看在训练集或验证集（validation data）上哪个表现最好。

下图是投影到 $w$ 上之后的一维分布（课件没有这张图）：
- 红色 $\omega_1$ 很紧凑，蓝色 $\omega_2$ 很分散。
- 黑色虚线是中点阈值 2.0。蓝色斜线区是 $\omega_2$ 落在虚线左边、被错判成 $\omega_1$ 的那部分。
- 绿线大约在 1.1，是两条曲线的交点，错误更少。它比中点**更靠近紧凑的那一类**。

![图 5-1 散度不等时的阈值](figs/lda/fig5_threshold.png)

### 5.3 与先验的关系（课件外延伸，考到"先验不等"时用）

- 中点阈值相当于**先验相等**。
- 如果把 LDA 看成"各类为高斯分布且协方差相同时的 Bayes 分类器"（见 §9 E8），先验不等时阈值还要加上 $\ln\dfrac{P(\omega_1)}{P(\omega_2)}$ 的修正，边界会往先验小的那一类移。
- 考试**按课件用中点法**算，必要时加一句说明即可。23-24 S1 两类样本数是 600 和 400，参考答案就是这样处理的。

---

## 6. 两种求 w 的方法对比（Slides4 p.32–37）

| | 直接公式 $w = S_W^{-1}(m_1-m_2)$ | 特征值法 $S_W^{-1}S_B w = \lambda w$ |
|---|---|---|
| 需要 $S_B$ 吗 | 不需要 | 需要 |
| 能手算吗 | 能，2×2 求逆即可 | 一般要用 MATLAB 的 `eig` |
| 考试用哪个 | **用这个** | 不用 |
| 符号（sign） | 保证 $\omega_1$ 在正侧 | **不保证**，要自己检查 |

**两种方法求出的 $w$ 看起来完全不一样，为什么都对？** 只有方向有意义，比较两个分量的比值 $w_1/w_2$，它们是一样的（Slides4 p.36–37）。

**符号检查**（老师强调过）：把类 1 的样本或 $m_1$ 代入 $g(x)$，如果结果是负的，就用 $-w$ 和 $-w_0$ 替换原来的 $w$ 和 $w_0$，保持"正值判 $\omega_1$"这条规则不变。

**特征值法要选哪个特征向量？** 选非零特征值，也就是最大特征值对应的那个（见 §3.3）。选特征值为 0 的那个，投影后两类完全混在一起，不能用来分类。

**看课件 Slides4 p.33 和 p.34**（下图左、右）：横轴是样本序号（前 100 个是类 1，后 100 个是类 2），纵轴是 $g(x)$。

- 左图（p.33）用特征值 0.000 的特征向量：前后两半的 $g(x)$ 没有区别，"has no discriminative power"。
- 右图（p.34）用特征值 0.058 的特征向量：前一半整体偏高，后一半整体偏低，两类分开了。
- 特征值法求出的向量正负号不确定。如果类 1 的 $g(x)$ 整体是负的，就要按课件 p.36 把 $w$ 和 $w_0$ 同时取反，这就是符号检查。

![Slides4 p.33 和 p.34](figs/lda/slides_p33_p34.png)

**对照课件 Slides4 p.30 和 p.31**（下图）：直接公式法的结果。p.30 是每个样本的 $g(x)$，前 100 个为正、后 100 个为负；p.31 是平面上的决策边界，黄绿两个区域。p.35 用特征值法得到的边界和 p.31 一样，说明两种方法方向相同。

![Slides4 p.30 和 p.31](figs/lda/slides_p30_p31.png)

---

## 7. 计算题标准流程 + 两道例题

### 7.1 六步走（写全了就有步骤分）

1. **均值**：$m_i = \dfrac{1}{N_i}\sum x$。题目已给就跳过。
2. **散度**：$S_i = \sum (x - m_i)(x - m_i)^T$，然后 $S_W = S_1 + S_2$。
3. **求逆**：2×2 矩阵的逆
   $$\begin{bmatrix} a & b\\ c & d\end{bmatrix}^{-1} = \frac{1}{ad - bc}\begin{bmatrix} d & -b\\ -c & a\end{bmatrix}$$
   即主对角线互换、副对角线变号。
4. $w = S_W^{-1}(m_1 - m_2)$
5. $w_0 = -\tfrac12 w^T(m_1 + m_2)$，然后写出 $g(x) = w_1x_1 + w_2x_2 + w_0$ 和决策规则。
6. **分类 + 置信度**：代入 $g$，判断正负，算 $r = \lvert g\rvert/\lVert w\rVert$，再和 $\lvert g(m_i)\rvert/\lVert w\rVert$ 比较。

### 7.2 常见失分点

- **题目给的是 scatter 还是 covariance？**
  - 给 scatter：直接相加。
  - 给 covariance 而且两类样本数不同：要先还原成 $S_i = N_i\Sigma_i$ 再相加（若是无偏估计，用 $(N_i - 1)\Sigma_i$）。
  - 只有 $N_1 = N_2$ 时，直接用 $\Sigma_1 + \Sigma_2$ 才不影响方向。
  - 这一点和 Gaussian NB 正好相反：NB 要用方差，所以要用 scatter 除以 $N$。
- **$m_1 - m_2$ 的顺序别写反**，否则整条决策规则会反号。
- **$w$ 的数量级很小**（$10^{-2}$ 到 $10^{-3}$）是正常的，因为 $S_W$ 是没除以 $N$ 的散度，数值大。
  - **中间结果至少保留 4 位有效数字**，否则 $g(x)$ 的符号都可能算错。
  - 技巧：先算 $\mathrm{adj}(S_W)(m_1 - m_2)$，最后再除以 $\det$。$\det > 0$，所以除完方向不变。
- **置信度只写数字不下结论**：一定要写出 "high / low confidence" 和理由。

### 7.3 热身例题：从原始样本算起

**题目**：$\omega_1$ 的样本为 $(1,1)$、$(2,3)$、$(3,2)$；$\omega_2$ 的样本为 $(5,4)$、$(6,6)$、$(7,5)$。设计 Fisher 分类器，并对 $x = (3, 4)$ 分类。

**步骤 1：均值**
- $m_1 = (2, 2)$
- $m_2 = (6, 5)$

**步骤 2：散度**

$\omega_1$ 各样本减去 $m_1$ 后的偏差（deviation）为 $(-1,-1)$、$(0,1)$、$(1,0)$，所以

$$S_1 = \begin{bmatrix}1&1\\1&1\end{bmatrix} + \begin{bmatrix}0&0\\0&1\end{bmatrix} + \begin{bmatrix}1&0\\0&0\end{bmatrix} = \begin{bmatrix}2&1\\1&2\end{bmatrix}$$

$\omega_2$ 的偏差恰好相同，所以 $S_2 = \begin{bmatrix}2&1\\1&2\end{bmatrix}$，$S_W = \begin{bmatrix}4&2\\2&4\end{bmatrix}$。

**步骤 3：求逆**
- $\det = 16 - 4 = 12$
- $S_W^{-1} = \dfrac{1}{12}\begin{bmatrix}4&-2\\-2&4\end{bmatrix}$

**步骤 4：w**
- $m_1 - m_2 = (-4, -3)$
- $w = \dfrac{1}{12}(-16 + 6,\; 8 - 12) = \dfrac{1}{12}(-10, -4) = (-0.8333, -0.3333)$

**步骤 5：$w_0$**
- $m_1 + m_2 = (8, 7)$
- $w_0 = -\tfrac12(-0.8333\times 8 - 0.3333\times 7) = -\tfrac12(-9) = 4.5$
- $g(x) = -0.8333x_1 - 0.3333x_2 + 4.5$；$g > 0$ 判 $\omega_1$，$g < 0$ 判 $\omega_2$
- 决策边界：$x_2 = -2.5x_1 + 13.5$，经过中点 $(4, 3.5)$

**步骤 6：分类与置信度**

- 自检：$g(m_1) = 2.167 > 0$，$\omega_1$ 确实在正侧。
- $g(3, 4) = -2.5 - 1.333 + 4.5 = 0.667 > 0$，**判为 $\omega_1$**
- $\lVert w\rVert = \sqrt{0.6944 + 0.1111} = 0.8975$
- $r = 0.667/0.8975 = 0.743$
- 类均值到边界的距离：$\lvert g(m_1)\rvert/\lVert w\rVert = 2.167/0.8975 = 2.414$
- 0.743 只有类均值距离的 30% 左右，样本偏向边界，**置信度中等偏低**。

![图 7-1 热身例题](figs/lda/fig7_warmup.png)

图里可以看出：这个例子 $S_1 = S_2$ 且两个特征散度相同，所以决策边界（斜率 $-2.5$）和 $m_1m_2$ 连线（斜率 0.75）**并不垂直**（垂直的话斜率应为 $-1.333$）。原因是 $S_W$ 的非对角元 2 表示两个特征相关，$S_W^{-1}$ 把方向扭了一下。

### 7.4 考试格式例题：给定散度矩阵（数值已用程序核对）

**题目**：已知
- $m_1 = [1.0, 2.0]^T$，$m_2 = [3.0, 0.5]^T$
- $S_1 = \begin{bmatrix}40&12\\12&30\end{bmatrix}$，$S_2 = \begin{bmatrix}36&10\\10&44\end{bmatrix}$

(a) 设计 Fisher LDA 分类器。
(b) 对 $x_a = [2.0, 1.5]^T$ 和 $x_b = [0, 3]^T$ 分类，并说明置信度。

**(a) 设计**
1. $S_W = S_1 + S_2 = \begin{bmatrix}76&22\\22&74\end{bmatrix}$
2. $\det = 76\times 74 - 22^2 = 5624 - 484 = 5140$
3. $S_W^{-1} = \dfrac{1}{5140}\begin{bmatrix}74&-22\\-22&76\end{bmatrix} = \begin{bmatrix}0.014397&-0.004280\\-0.004280&0.014786\end{bmatrix}$
4. $m_1 - m_2 = [-2, 1.5]^T$
   - $w = \dfrac{1}{5140}\begin{bmatrix}74\times(-2) - 22\times 1.5\\ -22\times(-2) + 76\times 1.5\end{bmatrix} = \dfrac{1}{5140}\begin{bmatrix}-181\\158\end{bmatrix} = \begin{bmatrix}-0.035214\\0.030739\end{bmatrix}$
5. $m_1 + m_2 = [4, 2.5]^T$
   - $w_0 = -\tfrac12(-0.035214\times 4 + 0.030739\times 2.5) = -\tfrac12(-0.064008) = 0.032004$
6. $g(x) = -0.035214x_1 + 0.030739x_2 + 0.032004$；$g > 0$ 判 $\omega_1$，$g < 0$ 判 $\omega_2$
   - 决策边界：$x_2 = 1.1456x_1 - 1.0411$
   - $\lVert w\rVert = 0.046743$

**(b) 分类与置信度**

| | $g(x)$ | 判为 | $r = \lvert g\rvert/\lVert w\rVert$ | 和类均值距离 1.2466 比较 | 置信度 |
|---|---|---|---|---|---|
| $m_1$（自检） | +0.058268 | $\omega_1$ ✓ | 1.2466 | 参照 | — |
| $x_a = [2.0, 1.5]$ | +0.007685 | $\omega_1$ | **0.164** | 约为 13% | **低**：几乎贴着边界 |
| $x_b = [0, 3]$ | +0.124222 | $\omega_1$ | **2.658** | 约为 2.1 倍 | **高**：比类中心离边界还远 |

![图 7-2 §7.4 例题](figs/lda/fig7_2_exam_example.png)

怎么看这张图：
- 黑实线是决策边界，经过中点 $(2, 1.25)$。紫色箭头 $w$ 指向 $\omega_1$ 一侧。
- 三条彩色短线是到边界的垂直距离：红色是 $m_1$（1.2466，作为参照），橙色是 $x_a$（0.164，很短），绿色是 $x_b$（2.658，比红线还长）。
- 灰色虚线是"垂直于 $m_1m_2$ 连线"的直线，它和真正的边界不重合，这就是 §7.5 说的总复习 p.9 示意图的坑。

**附：用特征值法验算（考试不用做，帮助理解 §3.3 和 §6）**

- $S_W^{-1}S_B$ 的特征值是 **0.1165 和 0**，正好对应"两类问题里 $S_B$ 的秩为 1"。
- 0.1165 恰好等于 $J(w)$，印证了 $\lambda = J$。
- 最大特征值对应的特征向量是 $v = [0.7534, -0.6576]^T$，分量比 $-1.1456$，和 $w$ 的分量比相同。所以**方向相同，但符号相反**。
- 用 $v$ 算出的 $g(m_1) = -1.2466 < 0$，必须换成 $-v$ 和 $-w_0$。距离 1.2466 不变，因为距离与 $w$ 的缩放无关。

### 7.5 画图（总复习 p.7、p.9 两张图）

论述题写 "use a sketch" 时，或计算题想让答案更清楚时，就画这两张图。

两张原图见 §0.3。另外两张可以参考着画：
- 想说明"为什么要除以类内散度"，照着图 2-1 的 (a) 画两个方向即可。
- 计算题画决策边界，照着图 7-2 画。

**图 1：投影示意图（总复习 p.7）**，用来说明 Fisher 的思想。
1. 画坐标轴 $x_1$、$x_2$，再画两团样本（一团 ○、一团 ×），各圈一个椭圆。
2. 从原点画一条带箭头的直线，标上 $w$。
3. 从几个样本向这条直线画虚线投影，投影点画在直线上。
4. 标注：两类的投影各自聚在一段，而且两段分开，"well separated"。
5. 想说明"为什么要除以类内散度"，就在旁边再画一条差的方向：投影后两段重叠，标 "intermingled"。

**图 2：阈值示意图（总复习 p.9）**，用来说明 $w_0$ 的原理。
1. 标出 $m_1$、$m_2$，用虚线连起来。
2. 在中点打 ×。
3. 画一条经过 × 的直线作为决策边界，写上 $w^T\frac{m_1 + m_2}{2} + w_0 = 0$。

**这张图有个容易画错的地方。** 总复习 p.9 里，决策边界恰好和 $m_1m_2$ 连线**垂直**。那只是示意，只有 $S_W \propto I$（两个特征类内散度相同且不相关）时才成立。一般情况下：

- 决策边界**一定经过中点**，这一点永远成立。
- 决策边界**垂直于 $w = S_W^{-1}(m_1 - m_2)$**，而不是垂直于 $m_1 - m_2$。

用 §7.4 的数字验证：
- $m_1 - m_2 = (-2, 1.5)$，垂直于它的直线斜率是 $4/3 \approx 1.333$。
- 实际决策边界 $x_2 = 1.1456x_1 - 1.0411$，斜率是 1.1456。
- 两者不同，差别就来自 $S_W^{-1}$ 的修正。

**计算题里怎么画决策边界**（以 §7.4 为例）：
1. 找三个点：中点 $(2, 1.25)$；与 $x_2$ 轴的交点 $(0, -1.041)$；与 $x_1$ 轴的交点 $(0.909, 0)$。连成直线。
2. 标出 $m_1$、$m_2$ 和测试点，并画一个箭头表示 $w$，指向 $\omega_1$ 一侧（$g > 0$ 的一侧）。
3. 一眼就能看出 $x_a$ 贴着边界、$x_b$ 离边界很远，和计算出的置信度结论一致。

---

## 8. Multiple Discriminant Analysis（多类判别分析，MDA）：只需知道三句话

老师说 "normally no"，不用会算。但论述题提到"LDA 用于降维"时可以引用：

1. $c$ 类问题需要 **$c - 1$ 个判别函数**，投影把 $d$ 维空间映到 $c - 1$ 维，所以 LDA 也是一种 **supervised dimensionality reduction（有监督降维）**方法。
2. $S_B$ 改用各类均值相对**总均值（total mean vector）**来定义：$S_B = \sum_i N_i(m_i - m)(m_i - m)^T$。准则改用散度矩阵的**行列式（determinant）**之比 $\lvert\tilde S_B\rvert/\lvert\tilde S_W\rvert$。
3. 由于 $\mathrm{rank}(S_B) \le c - 1$，**最多只有 $c - 1$ 个有用的特征向量**。

---

## 9. 论述题专项（本章重点）

> **答题模板 D-E-E-C**：Define（定义）→ Explain（机制、公式中每项的含义）→ Evaluate（前提、优缺点）→ Conclude（回到题目情境）。
> 分值 ÷ 2 ≈ 要写的要点数。12 分题至少写 6 个得分点。

### E1 ★★★ Formulate Fisher LDA as an optimization problem and explain the rationale.（25-26 S1，12 分）
（把 Fisher LDA 表述为一个优化问题，并解释其原理。）

**得分点**（每点约 1.5～2 分）：
1. 投影 $y = w^Tx$，目标是投影后两类分开。
2. 投影均值 $\tilde m_i = w^Tm_i$ 和投影散度 $\tilde s_i^2$ 的定义。
3. $J(w)$ 的标量形式。
4. 矩阵形式：$S_W$、$S_B$ 的定义，$J = \dfrac{w^TS_Bw}{w^TS_Ww}$。
5. 求解：$S_Bw = \lambda S_Ww$，再到 $w = S_W^{-1}(m_1-m_2)$。
6. **Rationale**：为什么用比值、每一项代表什么、为什么 $w$ 只有方向有意义。
7. $w_0$ 和决策规则。

**范文**：

> **Idea.** Fisher LDA projects each $d$-dimensional sample onto a line, $y = w^Tx$, and looks for the direction $w$ along which the two classes are best separated. Only the direction of $w$ matters; its magnitude merely scales $y$.
>
> **Formulation.** After projection, the class means are $\tilde m_i = w^Tm_i$ and the within-class scatters are $\tilde s_i^2 = \sum_{x\in\omega_i}(w^Tx - \tilde m_i)^2$. LDA solves
>
> $$\max_w J(w) = \frac{(\tilde m_1 - \tilde m_2)^2}{\tilde s_1^2 + \tilde s_2^2} = \frac{w^TS_B w}{w^TS_W w},$$
>
> where $S_B = (m_1 - m_2)(m_1 - m_2)^T$ is the between-class scatter matrix and $S_W = S_1 + S_2$, with $S_i = \sum_{x\in\omega_i}(x - m_i)(x - m_i)^T$, is the within-class scatter matrix. $J(w)$ is a generalized Rayleigh quotient.
>
> **Solution.** Setting $\partial J/\partial w = 0$ gives $S_B w = \lambda S_W w$ ($\lambda = J(w)$), a generalized eigenvalue problem; if $S_W$ is non-singular, $S_W^{-1}S_B w = \lambda w$. Because $S_B w$ is always in the direction of $(m_1 - m_2)$, the solution is $w = S_W^{-1}(m_1 - m_2)$ up to a scale factor. The threshold $w_0 = -\tfrac12 w^T(m_1 + m_2)$ places the boundary at the midpoint of the projected means, and $x$ is assigned to $\omega_1$ if $g(x) = w^Tx + w_0 > 0$, otherwise to $\omega_2$.
>
> **Rationale.** The numerator measures between-class separation, which should be large; the denominator measures within-class spread, which should be small. Maximizing the mean difference alone is not enough: if the classes are widely spread along that direction they still overlap. The ratio balances the two requirements in a single criterion and yields the direction with the least overlap. In the solution, $(m_1 - m_2)$ accounts for between-class separation and $S_W^{-1}$ acts as the "denominator" that down-weights directions with large within-class scatter.

> 【中文】
> **思想。** Fisher LDA 把每个 d 维样本投影到一条直线上，y = wᵀx，寻找使两类分得最开的方向 w。只有 w 的方向有意义，它的大小只是把 y 缩放。
>
> **表述。** 投影后，两类的均值为 m̃ᵢ = wᵀmᵢ，类内散度为 s̃ᵢ² = Σ_{x∈ωᵢ}(wᵀx − m̃ᵢ)²。LDA 求解
>
> max_w J(w) = (m̃₁ − m̃₂)² / (s̃₁² + s̃₂²) = wᵀS_B w / wᵀS_W w，
>
> 其中 S_B = (m₁ − m₂)(m₁ − m₂)ᵀ 是类间散度矩阵，S_W = S₁ + S₂（Sᵢ = Σ_{x∈ωᵢ}(x − mᵢ)(x − mᵢ)ᵀ）是类内散度矩阵。J(w) 是广义瑞利商。
>
> **求解。** 令 ∂J/∂w = 0，得 S_B w = λS_W w（λ = J(w)），这是广义特征值问题；若 S_W 非奇异，则 S_W⁻¹S_B w = λw。由于 S_B w 总与 (m₁ − m₂) 同方向，解为 w = S_W⁻¹(m₁ − m₂)（相差一个比例因子）。阈值 w₀ = −½wᵀ(m₁ + m₂) 把边界放在两类投影均值的中点；若 g(x) = wᵀx + w₀ > 0，则 x 判为 ω₁，否则判为 ω₂。
>
> **原理。** 分子衡量类间分离程度，应当大；分母衡量类内离散程度，应当小。只最大化均值差是不够的：如果两类沿该方向分布很散，仍会重叠。比值把两个要求合成一个准则，得到重叠最少的方向。在解中，(m₁ − m₂) 体现类间分离，S_W⁻¹ 起"分母"的作用，降低类内散度大的方向的权重。

### E2 ★★★ Why does the LDA solution contain both (m₁ − m₂) and S_W⁻¹? Why not use only the mean difference, or S_B⁻¹?
（为什么 LDA 的解里既有 (m₁ − m₂) 又有 S_W⁻¹？为什么不只用均值差，或者用 S_B⁻¹？）

> $(m_1 - m_2)$ is the between-class term; $S_W^{-1}$ is the within-class term. Using only $w = m_1 - m_2$ ignores the shape of the class distributions: along a feature with large within-class scatter the projected classes overlap even if the means differ. $S_W^{-1}$ rotates and rescales the mean-difference direction so that features with small within-class scatter (more reliable) receive larger weights. Using $S_B^{-1}$ would be wrong: it ignores within-class scatter completely, and $S_B = (m_1 - m_2)(m_1 - m_2)^T$ has rank 1, so it is not even invertible.

> 【中文】(m₁ − m₂) 是类间项，S_W⁻¹ 是类内项。只用 w = m₁ − m₂ 忽略了各类分布的形状：在类内散度大的特征上，即使均值不同，投影后两类也会重叠。S_W⁻¹ 对均值差方向做旋转和缩放，使类内散度小（更可靠）的特征获得更大的权重。用 S_B⁻¹ 是错的：它完全忽略了类内散度，而且 S_B = (m₁ − m₂)(m₁ − m₂)ᵀ 的秩为 1，根本不可逆。

可以补一个 §3.4 的数值例子：两个方向的 $J$ 分别是 0.036 和 0.110。论述里有具体数字，阅卷人会认为你真懂。

### E3 ★★★ How is the threshold w₀ determined? What assumption is behind it? What if it does not hold?
（阈值 w₀ 怎么确定？背后有什么假设？假设不成立时怎么办？）

> $w_0$ is chosen so that the midpoint of the two class means lies on the decision boundary: $g\!\left(\frac{m_1 + m_2}{2}\right) = 0$, hence $w_0 = -\tfrac12 w^T(m_1 + m_2)$. This assumes that the projected samples of the two classes have the same or similar scatter (and, implicitly, equal priors), so the optimal boundary is halfway between the projected means. If one class is much more spread out than the other, or the class sizes or priors differ greatly, the midpoint is no longer optimal. We then take the computed $w_0$ as a starting point and try a few nearby values, choosing the one that gives the best accuracy on training or validation data. Alternatively, under a Gaussian model $w_0$ can include $\ln[P(\omega_1)/P(\omega_2)]$.

> 【中文】选 w₀ 使两类均值的中点落在决策边界上：g((m₁ + m₂)/2) = 0，所以 w₀ = −½wᵀ(m₁ + m₂)。这假设两类投影后的样本散度相同或相近（并隐含先验相等），因此最优边界在两个投影均值的正中间。如果一类比另一类分散得多，或者两类样本数、先验差别很大，中点就不再最优。这时以算出的 w₀ 为起点，在附近试几个值，选在训练集或验证集上准确率最高的那个。另一种做法是在高斯模型下，给 w₀ 加上 ln[P(ω₁)/P(ω₂)]。

### E4 ★★★ Under what conditions does LDA perform best? When does it fail?（25-26 S2）
（LDA 在什么条件下表现最好？什么时候会失效？）

**最佳条件**：

> LDA performs best when:
> 1. Each class is (approximately) Gaussian with a **common covariance matrix**. The Bayes-optimal boundary is then linear, and LDA's direction coincides with the Bayes classifier.
> 2. The classes differ mainly in their **means**. LDA's criterion is driven by $m_1 - m_2$.
> 3. Each class is **unimodal** and the classes are (nearly) **linearly separable**.
> 4. There are **enough samples relative to the dimension** ($N \gg d$), so that $S_W$ is well estimated and invertible.
> 5. The two classes have similar scatter, so the midpoint threshold is appropriate.

> 【中文】LDA 在以下情况表现最好：
> 1. 每一类（近似）服从高斯分布，且**协方差矩阵相同**。此时 Bayes 最优边界是线性的，LDA 的方向与 Bayes 分类器一致。
> 2. 两类主要在**均值**上不同。LDA 的准则由 m₁ − m₂ 驱动。
> 3. 每一类都是**单峰**的，两类（近似）**线性可分**。
> 4. **样本数相对维数足够多**（N ≫ d），S_W 估计得准且可逆。
> 5. 两类散度相近，中点阈值才合适。

**会失效的情形**（每条对应上面一个条件）：

> It fails when:
> - The classes have equal or close means but different variances (e.g. one class inside a ring of the other). Then $m_1 - m_2 \approx 0$ and $J(w) \approx 0$ in every direction.
> - A class is multimodal (e.g. XOR), so a single linear boundary cannot separate the classes.
> - The covariances differ strongly. The optimal boundary is then quadratic, as in the Gaussian Bayes classifier.
> - $d$ is large relative to $N$, so $S_W$ is singular or ill-conditioned (see E5).
> - There are outliers. Means and scatter matrices are not robust, so a few outliers can tilt $w$.

> 【中文】以下情况会失效：
> - 两类均值相同或接近，只是方差不同（例如一类被另一类的圆环包在中间）。这时 m₁ − m₂ ≈ 0，任何方向上 J(w) ≈ 0。
> - 某一类是多峰的（如 XOR），一条直线分不开两类。
> - 两类协方差差别很大。最优边界是二次曲线，就像高斯 Bayes 分类器那样。
> - d 相对 N 很大，S_W 奇异或病态（见 E5）。
> - 有离群点。均值和散度矩阵不稳健，几个离群点就能把 w 拉歪。

下图把这几种情形各画了一个例子，黑线是 LDA 求出的边界（课件没有这张图）：
- **(a)** 两类都是高斯、协方差相同：LDA 正好合适，训练准确率 100%。
- **(b)** 一类在中间、另一类成圆环：两类均值几乎重合，任何直线都分不开，准确率约 50%。
- **(c)** XOR 型，每类两团：均值同样几乎重合，LDA 失效。
- **(d)** 红类右下角有一小撮离群点：红类的均值和散度被拉偏，边界跟着歪，蓝类有一部分被划到红色一侧。

![图 9-1 LDA 的适用与失效情形](figs/lda/fig9_when_lda_fails.png)

论述题可以画 (b) 或 (c) 的草图，一句话说明 "the class means coincide, so $m_1 - m_2 \approx 0$ and no linear boundary can separate them"。

### E5 ★★★ Why does LDA fail for high-dimensional data with a small sample size? How to fix it?（25-26 S2）
（为什么 LDA 在高维小样本数据上会失效？如何补救？）

> **Cause.** LDA needs $S_W^{-1}$. $S_W$ is $d \times d$ but is built from only $N$ samples, so $\mathrm{rank}(S_W) \le N - 2$. When $d > N - 2$, $S_W$ is singular and cannot be inverted. Even when $d$ is slightly below $N$, $S_W$ is ill-conditioned: its $d(d+1)/2$ entries are estimated from too few samples, its small eigenvalues are under-estimated, and inversion amplifies these errors. $w$ then fits noise in the training set: training accuracy is high but test accuracy is poor (overfitting, curse of dimensionality).
>
> **Remedies.**
> 1. **Regularization**: replace $S_W$ by $S_W + \beta I$ with a small $\beta > 0$ (e.g. 0.001). This makes the matrix invertible and shrinks $w$, reducing overfitting. It is useful even when $S_W$ is non-singular. Choose $\beta$ by cross-validation.
> 2. **Dimensionality reduction first**: apply feature selection or PCA so that $d < N$.
> 3. Use the pseudo-inverse of $S_W$, or assume a diagonal $S_W$ (ignore correlations).
> 4. Collect more training samples.

> 【中文】
> **原因。** LDA 需要 S_W⁻¹。S_W 是 d × d 矩阵，却只由 N 个样本构成，所以 rank(S_W) ≤ N − 2。当 d > N − 2 时 S_W 奇异，不可逆。即使 d 略小于 N，S_W 也是病态的：它的 d(d+1)/2 个元素用太少的样本估计，小特征值被低估，求逆又放大了这些误差。于是 w 拟合了训练集里的噪声：训练准确率高，测试准确率差（过拟合、维数灾难）。
>
> **补救。**
> 1. **正则化**：用 S_W + βI 代替 S_W，β 是小正数（如 0.001）。这使矩阵可逆，并收缩 w，减轻过拟合。即使 S_W 非奇异也有用。β 用交叉验证选。
> 2. **先降维**：做特征选择或 PCA，使 d < N。
> 3. 用 S_W 的伪逆，或假设 S_W 是对角阵（忽略相关性）。
> 4. 增加训练样本。

### E6 ★★★ What does |g(x)|/‖w‖ mean and how does it indicate confidence?
（|g(x)|/‖w‖ 是什么意思？它怎样表示置信度？）

见 §1.4 的模板句。加分写法：先写出 $r = g(x)/\lVert w\rVert$ 的推导（$x = x_p + r\,w/\lVert w\rVert$），再用"40 分及格线"那个比方的英文版：

> just as an estimated exam score of 90 gives much more confidence of passing a 40-mark threshold than an estimate of 41

> 【中文】就像估计考试能得 90 分，比估计能得 41 分，对"过 40 分及格线"要有把握得多。

### E7 ★★ Main differences between LDA and linear SVM.（23-24 S2，7 分）
（LDA 与线性 SVM 的主要区别。）

| 维度 | LDA | Linear SVM |
|---|---|---|
| Criterion（准则） | Maximize the Fisher ratio: between-class / within-class scatter of the projections | Maximize the margin $2/\lVert w\rVert$ |
| Samples used（用到的样本） | All samples, through means and scatter matrices | Only the support vectors（支持向量） |
| Assumption（假设） | Implicitly Gaussian classes with equal covariance; statistics-based | No distributional assumption; geometry-based |
| Solution（求解） | Closed form: $w = S_W^{-1}(m_1 - m_2)$ | Convex quadratic programming via the dual problem and Lagrange multipliers |
| Bias（偏置） | Midpoint of the projected means | From support vectors with $0 < \alpha < C$ |
| Outliers（离群点） | Sensitive: they shift means and scatter | Hard margin is sensitive; soft margin limits their effect through $C$ |
| Non-linear extension | Not directly | Kernel trick（核技巧） |
| High dimension, small $N$ | $S_W$ singular, needs regularization | Margin maximization acts as built-in regularization |

> In one sentence: LDA is a statistics-based method that uses all samples through their means and scatters, while SVM is a geometry-based method determined only by the samples near the boundary.

> 【中文】一句话总结：LDA 是基于统计的方法，通过均值和散度用到全部样本；SVM 是基于几何的方法，只由靠近边界的样本决定。

**上表中文对照**

| 维度 | LDA | 线性 SVM |
|---|---|---|
| 准则 | 最大化 Fisher 比：投影后的类间散度 / 类内散度 | 最大化间隔 2/‖w‖ |
| 用到的样本 | 全部样本，通过均值和散度矩阵 | 只用支持向量 |
| 假设 | 隐含高斯、协方差相同；基于统计 | 无分布假设；基于几何 |
| 求解 | 闭式解 w = S_W⁻¹(m₁ − m₂) | 通过对偶问题和拉格朗日乘子求解凸二次规划 |
| 偏置 | 投影均值的中点 | 由 0 < α < C 的支持向量求出 |
| 离群点 | 敏感：会拉动均值和散度 | 硬间隔敏感；软间隔通过 C 限制其影响 |
| 非线性扩展 | 不能直接扩展 | 核技巧 |
| 高维、小 N | S_W 奇异，需要正则化 | 最大化间隔本身就是正则化 |

老师在 week9 开头的衔接也值得写一句：两者都是线性分类器，用 hyperplane 划分空间。SVM 多了一个要求，就是 "classify the samples **correctly and confidently**"，让离边界最近的样本也尽量远。

### E8 ★★ Relationship between LDA and the Bayes classifier.
（LDA 与 Bayes 分类器的关系。）

> If both classes are Gaussian with a common covariance $\Sigma$, the Bayes discriminant $g(x) = \ln p(x\mid\omega_1)P(\omega_1) - \ln p(x\mid\omega_2)P(\omega_2)$ loses its quadratic terms and becomes linear: $g(x) = w^Tx + w_0$ with $w = \Sigma^{-1}(\mu_1 - \mu_2)$ and $w_0 = -\tfrac12(\mu_1 + \mu_2)^T\Sigma^{-1}(\mu_1 - \mu_2) + \ln\dfrac{P(\omega_1)}{P(\omega_2)}$. Since $S_W$ is proportional to the pooled estimate of $\Sigma$, this $w$ has the same direction as Fisher's $w = S_W^{-1}(m_1 - m_2)$, and with equal priors the thresholds also coincide. So LDA is Bayes-optimal (minimum error rate) under the equal-covariance Gaussian assumption. When the covariances differ, the Bayes boundary is quadratic and LDA is only an approximation.

> 【中文】如果两类都是高斯分布且协方差 Σ 相同，Bayes 判别函数 g(x) = ln p(x|ω₁)P(ω₁) − ln p(x|ω₂)P(ω₂) 的二次项消失，变成线性：g(x) = wᵀx + w₀，其中 w = Σ⁻¹(μ₁ − μ₂)，w₀ = −½(μ₁ + μ₂)ᵀΣ⁻¹(μ₁ − μ₂) + ln[P(ω₁)/P(ω₂)]。由于 S_W 与 Σ 的合并估计成正比，这个 w 和 Fisher 的 w = S_W⁻¹(m₁ − m₂) 方向相同；先验相等时阈值也一致。所以在等协方差高斯假设下，LDA 是 Bayes 最优的（错误率最小）。协方差不同时，Bayes 边界是二次的，LDA 只是近似。

### E9 ★★ Why does only the direction of w matter? Why can the eigenvalue method give a w that looks different?
（为什么只有 w 的方向有意义？为什么特征值方法得到的 w 看起来不一样？）

> $J(w)$ is invariant to scaling: $J(cw) = J(w)$ for any $c \ne 0$. Scaling $w$ (together with $w_0$) multiplies $g(x)$ by $c$ but does not change its sign for $c > 0$, nor the distance $g(x)/\lVert w\rVert$. Hence the closed-form $w$ and the eigenvector found by `eig` may have very different values, yet the same ratio of components, i.e. the same direction. The eigenvector may point the opposite way, so we must check that class-1 samples give $g > 0$ and replace $(w, w_0)$ by $(-w, -w_0)$ if not. We must also use the eigenvector with the non-zero (largest) eigenvalue; the one with eigenvalue 0 gives $J = 0$ and has no discriminative power.

> 【中文】J(w) 对缩放不变：对任何 c ≠ 0，J(cw) = J(w)。把 w（连同 w₀）乘以 c，g(x) 也乘以 c；c > 0 时不改变 g(x) 的符号，也不改变距离 g(x)/‖w‖。因此闭式解的 w 和 `eig` 求出的特征向量数值可能差很多，但各分量的比例相同，即方向相同。特征向量可能指向相反方向，所以要检查第 1 类样本是否给出 g > 0，否则把 (w, w₀) 换成 (−w, −w₀)。还必须用非零（最大）特征值对应的特征向量；特征值为 0 的那个给出 J = 0，没有判别能力。

### E10 ★ LDA vs PCA for dimensionality reduction.（课件外延伸，联系 week12 特征提取）
（用于降维时 LDA 与 PCA 的比较。）

> PCA is **unsupervised**: it keeps the directions of maximum total variance and ignores class labels. LDA is **supervised**: it keeps the directions that maximize between-class relative to within-class scatter, and gives at most $c - 1$ dimensions. The direction of largest variance is not necessarily the most discriminative one (as noted in the lecture, "important information is not necessarily discriminative information"), so LDA is usually better for classification, while PCA is better for compression or when labels are unavailable.

> 【中文】PCA 是**无监督**的：保留总方差最大的方向，不看类别标签。LDA 是**有监督**的：保留类间散度相对类内散度最大的方向，最多给出 c − 1 维。方差最大的方向不一定最有判别力（课上说过 "important information is not necessarily discriminative information"），所以分类时 LDA 通常更好；压缩数据或没有标签时 PCA 更合适。

---

## 10. 自测清单（能不看讲义说出来，才算过关）

- [ ] **（第一级）默写总复习 PPT p.7–9：$J(w)$ 的两种写法、$S_B$、$S_i$（$x \in D_i$）、$S_W$、$w$、$w_0$ 和 "midpoint on the boundary" 的原理**
- [ ] **（第一级）画出总复习 p.7 的投影图和 p.9 的中点图，并说出 p.9 那张图在什么条件下才严格成立**
- [ ] 写出 $g(x)$、决策规则，并证明 $w \perp$ 超平面
- [ ] 推出 $r = g(x)/\lVert w\rVert$；说出原点距离 $w_0/\lVert w\rVert$ 的三种情况
- [ ] 写出 $J(w)$ 的标量形式和矩阵形式，定义 $S_i$、$S_W$、$S_B$
- [ ] 从 $S_Bw = \lambda S_Ww$ 推到 $w = S_W^{-1}(m_1-m_2)$，并说清 $\rho$、$\lambda$ 为什么能扔掉
- [ ] 解释为什么不能用 $S_B^{-1}$，以及为什么只看均值差不够
- [ ] $w_0$ 的公式、背后的假设、假设不成立时怎么办
- [ ] $S_W$ 什么时候奇异、怎么补救、为什么补救也能防过拟合
- [ ] 特征值法的两个坑：选非零特征值、检查符号
- [ ] 十分钟内手算完 §7.4，置信度有参照、有结论
- [ ] 默写 LDA vs SVM 对比表中至少 5 行

**综合模拟卷**：[ML综合模拟卷.md](ML综合模拟卷.md)（各专题练习题已删除，统一用这张卷子练）。

**其他配套练习**：
- 《ML两天学习计划_含仿真题.md》里的仿真题 M1（LDA + Gaussian NB）
- 真题：24-25 S1 Q3(a)、24-25 S2 Q3(b)(c)
- 25-26 两份按计划留到 Day 2 模考再做

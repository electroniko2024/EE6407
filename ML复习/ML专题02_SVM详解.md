# ML 专题 02：Linear Support Vector Machine（线性支持向量机，SVM）详解

> 依据：课件 ML-Slides5（52 页）、week9 录音稿（整节课都在讲 SVM）、总复习 PPT Revision2026S1 p.10–14、week13 录音稿（老师逐页讲总复习）、四份考过 SVM 的期末卷参考答案。
> 约定：专业术语用英文，括号里附中文。例题数字都用程序核对过。图片在 `figs/svm/` 文件夹，每张图前面写了看哪里。公式用 LaTeX 书写，沿用课件记号 $x(i)$、$d(i)$、$\alpha(i)$。

---

## 0. 这一章在考试里的分量（先看这个）

### 0.1 考频

SVM 在六份卷子里出现了四次，共 81 分，约占全部分值的 13.5%（不算 23-24 S2 那道 LDA 与 SVM 对比题）。

| 试卷 | 题号 | 考什么 | 分值 |
|---|---|---|---|
| 23-24 S2 | Q2(a) | 可分情形的 primal problem（原问题）：写出并解释 | 8 |
| 23-24 S2 | Q2(b) | 给 4 个 support vector（支持向量，4 维）和 $\alpha$，求 $w$、$b$，分类一个样本 | 10 |
| 23-24 S2 | Q2(c) | LDA 与 linear SVM 的主要区别 | 7 |
| 24-25 S1 | Q3(b) | 给 4 个支持向量（3 维）和 $\alpha$，**没给 $C$**，按可分情形求 $w$、$b$ 并分类 | 13 |
| 24-25 S2 | Q2(a) | **不可分**情形的 primal problem：写出并解释各项 | 10 |
| 24-25 S2 | Q2(b) | 给 6 个支持向量和 $\alpha$，**$C = 1$**，求 $w$、$b$（只能用 $0 < \alpha < C$ 的支持向量求 $b$） | 15 |
| 25-26 S2 | Q3(a) | slack variable（松弛变量）$\xi$ 和 $C$ 的作用 | 7 |
| 25-26 S2 | Q3(b) | 给 8 个支持向量和 $\alpha$，**$C = 3$**，求 $w$、$b$ | 12 |
| 25-26 S2 | Q3(c) | 分类测试样本，求距离并判断置信度 | 6 |

**规律**：
- 计算题每次都是同一个套路：**给 $\alpha$ 求 $w$ 和 $b$，再分类**。考试从来不让你解二次规划求 $\alpha$。
- 题目在变难：可分 → 软间隔（$C = 1$）→ 软间隔（$C = 3$，支持向量更多）→ 加上距离和置信度。
- 论述题三种：可分 primal、软间隔 primal、$\xi$ 和 $C$ 的作用。

### 0.2 老师在课上说过的话（week9 录音稿）

1. **SVM 的两个要求**：分类既要正确又要有信心，"classify the samples **correctly and confidently**"。confidently 的意思是样本离决策边界要远。
2. **间隔的定义**：margin of separation（间隔）是"the separation between the hyperplane and the **closest** data point"。
3. **和 LDA 的根本区别**（论述题素材）：LDA 用两类**均值**之差衡量类间分离；SVM 用**最近样本**之间的距离。老师的原话是 "different ideas for good classifiers… formulated into different optimization problems"。
4. **标签为什么必须是 +1 和 −1**：这样两条约束才能合成一条 $d(i)\,g(x(i)) \ge 1$。
5. **名字的来历**："the hyperplane is determined by the support vectors only"，其他样本的 $\alpha = 0$，对 $w$ 没有贡献。
6. **对偶问题**：对偶问题里只剩 $\alpha$ 一种未知数，是一个 quadratic programming（二次规划，QP）问题。约束个数等于样本数，规模很大，所以早年难解。后来出现了 SMO（sequential minimal optimization），把大 QP 拆成许多只含两个 $\alpha$ 的小 QP，SVM 才流行起来。
7. **$b$ 要取平均**：理论上用任何一个支持向量都能算出 $b$，实际上各个值略有差别，"we take the average"。
8. **考试提示（最重要）**："I remember exam question like this, I give you the values of α, I also give the C value, then ask you to calculate the b value… Maybe we can have a similar question this year… **this is very important** to know what support vectors we have on the margin."
9. **软间隔里 $C$ 的地位**：$C$ 是 hyperparameter（超参数），由用户给定，MATLAB 默认是 1。在对偶问题里，$C$ 变成 $\alpha$ 的**上界**：$0 \le \alpha \le C$。
10. **MATLAB 例子**（课件 p.42–49）：200 个样本里有 36 个支持向量，其中只有 3 个的 $\alpha < C$，"only these three" 能用来求 $b$。
11. **SVM 的局限**：它和 LDA、Bayes 一样基于欧氏距离。数值特征没问题，但性别、国籍这类 categorical / nominal feature（类别特征）用距离衡量就没意义了，引出下一章的 classification tree（分类树）。
12. **Kernel SVM（核 SVM）只简单提了一下**：先把样本非线性映射到新空间，再在新空间里做线性 SVM；详细内容在 EE7207 讲。

### 0.3 总复习 PPT（Revision2026S1）里的 SVM：p.10–14

| 页 | 内容 | 对应本讲义 |
|---|---|---|
| p.10 | $g(x) = w^Tx + b$；margin of separation 的定义；配图：两条虚线之间的宽度是 $2/\lVert w\rVert$ | §1、§2 |
| p.11 | **可分**情形的 primal problem：约束 $d(i)[w^Tx(i) + b] \ge 1$，目标 $J(w) = \tfrac12 w^Tw$ | §3 |
| p.12 | 可分情形的解：$w^* = \sum \alpha(i)d(i)x(i)$（只对 $\alpha > 0$ 求和）；$b^* = \dfrac{1}{N_s}\sum_{\alpha>0} [d(i) - w^{*T}x(i)]$；条件 $\alpha \ge 0$、$\sum\alpha(i)d(i) = 0$ | §4 |
| p.13 | **不可分**情形的 primal problem：约束 $d(i)[w^Tx(i) + b] \ge 1 - \xi(i)$，$\xi(i) \ge 0$；目标 $J = \tfrac12 w^Tw + C\sum\xi(i)$；"C controls the trade-off between error and margin" | §5 |
| p.14 | 不可分情形的解：$w^*$ 同上；**$b^*$ 只对 $0 < \alpha(i) < C$ 求平均**；条件 $0 \le \alpha \le C$、$\sum\alpha(i)d(i) = 0$ | §5、§7 |

![Revision p.10](figs/svm/revision_p10.png)

![Revision p.11 和 p.12](figs/svm/revision_p11_p12.png)

![Revision p.13 和 p.14](figs/svm/revision_p13_p14.png)

**week13 老师讲这几页时说的话**：
- SVM 的目标还是 "correctly and confidently"：correctly 变成约束，confidently 变成最小化 $\tfrac12\lVert w\rVert^2$（等价于最大化间隔）。
- $w$ 是支持向量的**加权和**，"for every support vector we have a weight… the corresponding α multiplied by the class label"。
- 求 $b$：把支持向量代入，令 $g(x)$ 等于它的标签 $\pm1$，解出 $b$，再取平均。
- 不可分情形："C plays an important role because C is the **upper bound** of α"。求 $b$ 只能用 $g(x) = \pm1$ 那两条线上的支持向量，"whose α is less than C, not equal to C"。

**注意**：总复习 PPT **没有写对偶问题 $Q(\alpha)$**，也没写拉格朗日推导。考试只要求会用结果。

**由此得到的复习优先级**：

| 级别 | 内容 | 理由 |
|---|---|---|
| **第一级：必须能默写和解释** | 总复习 p.10–14 的全部内容：间隔 $2/\lVert w\rVert$、两种 primal problem、$w^*$、$b^*$、$\alpha$ 的约束 | 总复习专门列出，四卷计算题全靠它 |
| **第二级：必须会做** | 分类 + 距离 + 置信度（和 1 比）；$\xi$ 和 $C$ 的作用；哪些支持向量能求 $b$ 以及为什么；自检技巧；LDA vs SVM | 总复习没写，但真题考过：$\xi$ 和 $C$（7 分）、置信度（6 分）、LDA vs SVM（7 分） |
| **第三级：理解即可** | 拉格朗日函数与两个最优性条件；KKT 条件；对偶问题 $Q(\alpha)$；$\beta = C - \alpha$ 的推导；为什么 $\alpha = C$ 的点不在间隔线上 | 用来回答"为什么"，论述题能加分 |
| **知道一句话就行** | kernel SVM；SMO；MATLAB 输出 | 老师说 kernel 在 EE7207 讲 |

---

## 1. Motivation（动机）：不仅要分对，还要分得有信心（Slides5 p.1–8）

### 1.1 能把两类分开的直线有无数条

线性分类的目标是找一个 hyperplane（超平面）$w^Tx + b = 0$，把特征空间分成两半：
- **positive side（正侧）**：$g(x) > 0$，判为 $d = +1$ 的类。
- **negative side（负侧）**：$g(x) < 0$，判为 $d = -1$ 的类。

样本线性可分时，能把训练样本全分对的直线有无数条。选哪一条？

### 1.2 老师的 A、B、C 三点例子（p.3–4）

三个测试点都落在正侧，都判为 +1，但信心不一样：
- A 离边界很远：很有把握。
- C 离边界很近：虽然判为 +1，但超平面稍微动一下，结果就可能翻转。

所以理想的超平面要做到：
1. **correctly（正确）**：训练样本全部分对。
2. **confidently（有信心）**：样本离超平面尽量远。

**看课件 Slides5 p.5**（下图）：同样的数据，(a) 的直线贴着样本，(b) 的直线离两类都远，所以 (b) 更好。

![Slides5 p.5](figs/svm/slides_p5.png)

### 1.3 Margin of separation（间隔）与 optimal hyperplane（最优超平面）

- **Margin of separation（间隔）**：超平面与**最近**样本之间的距离（课件 p.7）。
- **SVM 的目标**：找出使间隔最大的那个超平面（p.8）。
- 这个超平面叫 **optimal hyperplane（最优超平面）**。老师强调，说"最优"必须先有标准，SVM 的标准就是间隔。

---

## 2. 几何：距离、规范化缩放与支持向量（Slides5 p.9–14）

### 2.1 点到超平面的距离（和 LDA 完全一样）

$$r = \frac{g(x)}{\lVert w\rVert}$$

推导和 LDA 专题 §1.3 相同：把 $x$ 写成 $x_p + r\,\dfrac{w}{\lVert w\rVert}$，代入 $g$，利用 $g(x_p) = 0$（课件 p.9–10）。SVM 课件把偏置记作 $b$，LDA 记作 $w_0$，是同一个东西。

### 2.2 Canonical scaling（规范化缩放）

$w$ 和 $b$ 同时乘一个正数，超平面不变。所以可以**选一个缩放**，使离超平面最近的样本满足 $\lvert w^Tx + b\rvert = 1$（课件 p.11）：
- $d = +1$ 的样本：$w^Tx + b \ge 1$
- $d = -1$ 的样本：$w^Tx + b \le -1$

两式合成一条（这就是标签必须取 $\pm1$ 的原因）：

$$d(i)\,[w^Tx(i) + b] \ge 1,\qquad i = 1, \dots, N$$

### 2.3 Support vector（支持向量）

**等号成立**的样本，也就是 $g(x) = +1$ 或 $g(x) = -1$ 的样本，叫 support vector（支持向量）。
- 它们落在两条平行线 $g = \pm1$ 上，是离决策边界最近的样本。
- 它们最难分类（"the most difficult to classify"）。
- 后面会看到，**$w$ 和 $b$ 只由支持向量决定**。

### 2.4 间隔 = 2/‖w‖（必背）

- 正类支持向量：$g = +1$，到边界的距离是 $1/\lVert w\rVert$。
- 负类支持向量：$g = -1$，到边界的距离也是 $1/\lVert w\rVert$。
- 两条线之间的距离：

$$\rho = \frac{2}{\lVert w\rVert}\quad\text{（课件 p.13）}$$

所以 **最大化间隔 $\Longleftrightarrow$ 最小化 $\lVert w\rVert$ $\Longleftrightarrow$ 最小化 $\tfrac12 w^Tw$**（课件 p.14）。

**看课件 Slides5 p.12 和 p.19**（下图左、右）：黑实线是 $g = 0$，两条虚线是 $g = +1$ 和 $g = -1$，箭头指的就是支持向量，正好压在虚线上。p.19 标出它们的 $\alpha \ne 0$。

![Slides5 p.12 和 p.19](figs/svm/slides_p12_p19.png)

下图是 §7.3 热身例题的结果：
- 绿圈是 3 个支持向量，都压在虚线上，$\alpha$ 分别是 $4/9$、$2/9$、$2/9$。
- 其余 3 个点在虚线外面，$\alpha = 0$。
- 紫色双箭头是间隔 $2/\lVert w\rVert = 2.121$。

![图 2-1 硬间隔 SVM](figs/svm/fig1_hard_margin.png)

---

## 3. Primal problem（原问题）：可分情形（Slides5 p.15–16，总复习 p.11）

> 23-24 S2 Q2(a)（8 分）原题考的就是这一节。

### 3.1 表述（要能默写）

Given the training samples $\{x(i), d(i)\}_{i=1}^N$, find the optimal values of the weight vector $w$ and bias $b$ such that they satisfy the constraints

$$d(i)\,[w^Tx(i) + b] \ge 1,\qquad i = 1, 2, \dots, N$$

and the weight vector minimizes the cost function

$$J(w) = \frac12 w^Tw$$

### 3.2 每一项的来历

| 部分 | 对应要求 | 解释 |
|---|---|---|
| 约束 $d(i)\,g(x(i)) \ge 1$ | correctly | 每个训练样本都在正确一侧，而且在间隔线之外 |
| 目标 $\tfrac12 w^Tw$ | confidently | 间隔 $= 2/\lVert w\rVert$，最小化 $\lVert w\rVert$ 就是最大化间隔 |
| 平方和 $\tfrac12$ | 计算方便 | 平方去掉根号，$\tfrac12$ 在求导时和 2 抵消；问题变成二次规划 |

### 3.3 这个问题的性质（课件 p.16）

- 目标函数 $J(w)$ 是 **convex（凸）**的。
- 约束对 $w$ 是 **linear（线性）**的。
- 所以这是一个凸二次规划，有唯一的全局最优解，可以用 **method of Lagrange multipliers（拉格朗日乘子法）**求解。

---

## 4. 拉格朗日函数、对偶问题与 w、b 的求法（Slides5 p.16–24，总复习 p.12）

> 这一节的推导属于第三级，但**结论**（$w^*$ 和 $b^*$ 的公式）是第一级，四份卷子的计算题都靠它。

### 4.1 Lagrangian（拉格朗日函数）

$$L(w, b, \alpha) = \frac12 w^Tw - \sum_{i=1}^N \alpha(i)\big\{d(i)[w^Tx(i) + b] - 1\big\}$$

其中 $\alpha(i) \ge 0$ 叫 Lagrange multiplier（拉格朗日乘子）。每个训练样本有一条约束，所以有 $N$ 个 $\alpha$。

> 为什么中间是减号：标准形式的约束是 $g_i(x) \le 0$，而这里的约束是 $d\,g - 1 \ge 0$，移项后变号。老师在 week9 专门解释过这一点。

### 4.2 两个最优性条件

**Condition 1**：$\dfrac{\partial L}{\partial w} = 0$

$w - \sum \alpha(i)d(i)x(i) = 0$，所以

$$w = \sum_{i} \alpha(i)d(i)x(i)$$

$w$ 是训练样本的**加权和**，第 $i$ 个样本的权重是 $\alpha(i)d(i)$。老师特别指出，这和 LDA 的 $w = S_W^{-1}(m_1 - m_2)$ 完全不同。

**Condition 2**：$\dfrac{\partial L}{\partial b} = 0$

$$\sum_{i} \alpha(i)d(i) = 0$$

也就是说，正类支持向量的 $\alpha$ 之和等于负类支持向量的 $\alpha$ 之和。**考试时先用它验算题目给的 $\alpha$**。

### 4.3 KKT condition（KKT 条件）：为什么只有支持向量 α > 0

最优解处满足 complementary slackness（互补松弛）：

$$\alpha(i)\,\big\{d(i)[w^Tx(i) + b] - 1\big\} = 0$$

两个因子至少有一个为 0：
- 样本不在间隔线上（$d\,g > 1$）：后一个因子不为 0，所以 **$\alpha(i) = 0$**。
- $\alpha(i) > 0$：后一个因子必须为 0，即 $d\,g = 1$，样本**一定在间隔线上**，就是支持向量。

所以 $w = \sum_{\alpha>0} \alpha(i)d(i)x(i)$：**只有支持向量参与计算**，其他样本删掉也不影响结果。这就是 "support vector machine" 名字的由来。

### 4.4 Dual problem（对偶问题）（总复习没列，知道即可）

把 $w = \sum\alpha d x$ 和 $\sum\alpha d = 0$ 代回 $L$，$w$ 和 $b$ 都消掉了，只剩 $\alpha$：

$$\max_\alpha\; Q(\alpha) = \sum_i \alpha(i) - \frac12 \sum_i\sum_j \alpha(i)\alpha(j)d(i)d(j)\,x(i)^Tx(j)$$

$$\text{subject to}\quad \sum_i \alpha(i)d(i) = 0,\qquad \alpha(i) \ge 0$$

（课件写成"$\min -Q$"，形式不同，意思一样。）

- 未知数只剩 $\alpha$ 一种，是标准的 QP 问题，用 QP solver（二次规划求解器）求解。
- 样本数可能是几万甚至上百万，规模很大；SMO 算法把它拆成一系列只含两个 $\alpha$ 的小问题。
- 对偶问题里样本只以**内积** $x(i)^Tx(j)$ 的形式出现，这正是后面 kernel SVM 能把内积换成核函数的原因（课件外延伸）。

### 4.5 求 b（课件 p.23–24）

支持向量满足 $d(i)[w^Tx(i) + b] = 1$。因为 $d(i) = \pm1$，所以 $1/d(i) = d(i)$，得到

$$b = d(i) - w^Tx(i)$$

**每个支持向量算一个 $b$，再取平均**：

$$b^* = \frac{1}{N_s}\sum_{\alpha(i)>0} \big[d(i) - w^{*T}x(i)\big],\qquad N_s\ \text{是支持向量的个数}$$

各个 $b$ 应该几乎相等。如果差得多，说明 $w$ 算错了，这是很好的自检。

### 4.6 自检技巧（课件外，但非常实用）

可分情形下 $\lVert w\rVert^2 = \sum \alpha(i)$。推导：

$$w^Tw = \sum \alpha(i)d(i)\,w^Tx(i) = \sum \alpha(i)d(i)\,[d(i) - b] = \sum \alpha(i) - b\sum \alpha(i)d(i) = \sum \alpha(i)$$

（用到 $d(i)^2 = 1$ 和 $\sum\alpha d = 0$。）

参考答案里 24-25 S1 的验算是 $\lVert w\rVert^2 = 0.8586$，$\sum\alpha = 0.8588$，吻合。**软间隔情形不成立**，见 §5.6。

---

## 5. Soft margin（软间隔）：不可分情形（Slides5 p.31–40，总复习 p.13–14）

> 24-25 S2 Q2（25 分）和 25-26 S2 Q3（25 分）整题都是这一节，是 SVM 最重要的部分。

### 5.1 为什么需要软间隔

实际问题几乎不可能 100% 线性可分。可分情形的约束 $d\,g \ge 1$ 对某些样本**无法满足**，原问题就无解。

违反约束有两种情况（课件 p.32–33）：
- **(a)** 样本落在间隔**内部**，但仍在**正确一侧**：分类正确，只是不够有信心。
- **(b)** 样本落在决策边界的**错误一侧**：分错了。

**看课件 Slides5 p.33**（下图）：标 "correct side" 的点在间隔里但没越过实线；标 "wrong side" 的点已经越过实线。

![Slides5 p.33](figs/svm/slides_p33.png)

**再看课件 Slides5 p.29 和 p.30**（下图左、右）：p.29 是硬间隔解，为了把一个靠近对方的点也分对，间隔被压得很窄；p.30 允许错 1 个训练样本，间隔宽得多。老师问哪个好，答案是 **p.30**：对未来数据的 misclassification risk（误分风险）更小。

![Slides5 p.29 和 p.30](figs/svm/slides_p29_p30.png)

### 5.2 Slack variable（松弛变量）ξ(i)

给每个样本引入一个非负变量 $\xi(i)$，把约束放宽为

$$d(i)\,[w^Tx(i) + b] \ge 1 - \xi(i),\qquad \xi(i) \ge 0$$

$\xi(i)$ 衡量第 $i$ 个样本违反约束的程度（课件 p.34）：

| $\xi(i)$ 的值 | 样本位置 | 分类结果 |
|---|---|---|
| $\xi = 0$ | 在间隔线上或间隔外 | 正确 |
| $0 < \xi \le 1$ | 在间隔**内部**，正确一侧 | 正确 |
| $\xi > 1$ | 越过决策边界 | **错误** |

因为分错的样本 $\xi > 1$，所以 **$\sum\xi(i)$ 是训练错分个数的上界**。

### 5.3 软间隔的 primal problem（要能默写）

Given the training samples $\{x(i), d(i)\}_{i=1}^N$, find the optimal values of $w$, $b$ and the slack variables $\xi(i)$ such that they satisfy the constraints

$$d(i)\,[w^Tx(i) + b] \ge 1 - \xi(i),\qquad \xi(i) \ge 0,\qquad i = 1, \dots, N$$

and minimize the cost function

$$J(w, \xi) = \frac12 w^Tw + C\sum_{i=1}^N \xi(i)$$

where the weight **$C$ controls the trade-off between error and margin**.

| 项 | 作用 |
|---|---|
| $\tfrac12 w^Tw$ | 间隔要大（confidently） |
| $C\sum\xi(i)$ | 违规要少（尽量 correctly） |
| $C$ | 两个目标的权重，hyperparameter，由用户给定或用 validation data（验证集）选 |
| $\xi(i) \ge 0$ | 不允许负的松弛"奖励"分得好的样本 |

### 5.4 C 的作用（25-26 S2 Q3(a) 考过）

- **$C$ 大**：违规的代价高，模型尽量把每个训练样本都分对，间隔变窄，接近硬间隔，容易 **overfitting（过拟合）**，对离群点敏感。
- **$C$ 小**：容忍更多违规，间隔变宽，模型更简单，可能 **underfitting（欠拟合）**。
- **$C \to \infty$**：退化成硬间隔 SVM。
- 在对偶问题里，$C$ 变成 $\alpha$ 的上界：$0 \le \alpha(i) \le C$。

下图用同一组数据对比两种 $C$（$\omega_1$ 里有两个点跑到了 $\omega_2$ 附近）：
- 左图 $C = 0.05$：不理会那两个点，间隔宽 2.61，训练错 3 个。
- 右图 $C = 100$：为了把它们分对，间隔压到 0.28，只剩 3 个支持向量，训练错 1 个。右图的边界很可能在新数据上表现更差。

![图 5-1 C 的作用](figs/svm/fig3_effect_of_C.png)

### 5.5 软间隔的解：哪些支持向量能用来求 b（本章最大考点）

对偶问题和可分情形几乎一样，只是约束变成 $0 \le \alpha(i) \le C$（课件 p.37）。$w$ 的公式不变：

$$w^* = \sum_{\alpha(i)>0} \alpha(i)d(i)x(i)\quad\text{（所有 $\alpha > 0$ 的支持向量都参与）}$$

**$b$ 只能用 $0 < \alpha < C$ 的支持向量**。原因来自 KKT 条件（课件 p.36–39，录音稿讲得很细）：

1. 对 $\xi(i)$ 求导：$\dfrac{\partial L}{\partial \xi(i)} = C - \alpha(i) - \beta(i) = 0$，所以 $\beta(i) = C - \alpha(i)$，$\beta(i)$ 是约束 $\xi(i) \ge 0$ 的乘子。
2. 互补松弛：$\beta(i)\xi(i) = 0$，即 $(C - \alpha(i))\,\xi(i) = 0$。
3. 分两种情况：
   - **$\alpha(i) = C$**：$\xi(i)$ 可以大于 0。样本可能在间隔内部，也可能被分错，**不一定在间隔线上**，不知道它的 $g$ 是不是 $\pm1$，所以**不能**用来求 $b$。
   - **$0 < \alpha(i) < C$**：$C - \alpha(i) > 0$，所以 $\xi(i) = 0$，样本**恰好在间隔线上**，$g = \pm1$，可以用 $b = d(i) - w^Tx(i)$ 求 $b$。

**$\alpha$ 的四种情况一览**（考试必备）：

| $\alpha(i)$ | $\xi(i)$ | 样本位置 | 参与算 $w$ | 参与算 $b$ |
|---|---|---|---|---|
| $\alpha = 0$ | 0 | 间隔外，正确 | 否 | 否 |
| $0 < \alpha < C$ | 0 | **恰好在间隔线上**（$g = \pm1$） | 是 | **是** |
| $\alpha = C$ | $0 < \xi \le 1$ | 间隔内，正确一侧 | 是 | **否** |
| $\alpha = C$ | $\xi > 1$ | 错误一侧（被分错） | 是 | **否** |

**总复习 p.14 的公式**：

$$b^* = \frac{1}{N_s}\sum_{0<\alpha(i)<C} \big[d(i) - w^{*T}x(i)\big]$$

这里 $N_s$ 是 $0 < \alpha < C$ 的支持向量个数。

下图是 §7.4 例题的结果（$C = 1$）：
- **绿圈**（#1、#3、#5）：$0 < \alpha < C$，正好压在虚线上，**只用它们算 $b$**。
- **橙圈**（#4、#6）：$\alpha = C$，在间隔里面但没越过实线，$\xi$ 分别是 0.85 和 0.17。
- **红圈**（#2）：$\alpha = C$，红点跑到了实线下方的蓝色一侧，被分错，$\xi = 2.31$。
- 没圈的点 $\alpha = 0$，不参与任何计算。

![图 5-2 软间隔 SVM](figs/svm/fig2_soft_margin.png)

**看课件 Slides5 p.44 和 p.48**（下图左、右）：p.44 是 MATLAB 例子中 36 个支持向量的 $\alpha$ 值（$C = 1$），绝大多数正好等于 1，只有 3 个往下凹（$\alpha < 1$），老师说只能用这 3 个求 $b$。p.48 用红圈标出了全部支持向量，比可分情形多得多，散布在间隔内和错误一侧。

![Slides5 p.44 和 p.48](figs/svm/slides_p44_p48.png)

### 5.6 软间隔的自检

- $\sum\alpha(i)d(i) = 0$：先验算题目给的 $\alpha$。
- 几个 $b$ 应该几乎相等。
- 求出 $w$、$b$ 后，可以把 $\alpha = C$ 的支持向量代入，**$d\,g$ 应该小于 1**（在间隔内或被分错）。参考答案里两道软间隔题都这样验证过。
- 课件外的等式：$\lVert w\rVert^2 = \sum\alpha(i) - C\sum\xi(i)$，推导和 §4.6 一样，只是 $d\,g = 1 - \xi$。可分情形 $\xi$ 全为 0，退化成 $\lVert w\rVert^2 = \sum\alpha$。

---

## 6. 软间隔里 C、ξ 和 α 的关系（把前面串起来）

$$
\begin{aligned}
&\textbf{primal:} && \min\ \tfrac12\lVert w\rVert^2 + C\textstyle\sum\xi(i)\quad\text{s.t. } d\,g \ge 1 - \xi,\ \xi \ge 0 && \leftarrow C\ \text{是违规的"单价"}\\
&\textbf{dual:} && 0 \le \alpha(i) \le C && \leftarrow \text{同一个 } C \text{ 变成 } \alpha \text{ 的上限}\\
&\textbf{KKT:} && \alpha = 0 \;\Rightarrow\; \text{间隔外}\\
& && 0 < \alpha < C \;\Rightarrow\; \xi = 0,\ \text{恰在间隔线上} && \Rightarrow \text{用来算 } b\\
& && \alpha = C \;\Rightarrow\; \xi > 0,\ \text{间隔内或分错} && \Rightarrow \text{不能算 } b
\end{aligned}
$$

一句话记忆：**$\alpha$ 碰到上限 $C$ 的点，都是"违规"的点**。

---

## 7. 计算题标准流程 + 两道例题

### 7.1 五步走（写全了就有步骤分）

1. **验算** $\sum\alpha(i)d(i) = 0$，并写出 $C$ 的值；**分出哪些支持向量满足 $0 < \alpha < C$**。
2. **求 $w$**：$w = \sum \alpha(i)d(i)x(i)$。建议列表：每行写一个支持向量的 $\alpha(i)d(i)x(i)$，最后一行求和。正类、负类分开算再相减，不容易错符号。
3. **求 $b$**：对每个满足条件的支持向量算 $b = d(i) - w^Tx(i)$，再取平均。
   - 可分情形（或题目没给 $C$）：所有 $\alpha > 0$ 的支持向量都能用。
   - 软间隔：**只用 $0 < \alpha < C$ 的**。
4. **写出分类器**：$g(x) = w^Tx + b$；$g > 0$ 判 +1 类，$g < 0$ 判 −1 类。
5. **分类 + 距离 + 置信度**：代入 $g(x)$ 判类别；$r = \lvert g(x)\rvert/\lVert w\rVert$；和间隔半宽 $1/\lVert w\rVert$ 比较，等价于看 $\lvert g(x)\rvert$ 是否大于 1。

### 7.2 常见失分点

- **漏乘标签 $d(i)$**：负类支持向量的贡献要带负号。参考答案把它列为第一大易错点。
- **软间隔里用 $\alpha = C$ 的点算 $b$**：24-25 S2 用 $s_1$ 会得到 $b = 4.18$，正确答案是 3.5877；25-26 S2 用 #1 会得到 3.876，正确答案是 3.2668。
- **把可分情形当软间隔**：24-25 S1 没给 $C$，就按可分处理，所有支持向量都能用。
- **题目说 separable 时写了 $\xi$ 和 $C$**：可分情形的 primal 不要加松弛变量。
- **置信度只写数字**：要写出 "inside / outside the margin" 和 "high / low confidence"。
- **SVM 和 LDA 的置信度参照混用**：SVM 和 1（或 $1/\lVert w\rVert$）比；LDA 和类均值到边界的距离比。

### 7.3 热身例题：可分情形（数值已用程序核对）

**题目**：$\omega_1$（$d = +1$）的样本为 $(2,2)$、$(3,3)$、$(4,1)$；$\omega_2$（$d = -1$）的样本为 $(0,0)$、$(1,0)$、$(0,1)$。训练硬间隔 SVM 得到的拉格朗日乘子为：$(2,2)$ 的 $\alpha = 4/9$，$(1,0)$ 和 $(0,1)$ 的 $\alpha = 2/9$，其余 $\alpha = 0$。
(a) 求 $w$、$b$ 和间隔。(b) 对 $x = (2, 1)$ 和 $x = (3, 3)$ 分类，并说明置信度。

**步骤 1：验算**
- $\sum\alpha(i)d(i) = 4/9 - 2/9 - 2/9 = 0$ ✓
- 支持向量：$(2,2)$、$(1,0)$、$(0,1)$。

**步骤 2：w**

| 支持向量 | $\alpha(i)d(i)$ | $\alpha(i)d(i)x(i)$ |
|---|---|---|
| $(2,2)$，$d = +1$ | $+4/9$ | $(8/9,\ 8/9)$ |
| $(1,0)$，$d = -1$ | $-2/9$ | $(-2/9,\ 0)$ |
| $(0,1)$，$d = -1$ | $-2/9$ | $(0,\ -2/9)$ |
| **和** | | $w = (6/9,\ 6/9) = (2/3,\ 2/3)$ |

**步骤 3：b**

| 支持向量 | $d(i)$ | $w^Tx(i)$ | $b = d(i) - w^Tx(i)$ |
|---|---|---|---|
| $(2,2)$ | +1 | $8/3$ | $-5/3$ |
| $(1,0)$ | −1 | $2/3$ | $-5/3$ |
| $(0,1)$ | −1 | $2/3$ | $-5/3$ |

三个值相同，$b = -5/3 \approx -1.6667$。

**步骤 4：分类器**
- $g(x) = \tfrac23 x_1 + \tfrac23 x_2 - \tfrac53$；$g > 0$ 判 $\omega_1$，$g < 0$ 判 $\omega_2$。
- 决策边界：$x_1 + x_2 = 2.5$。
- $\lVert w\rVert = 2\sqrt2/3 = 0.9428$，**间隔 $2/\lVert w\rVert = 3/\sqrt2 = 2.1213$**。
- 自检：$\lVert w\rVert^2 = 8/9$，$\sum\alpha = 4/9 + 2/9 + 2/9 = 8/9$ ✓
- 几何验证：$g = +1$ 的线是 $x_1 + x_2 = 4$，$g = -1$ 的线是 $x_1 + x_2 = 1$，两条平行线的距离是 $3/\sqrt2$ ✓
- 非支持向量的验证：$g(3,3) = 7/3 \ge 1$，$g(4,1) = 5/3 \ge 1$，$g(0,0) = -5/3 \le -1$ ✓

**步骤 5：分类与置信度**

| 测试点 | $g(x)$ | 判为 | $r = \lvert g\rvert/\lVert w\rVert$ | 和半宽 $1/\lVert w\rVert = 1.0607$ 比较 | 置信度 |
|---|---|---|---|---|---|
| $(2, 1)$ | $1/3$ | $\omega_1$ | 0.354 | $\lvert g\rvert < 1$，落在间隔**内** | **低** |
| $(3, 3)$ | $7/3$ | $\omega_1$ | 2.475 | $\lvert g\rvert > 1$，在间隔外，比支持向量还远 | **高** |

### 7.4 考试格式例题：软间隔（数值已用程序核对）

**题目**：一个线性 SVM 用 $C = 1$ 训练，得到 6 个支持向量：

| # | $x(i)$ | $d(i)$ | $\alpha(i)$ |
|---|---|---|---|
| 1 | $(3.6, 2.1)$ | +1 | 0.4904 |
| 2 | $(1.6, 0.4)$ | +1 | 1 |
| 3 | $(2.9, 3.1)$ | +1 | 0.5639 |
| 4 | $(2.5, 1.4)$ | −1 | 1 |
| 5 | $(1.4, 1.3)$ | −1 | 0.0543 |
| 6 | $(1.7, 1.2)$ | −1 | 1 |

(a) 求分类器。(b) 对 $x_a = (2.4, 2.0)$ 和 $x_b = (4.5, 3.5)$ 分类，求距离并判断置信度。

**步骤 1：验算和分类支持向量**
- $\sum\alpha(i)d(i) = (0.4904 + 1 + 0.5639) - (1 + 0.0543 + 1) = 2.0543 - 2.0543 = 0$ ✓
- $C = 1$。$0 < \alpha < C$ 的是 **#1、#3、#5**，只用它们求 $b$。#2、#4、#6 的 $\alpha = C$，只参与求 $w$。

**步骤 2：w**

| # | $\alpha(i)d(i)$ | $\alpha(i)d(i)x(i)$ |
|---|---|---|
| 1 | +0.4904 | $(1.76544,\ 1.02984)$ |
| 2 | +1 | $(1.6,\ 0.4)$ |
| 3 | +0.5639 | $(1.63531,\ 1.74809)$ |
| 4 | −1 | $(-2.5,\ -1.4)$ |
| 5 | −0.0543 | $(-0.07602,\ -0.07059)$ |
| 6 | −1 | $(-1.7,\ -1.2)$ |
| **和** | | $w = (0.72473,\ 0.50734)$ |

**步骤 3：b（只用 #1、#3、#5）**

| # | $d(i)$ | $w^Tx(i)$ | $b = d(i) - w^Tx(i)$ |
|---|---|---|---|
| 1 | +1 | 3.67444 | −2.67444 |
| 3 | +1 | 3.67447 | −2.67447 |
| 5 | −1 | 1.67416 | −2.67416 |

取平均，$b = -2.6744$。三个值几乎一样，说明 $w$ 没算错。

> 如果误用 #2（$\alpha = C$），会得到 $b = 1 - 1.3625 = -0.3625$；误用 #4 会得到 $-3.5221$。都和正确值差很远。

**步骤 4：分类器**
- $g(x) = 0.7247x_1 + 0.5073x_2 - 2.6744$；$g > 0$ 判 +1 类，$g < 0$ 判 −1 类。
- $\lVert w\rVert = 0.8847$，间隔半宽 $1/\lVert w\rVert = 1.1304$，间隔 2.2607。
- 决策边界：$x_2 = -1.4285x_1 + 5.2713$。
- 验证 $\alpha = C$ 的点（可写可不写）：

| # | $d\,g$ | $\xi = 1 - d\,g$ | 含义 |
|---|---|---|---|
| 2 | −1.3119 | 2.3119 | $\xi > 1$，**被分错** |
| 4 | 0.1523 | 0.8477 | 在间隔内，分对 |
| 6 | 0.8335 | 0.1665 | 在间隔内，分对 |

  三个 $d\,g$ 都小于 1，和 "$\alpha = C$ 的点违反间隔" 一致。另有课件外的等式：$\lVert w\rVert^2 = 0.7826$，$\sum\alpha - C\sum\xi = 4.1086 - 3.3261 = 0.7825$ ✓

**(b) 分类与置信度**

| 测试点 | $g(x)$ | 判为 | $r = \lvert g\rvert/\lVert w\rVert$ | 和 $1/\lVert w\rVert = 1.1304$ 比较 | 置信度 |
|---|---|---|---|---|---|
| $x_a = (2.4, 2.0)$ | +0.080 | +1 类 | **0.090** | $\lvert g\rvert < 1$，几乎贴着决策边界，在间隔内 | **低** |
| $x_b = (4.5, 3.5)$ | +2.362 | +1 类 | **2.671** | $\lvert g\rvert > 1$，在间隔外，比正类支持向量还远一倍多 | **高** |

可以直接套用的句子：

> $x_b$ gives $g(x_b) = 2.362 > 0$, so it is assigned to class +1. Its distance to the decision boundary is $r = \lvert g\rvert/\lVert w\rVert = 2.671$, which is larger than the half-margin $1/\lVert w\rVert = 1.130$ (equivalently $\lvert g\rvert > 1$). The sample lies outside the margin on the positive side, so the classification is made with **high confidence**.

### 7.5 画图（总复习 p.10 那张图）

论述题写 "use a sketch" 时就画这张图：
1. 画坐标轴，两类样本各画几个（○ 和 ×）。
2. 画实线 $w^Tx + b = 0$，两侧各画一条平行虚线，标 $w^Tx + b = +1$ 和 $w^Tx + b = -1$。
3. 在两条虚线之间画双箭头，标 margin $= 2/\lVert w\rVert$。
4. 把压在虚线上的样本圈出来，标 **support vectors（$\alpha > 0$）**。
5. 软间隔再加两点：一个在两虚线之间、正确一侧的样本，标 $0 < \xi \le 1$；一个越过实线的样本，标 $\xi > 1$。图 5-2 就是这个画法。

---

## 8. Kernel SVM（核 SVM）：知道三句话就够（Slides5 p.50–52）

1. 数据线性不可分（比如一类在中间、另一类成圆环）时，任何一条直线都分不开。
2. Kernel SVM 的思路分两步：先用非线性映射把样本送到一个新的特征空间，在新空间里样本变得线性可分；再在新空间里做线性 SVM。在原空间看，边界是一条曲线。
3. 课件只说 "kernel SVM is introduced in EE7207"，不考计算。

**看课件 Slides5 p.50 和 p.51**（下图）：p.50 的红点在中间、黑点围成一圈，直线分不开；p.51 的紫色圆圈就是 kernel SVM 的边界，圆圈附近的点是它的支持向量。老师举的现实例子是血压：只有在某个范围内才正常，太高太低都不正常。

![Slides5 p.50 和 p.51](figs/svm/slides_p50_p51.png)

下图演示映射的效果（课件没有这张图）：左边是原空间，分不开；右边把每个点映射成 $(x_1^2, x_2^2)$，一条直线就分开了。

![图 8-1 Kernel 的思想](figs/svm/fig8_kernel_idea.png)

---

## 9. LDA vs Linear SVM（23-24 S2，7 分）

| 方面 | Fisher LDA | Linear SVM |
|---|---|---|
| 什么叫"好的分类器" | 投影后类间均值差大、类内散度小 | 间隔最大（最近的样本离边界最远） |
| 优化问题 | 无约束最大化 $J(w) = \dfrac{w^TS_Bw}{w^TS_Ww}$ | 带约束最小化 $\tfrac12\lVert w\rVert^2$（$+\,C\sum\xi$） |
| 用哪些样本 | **所有**样本，通过均值和散度矩阵 | 只用**支持向量**（边界附近的少数样本） |
| 求解 | 闭式解 $w = S_W^{-1}(m_1 - m_2)$，需要 $S_W$ 可逆 | 解二次规划求 $\alpha$，再算 $w$、$b$ |
| 阈值 | $w_0$ 取两类均值的中点 | $b$ 由间隔线上的支持向量决定 |
| 分布假设 | 两类近似高斯、协方差相同时最优 | 不需要分布假设 |
| 远离边界的样本 | 会拉动均值和 $S_W$，从而改变边界 | 不影响（$\alpha = 0$） |
| 不可分 | 没有专门机制 | 软间隔：$\xi$ 和 $C$ |
| 高维小样本 | $S_W$ 奇异，需要正则化或降维 | 最大间隔本身有正则化作用，表现较好 |
| 置信度参照 | 类均值到边界的距离 $\lvert g(m_i)\rvert/\lVert w\rVert$ | 间隔半宽 $1/\lVert w\rVert$，即 $\lvert g\rvert$ 和 1 比 |

下图演示"用哪些样本"这一行的区别（课件没有这张图）：
- (a) 原始数据上，LDA（黑虚线）和 SVM（绿线）的边界差不多。
- (b) 给 $\omega_1$ 加一团离边界很远、本来就分对的样本：SVM 的支持向量（绿圈）没变，边界一点没动；LDA 的 $m_1$ 和 $S_W$ 被拉动，边界转了个大角度，甚至把一部分红点划错了。

![图 9-1 LDA vs SVM](figs/svm/fig6_lda_vs_svm.png)

一句话概括：**LDA is a statistics-based method using the means and scatters of all samples; SVM is a geometry-based method determined only by the support vectors near the boundary.**

---

## 10. 论述题专项（本章重点）

答题框架和 LDA 专题一样：**Definition → Explanation → Example / Equation → Conclusion**，要点个数 ≈ 分值 ÷ 2。

### E1 ★★★ Formulate the primal problem of the linear SVM for separable samples and justify it.（23-24 S2，8 分）
（写出样本可分时线性 SVM 的原问题，并说明理由。）

> **Formulation.** Given training samples $\{x(i), d(i)\}_{i=1}^N$ with $d(i) \in \{+1, -1\}$, find $w$ and $b$ that minimize $J(w) = \tfrac12 w^Tw$ subject to $d(i)[w^Tx(i) + b] \ge 1$ for $i = 1, \dots, N$.
>
> **Justification.**
> 1. *Constraints (correctly).* Since the samples are separable, a hyperplane $w^Tx + b = 0$ exists that separates the two classes. $w$ and $b$ can be scaled so that the closest samples satisfy $\lvert w^Tx + b\rvert = 1$. Then every positive sample has $w^Tx + b \ge 1$ and every negative sample has $w^Tx + b \le -1$; using the labels $\pm1$ the two conditions combine into $d(i)[w^Tx(i) + b] \ge 1$. This guarantees that all training samples are classified correctly and lie outside the margin.
> 2. *Objective (confidently).* The distance from a point to the hyperplane is $\lvert g(x)\rvert/\lVert w\rVert$, so the support vectors are at distance $1/\lVert w\rVert$ and the margin of separation is $\rho = 2/\lVert w\rVert$. Maximizing the margin is equivalent to minimizing $\lVert w\rVert$, or $\tfrac12 w^Tw$; the square and the factor $\tfrac12$ make it a differentiable quadratic function.
> 3. *Why a large margin.* Among the infinitely many separating hyperplanes, the one with the largest margin keeps all samples farthest from the boundary, so it is most tolerant to noise and gives the lowest risk of misclassifying future samples (best generalization).
> 4. *Properties.* The cost is convex and the constraints are linear in $w$, so it is a convex quadratic programming problem with a unique global optimum, solved via Lagrange multipliers and the dual problem.
>
> （配图：§7.5 的画法。）

> 【中文】
> **表述。** 给定训练样本 {x(i), d(i)}ᵢ₌₁ᴺ，d(i) ∈ {+1, −1}，求 w 和 b，使 J(w) = ½wᵀw 最小，约束为 d(i)[wᵀx(i) + b] ≥ 1，i = 1, …, N。
>
> **理由。**
> 1. *约束（分对，correctly）。* 样本可分，所以存在把两类分开的超平面 wᵀx + b = 0。可以缩放 w 和 b，使离超平面最近的样本满足 |wᵀx + b| = 1。于是每个正类样本 wᵀx + b ≥ 1，每个负类样本 wᵀx + b ≤ −1；利用 ±1 标签，两个条件合并为 d(i)[wᵀx(i) + b] ≥ 1。这保证所有训练样本都分对，并且都在间隔之外。
> 2. *目标（有把握，confidently）。* 点到超平面的距离是 |g(x)|/‖w‖，所以支持向量到边界的距离是 1/‖w‖，分类间隔 ρ = 2/‖w‖。最大化间隔等价于最小化 ‖w‖，即最小化 ½wᵀw；平方和系数 ½ 使它成为可导的二次函数。
> 3. *为什么要大间隔。* 能分开训练数据的超平面有无数个，间隔最大的那个让所有样本离边界最远，所以对噪声最宽容，对未来样本误分的风险最小（泛化最好）。
> 4. *性质。* 代价函数是凸的，约束对 w 是线性的，所以这是有唯一全局最优解的凸二次规划问题，用拉格朗日乘子和对偶问题求解。

### E2 ★★★ Formulate the primal problem of the linear SVM for non-separable samples and explain each term.（24-25 S2，10 分）
（写出样本不可分时线性 SVM 的原问题，并解释每一项。）

> **Formulation.** Find $w$, $b$ and slack variables $\xi(i)$ that minimize $J(w, \xi) = \tfrac12 w^Tw + C\sum_i \xi(i)$, subject to $d(i)[w^Tx(i) + b] \ge 1 - \xi(i)$ and $\xi(i) \ge 0$, $i = 1, \dots, N$.
>
> **Explanation.**
> 1. *Why it is needed.* When the samples are not linearly separable, the hard constraints $d(i)g(x(i)) \ge 1$ cannot all be satisfied and the separable formulation has no solution. The margin is therefore made "soft".
> 2. *$\tfrac12 w^Tw$.* As in the separable case, the margin of separation is $2/\lVert w\rVert$; minimizing this term maximizes the margin (classifying confidently).
> 3. *Slack variables $\xi(i)$.* Each sample gets a non-negative slack that measures how much it violates the margin constraint: $\xi = 0$ means the sample is on or outside the margin; $0 < \xi \le 1$ means it is inside the margin but on the correct side; $\xi > 1$ means it is misclassified. $\sum\xi(i)$ is an upper bound on the number of training errors, so minimizing it reduces errors.
> 4. *$C$.* $C$ is a user-specified hyperparameter that weights the two objectives and controls the trade-off between a large margin and few violations. A large $C$ penalizes violations heavily (narrow margin, risk of overfitting); a small $C$ tolerates more violations (wide margin, simpler model, risk of underfitting). It is usually chosen by validation.
> 5. *$\xi(i) \ge 0$* prevents correctly classified samples from obtaining a negative "reward".
> 6. The problem is still a convex QP. In its dual, $C$ appears as the upper bound of the Lagrange multipliers: $0 \le \alpha(i) \le C$.

> 【中文】
> **表述。** 求 w、b 和松弛变量 ξ(i)，使 J(w, ξ) = ½wᵀw + CΣᵢξ(i) 最小，约束为 d(i)[wᵀx(i) + b] ≥ 1 − ξ(i) 且 ξ(i) ≥ 0，i = 1, …, N。
>
> **解释。**
> 1. *为什么需要。* 样本线性不可分时，硬约束 d(i)g(x(i)) ≥ 1 无法全部满足，可分情形的表述无解。所以把间隔变"软"。
> 2. *½wᵀw。* 和可分情形一样，分类间隔是 2/‖w‖；最小化这一项就是最大化间隔（有把握地分类）。
> 3. *松弛变量 ξ(i)。* 每个样本有一个非负的松弛量，衡量它违反间隔约束的程度：ξ = 0 表示样本在间隔线上或间隔外；0 < ξ ≤ 1 表示在间隔内但在正确一侧；ξ > 1 表示被错分。Σξ(i) 是训练错分个数的上界，最小化它就减少错误。
> 4. *C。* C 是用户给定的超参数，给两个目标加权，控制"间隔大"和"违规少"之间的权衡。C 大则违规代价高（间隔窄，有过拟合风险）；C 小则容忍更多违规（间隔宽、模型简单，有欠拟合风险）。通常用验证集选。
> 5. *ξ(i) ≥ 0* 防止分对的样本得到负的"奖励"。
> 6. 问题仍是凸二次规划。在对偶问题中，C 成为拉格朗日乘子的上界：0 ≤ α(i) ≤ C。

### E3 ★★★ Explain the roles of the slack variables ξ(i) and the parameter C in the soft-margin SVM.（25-26 S2，7 分）
（解释软间隔 SVM 中松弛变量 ξ(i) 和参数 C 的作用。）

> **$\xi(i)$.** When the data are not linearly separable, the hard-margin constraints $d(i)[w^Tx(i) + b] \ge 1$ cannot all hold. The slack variable $\xi(i) \ge 0$ relaxes the constraint to $d(i)[w^Tx(i) + b] \ge 1 - \xi(i)$ and measures the degree of violation of sample $i$: $\xi(i) = 0$ on or outside the margin; $0 < \xi(i) \le 1$ inside the margin but correctly classified; $\xi(i) > 1$ misclassified. Hence $\sum\xi(i)$ upper-bounds the number of training errors and makes the optimization feasible.
>
> **$C$.** $C$ weights the total slack in the cost $\tfrac12 w^Tw + C\sum\xi(i)$ and controls the trade-off between maximizing the margin and minimizing the training violations. Large $C$ → violations are expensive → narrow margin, fewer training errors, close to the hard margin, prone to overfitting and sensitive to outliers. Small $C$ → more violations are tolerated → wider margin and a smoother, simpler classifier, possibly underfitting. $C$ is a hyperparameter chosen by the user or by cross-validation (MATLAB's default is 1). In the dual problem it becomes the upper bound $0 \le \alpha(i) \le C$; samples with $\alpha(i) = C$ are exactly those with $\xi(i) > 0$.

> 【中文】
> **ξ(i)。** 数据线性不可分时，硬间隔约束 d(i)[wᵀx(i) + b] ≥ 1 不能全部成立。松弛变量 ξ(i) ≥ 0 把约束放宽为 d(i)[wᵀx(i) + b] ≥ 1 − ξ(i)，并衡量样本 i 违反约束的程度：ξ(i) = 0 在间隔线上或间隔外；0 < ξ(i) ≤ 1 在间隔内但分对；ξ(i) > 1 被错分。所以 Σξ(i) 是训练错分数的上界，并使优化问题有可行解。
>
> **C。** C 是代价函数 ½wᵀw + CΣξ(i) 中总松弛量的权重，控制"最大化间隔"与"最小化训练违规"之间的权衡。C 大 → 违规代价高 → 间隔窄、训练错误少、接近硬间隔，容易过拟合，对离群点敏感。C 小 → 容忍更多违规 → 间隔宽、分类器更平滑简单，可能欠拟合。C 是超参数，由用户给定或用交叉验证选（MATLAB 默认为 1）。在对偶问题中它成为上界 0 ≤ α(i) ≤ C；α(i) = C 的样本恰好就是 ξ(i) > 0 的样本。

### E4 ★★★ What are support vectors? Why does the SVM depend only on them? Which support vectors can be used to compute b, and why?
（什么是支持向量？为什么 SVM 只依赖它们？哪些支持向量可以用来求 b，为什么？）

> 1. *Definition.* Support vectors are the training samples with nonzero Lagrange multipliers $\alpha(i) > 0$. In the separable case they lie exactly on the margin hyperplanes $w^Tx + b = \pm1$; they are the samples closest to the decision boundary and the most difficult to classify.
> 2. *Only support vectors matter.* From $\partial L/\partial w = 0$, $w = \sum\alpha(i)d(i)x(i)$. By the KKT condition $\alpha(i)\{d(i)[w^Tx(i) + b] - 1\} = 0$, every sample strictly outside the margin has $\alpha(i) = 0$ and contributes nothing. Removing all non-support vectors leaves the solution unchanged, which is why the method is called the support vector machine.
> 3. *Computing $b$.* A support vector on the margin satisfies $d(i)[w^Tx(i) + b] = 1$, so $b = d(i) - w^Tx(i)$. In the separable case all support vectors are on the margin and all can be used; in practice $b$ is averaged over them to reduce numerical error.
> 4. *Soft margin.* From $\partial L/\partial\xi(i) = 0$, $\beta(i) = C - \alpha(i)$, and from the KKT condition $\beta(i)\xi(i) = 0$. If $0 < \alpha(i) < C$, then $\xi(i) = 0$ and the sample lies exactly on the margin, so it can be used to compute $b$. If $\alpha(i) = C$, $\xi(i)$ may be positive: the sample lies inside the margin or on the wrong side, its $g(x)$ is not $\pm1$, and it cannot be used. Therefore $b = \dfrac{1}{N_s}\sum_{0<\alpha(i)<C}[d(i) - w^Tx(i)]$.

> 【中文】
> 1. *定义。* 支持向量是拉格朗日乘子非零（α(i) > 0）的训练样本。可分情形下它们恰好落在间隔超平面 wᵀx + b = ±1 上；它们是离决策边界最近、最难分类的样本。
> 2. *只有支持向量起作用。* 由 ∂L/∂w = 0，w = Σα(i)d(i)x(i)。由 KKT 条件 α(i){d(i)[wᵀx(i) + b] − 1} = 0，严格在间隔之外的样本 α(i) = 0，没有贡献。删掉所有非支持向量，解不变，这就是"支持向量机"名字的由来。
> 3. *求 b。* 间隔线上的支持向量满足 d(i)[wᵀx(i) + b] = 1，所以 b = d(i) − wᵀx(i)。可分情形下所有支持向量都在间隔线上，都能用；实际中对它们求出的 b 取平均，以减小数值误差。
> 4. *软间隔。* 由 ∂L/∂ξ(i) = 0，β(i) = C − α(i)；由 KKT 条件 β(i)ξ(i) = 0。若 0 < α(i) < C，则 ξ(i) = 0，样本恰好在间隔线上，可以用来求 b。若 α(i) = C，ξ(i) 可能为正：样本在间隔内或在错误一侧，它的 g(x) 不等于 ±1，不能用。所以 b = (1/N_s)Σ_{0<α(i)<C}[d(i) − wᵀx(i)]。

### E5 ★★ Why does the SVM maximize the margin? Why is a large margin desirable?
（SVM 为什么要最大化间隔？大间隔为什么好？）

> A linear classifier must classify the samples both correctly and confidently. Among the infinitely many hyperplanes that separate the training data, the one with the maximum margin places the boundary as far as possible from the closest samples of both classes. A test sample similar to a training sample (a small perturbation of it) is then unlikely to fall on the wrong side, so the risk of misclassifying future data is minimized and the classifier generalizes better. In contrast, a boundary that passes close to some training samples classifies them correctly but with low confidence (Slides5 p.5). A large margin also corresponds to a small $\lVert w\rVert$, which acts as a form of regularization.

> 【中文】线性分类器既要分得对，又要分得有把握。在能分开训练数据的无数个超平面中，间隔最大的那个让边界离两类最近的样本都尽可能远。与某个训练样本相似（只是略有扰动）的测试样本就不太可能落到错误一侧，所以对未来数据误分的风险最小，泛化更好。相反，贴着某些训练样本经过的边界虽然把它们分对了，但置信度很低（Slides5 p.5）。大间隔还对应较小的 ‖w‖，起到正则化的作用。

### E6 ★★ Main differences between LDA and linear SVM.（23-24 S2，7 分）
（LDA 与线性 SVM 的主要区别。）

见 §9 的表格。答题时至少写 4 条**对比**（每条同时说 LDA 怎样、SVM 怎样），推荐这 5 条：
1. 什么叫好的分类器（Fisher 比 vs 最大间隔）。
2. 用哪些样本（所有样本的统计量 vs 只有支持向量）。
3. 分布假设（高斯、协方差相同时最优 vs 不需要）。
4. 求解（闭式解 vs 二次规划）。
5. 不可分与离群点（没有专门机制、对远处样本敏感 vs 软间隔、对远处样本不敏感）。

最后加一句概括（见 §9 末尾）。

### E7 ★★ How do you judge the confidence of an SVM classification?
（如何判断 SVM 分类结果的置信度？）

> The distance from $x$ to the decision boundary is $r = \lvert g(x)\rvert/\lVert w\rVert$. The reference for the SVM is the margin: the support vectors lie at distance $1/\lVert w\rVert$, where $\lvert g\rvert = 1$. If $\lvert g(x)\rvert > 1$ ($r > 1/\lVert w\rVert$), the sample lies outside the margin, farther from the boundary than the support vectors, and is classified with high confidence. If $\lvert g(x)\rvert < 1$, it falls inside the margin, a region where the training data themselves are uncertain, so the confidence is low; the closer $\lvert g(x)\rvert$ is to 0, the lower the confidence. (For LDA the reference is instead the distance of the class means to the boundary.)

> 【中文】x 到决策边界的距离是 r = |g(x)|/‖w‖。SVM 的参照是间隔：支持向量到边界的距离是 1/‖w‖，此时 |g| = 1。若 |g(x)| > 1（r > 1/‖w‖），样本在间隔外，比支持向量离边界更远，分类置信度高。若 |g(x)| < 1，样本落在间隔内，这个区域连训练数据本身都不确定，所以置信度低；|g(x)| 越接近 0，置信度越低。（LDA 的参照则是类均值到边界的距离。）

### E8 ★ Why is the dual problem used?
（为什么要用对偶问题？）

> 1. The dual problem contains only one type of unknown, the Lagrange multipliers $\alpha(i)$, with simple constraints $\sum\alpha(i)d(i) = 0$ and $0 \le \alpha(i) \le C$; $w$ and $b$ are eliminated and recovered afterwards from the support vectors.
> 2. It is a standard quadratic programming problem that can be solved with QP solvers; for large data sets, SMO decomposes it into many small QPs with only two $\alpha$'s each.
> 3. The solution is sparse: most $\alpha(i) = 0$, which identifies the support vectors directly.
> 4. The training samples appear only through inner products $x(i)^Tx(j)$, which allows the kernel trick for nonlinear SVMs.

> 【中文】
> 1. 对偶问题只有一种未知数，即拉格朗日乘子 α(i)，约束也简单：Σα(i)d(i) = 0 和 0 ≤ α(i) ≤ C；w 和 b 被消去，事后再由支持向量求出。
> 2. 它是标准的二次规划问题，可以用 QP 求解器求解；数据量大时，SMO 把它拆成许多只含两个 α 的小 QP。
> 3. 解是稀疏的：大多数 α(i) = 0，直接识别出支持向量。
> 4. 训练样本只以内积 x(i)ᵀx(j) 的形式出现，因此可以用核技巧得到非线性 SVM。

### E9 ★ Limitations of the linear SVM.
（线性 SVM 的局限。）

> 1. It produces a linear boundary only; nonlinearly separable data require a kernel SVM.
> 2. Training requires solving a QP whose size equals the number of samples, which is expensive for very large data sets (mitigated by SMO).
> 3. Performance depends on the hyperparameter $C$, which must be tuned by validation.
> 4. Like LDA and the Gaussian Bayes classifier, it relies on Euclidean distance; this is not meaningful for categorical / nominal features (e.g. gender, nationality), where classification trees are more suitable (week9 lecture).
> 5. It is inherently a two-class classifier; multi-class problems need schemes such as one-vs-rest.
> 6. It outputs a decision value rather than a posterior probability.

> 【中文】
> 1. 只能产生线性边界；非线性可分的数据需要核 SVM。
> 2. 训练要解一个规模等于样本数的 QP，数据非常大时代价高（SMO 可缓解）。
> 3. 性能依赖超参数 C，必须用验证集调。
> 4. 和 LDA、高斯 Bayes 分类器一样，它依赖欧氏距离；这对类别 / 名义特征（如性别、国籍）没有意义，这时分类树更合适（week9 课上讲的）。
> 5. 它本质上是两类分类器；多类问题需要 one-vs-rest 等方案。
> 6. 输出的是决策值，而不是后验概率。

---

## 11. 自测清单（能不看讲义说出来，才算过关）

**第一级：总复习 p.10–14**
- [ ] 默写间隔的定义和 $\rho = 2/\lVert w\rVert$，并能说出 $2/\lVert w\rVert$ 怎么来的。
- [ ] 默写可分情形的 primal problem，并解释约束和目标分别对应 correctly 和 confidently。
- [ ] 默写软间隔的 primal problem，并解释 $\xi$、$C$ 和 $\xi \ge 0$。
- [ ] 默写 $w^* = \sum\alpha(i)d(i)x(i)$，以及两种情形下 $b^*$ 的公式和求和范围。
- [ ] 写出 $\alpha$ 的约束：可分 $\alpha \ge 0$；软间隔 $0 \le \alpha \le C$；两种都有 $\sum\alpha(i)d(i) = 0$。
- [ ] 能画出总复习 p.10 的图。

**第二级：计算与概念**
- [ ] 计算题五步走，特别是"先分出哪些 $\alpha$ 满足 $0 < \alpha < C$"。
- [ ] $\alpha$ 的四种情况表，以及为什么 $\alpha = C$ 的点不能算 $b$。
- [ ] 分类后会算 $r = \lvert g\rvert/\lVert w\rVert$，并和 $1/\lVert w\rVert$ 比较判断置信度。
- [ ] $C$ 大、$C$ 小分别会怎样。
- [ ] 至少写出 4 条 LDA vs SVM 的对比。
- [ ] 自检：$\sum\alpha d = 0$；几个 $b$ 是否一致；可分情形 $\lVert w\rVert^2 = \sum\alpha$；$\alpha = C$ 的点 $d\,g < 1$。

**第三级：理解**
- [ ] 能写出拉格朗日函数和两个最优性条件。
- [ ] 能用 KKT 互补松弛解释"只有支持向量 $\alpha > 0$"。
- [ ] 能说出对偶问题的好处。
- [ ] 能用一句话说清 kernel SVM 的思想。

**配套真题**：23-24 S2 Q2、24-25 S1 Q3(b)、24-25 S2 Q2、25-26 S2 Q3。按 §7.1 的五步走各手算一遍，再和参考答案核对。

**综合模拟卷**：[ML综合模拟卷.md](ML综合模拟卷.md)（各专题练习题已删除，统一用这张卷子练）。

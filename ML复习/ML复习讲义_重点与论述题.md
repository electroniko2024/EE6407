# EE6407 机器学习部分 · 复习讲义（重点 + 计算公式全集 + 论述题专项）

> 依据：课件 ML-Slides1–11、Revision2026S1、week5–13 录音稿、六份期末卷（23-24 S1 ～ 25-26 S2）参考答案、`6407历年考试题.docx`（含 21-22 S2、22-23 S2 题目概要）。
> 配套文件：[ML两天学习计划_含仿真题.md](ML两天学习计划_含仿真题.md)

---

## 0. 一页结论

1. **卷型**：4 题 × 25 分。Q1 是 GA，**Q2、Q3 是监督学习，Q4 是聚类**。老师 week5 原话：ML 占三题，"two questions from classification, one from clustering"。
2. **老师的警告**（week5）：每学期都换题，"If you just work on the exam question, you may fail"，真题只能当格式参考。所以近两年"为什么 / 什么时候失效 / 怎么补救"这类论述小问越来越多。
3. **论述分值**：ML 三题 75 分里，论述大约 25～45 分（25-26 S1 约 27 分，25-26 S2 约 24 分，23-24 S2 约 44 分）。**这块拿不满，计算再准也到不了 A。**
4. **计算题形式很固定**：LDA、SVM 求 w 和 b、朴素贝叶斯（三种变体）、Gini/熵、层次聚类、K-means。公式记牢、步骤写全就有分。**全部公式见 §3**。
5. **每道计算题后面基本都跟一句"state your confidence / justify"**。只给数字不写结论会扣分。
6. **总复习 PPT（Revision2026S1）是第一优先级的提纲**，但它没列出的内容照样考（见 §0.1）。

### 0.1 总复习 PPT 23 页对照表

week13 里老师把这份 PPT 逐页讲了一遍。

| 页 | 板块 | 列出的内容 | 真题里另外考过、但 PPT 没列的 |
|---|---|---|---|
| 1 | 1. Data preparation | 数据类型、缺失值、离群点、标准化与归一化（含 min-max 公式） | — |
| 2–6 | 2. Bayes | Bayes 定理（先验、类条件密度、后验）；多元高斯密度；Bayes 决策规则及其等价形式；Naive Bayes 三种：Gaussian、**Bernoulli**（p.6 有一张计数表）、Multinomial | Gaussian NB 的方差要用 scatter 除以 N |
| 7–9 | 3. Fisher LDA | g(x)、投影图；J(w) 的两种写法；S_B、Sᵢ、S_W；w = S_W⁻¹(m₁−m₂)；w₀ 取中点（配图） | **距离与置信度、高维小样本、适用条件、与 SVM 对比** |
| 10–14 | 4. Linear SVM | 间隔；可分情形的原问题，以及用 α > 0 的支持向量求 w、b；不可分情形的原问题（ξ、C），b 只用 **0 < α < C** 的支持向量 | 分类测试样本、距离与置信度 |
| 15–17 | 5. Classification tree | 熵、Gini（题目会指定用哪个）；随机森林 = bagging（bootstrapping + aggregating） | — |
| 18 | 6. Regression | 多元回归与 OLS；**Ridge（L2）、Lasso（L1）** | 至今未考 |
| 19 | 7. Evaluation | hold-out、k-fold、repeated k-fold、LOO；accuracy、sensitivity/specificity、precision/recall/F、**ROC/AUC** | — |
| 20–21 | 8. Feature selection | 搜索算法 + 评价准则；维数灾难、**peaking**、选择 vs 提取、单个特征 vs 特征子集；filter / wrapper；最优 vs 次优、前向 vs 后向 | embedded 方法（23-24 S2 考过） |
| 22–23 | 9. Clustering | K-means、凝聚层次聚类、**DBSCAN**、GMM；Silhouette、Dunn、DB、CH、Rand | 树状图（老师要求会画） |

用法：
- 每个板块先保证 PPT 上列出的东西能默写、能解释。
- 再补右列的"真题另考内容"。
- 各专题讲义（如《ML专题01_LDA详解.md》§0.3）会把对应页细化成"第一级 / 第二级 / 第三级"三个复习优先级。

---

## 1. 录音稿里老师亲口强调过的内容（按章节）

| 章节 | 老师说了什么（录音大意） | 出处 | 对考试意味着什么 |
|---|---|---|---|
| LDA | "考试不能用电脑，碰到这种题就用公式算 **w = S_W⁻¹(m₁ − m₂)**"；"如果你用 S_B 代替，那肯定错"——公式里一项管类间（m₁−m₂），一项管类内（S_W） | week8, week13 | 必考计算。论述可能问"为什么公式同时含类间和类内" |
| LDA | w₀ 的原理：**两类均值的中点应落在决策面上**，代入 g(x)=0 求 w₀；前提是两类散布相近，否则要调整 w₀ | week8, week13 | 论述"w₀ 怎么定、隐含什么假设" |
| LDA | "this part... normally at least tested in the exam in the past years" | week8 | 六卷全有 |
| LDA | 置信度 = 到决策面的距离。打比方：及格线 40 分，估分 90 很放心，估分 41～42 就很悬 | week8 | 判完类别要算 \|g(x)\|/‖w‖ 并说置信度 |
| **多类判别 MDA** | 学生问考试要不要看，老师答 **"Normally no, because it is too complex"** | week8 | **不用复习** |
| SVM | "**very essential to identify the samples on the margin**"；"往年我给过 α 值和 C，让你算 b……今年可能有类似的题" | week9 | 软间隔求 b **只能用 0 < α < C 的支持向量**，然后取平均 |
| SVM | 可分情形最少 2 个支持向量；不可分时支持向量多得多 | week9 | 概念题 |
| SVM | 总复习：目标是 "classify correctly **and confidently**"：confidently 对应最大化间隔（min ½‖w‖²），correctly 对应约束；C 是两者的权衡，**C 也是 α 的上界** | week13 | 原问题论述的标准答法 |
| 朴素贝叶斯 | Bernoulli NB："**This is very important. You must know this.**" | week7 | 必会 |
| 朴素贝叶斯 | 三种变体按特征类型选：连续用 Gaussian，二值用 Bernoulli，文本用 Multinomial | week7, week13 | 论述"选哪种、为什么" |
| 朴素贝叶斯 | 从频数表估计先验和类条件概率；"**不需要算 P(x)，比联合概率就行**" | week13 | 计算时可省分母 |
| 朴素贝叶斯 | 为什么要"朴素"：多元高斯的协方差矩阵参数多，样本少就估不准，假设特征独立后变成一维密度相乘 | week13 | 论述高频 |
| 决策树 | "考试会**指定用 entropy 还是 Gini**" | week13 | 两种都要会算（熵还没考过） |
| 决策树 | 选使子节点不纯度最小（不纯度下降最大）的特征和阈值；连续特征排序后取相邻值中点作候选阈值 | week10, week13 | 计算 + 论述 |
| 随机森林 | 两重随机：bootstrap 抽样本、随机抽特征子集，目的是**降低树之间的相关性、增加多样性**；聚合用多数投票。"Diversity is very important" | week10, week13 | 未考过，论述候选 |
| 回归 | Ridge 只把系数压小；**Lasso 能把系数压到 0，相当于做变量选择** | week13 | 未考过，论述候选 |
| 维数 | **Curse of dimensionality** "very important concept"，两方面：数据稀疏、距离集中 | week12 | 论述 |
| 维数 | **Peaking phenomenon** "very important concept"，它对**数据采集**和**特征选择**都有指导意义 | week12 | 23-24 S2 考过原题 |
| 特征 | 特征提取压缩信息更高效，特征选择可能丢失信息；但"重要的信息不一定是有判别力的信息" | week12 | 选择 vs 提取 |
| 聚类 | 层次聚类："you should be able to know how to do this hierarchical clustering, and you should be able to **sketch this dendrogram**" | week13 | 每年都考 |
| 聚类 | K-means 就是重复两步：分配到最近质心、按新分配重新算质心 | week12, week13 | 步骤描述 |
| 聚类指标 | 四个内部指标都同时考虑 **within-cluster cohesion 和 between-cluster separation**，只是度量方式不同；**DB 越小越好**；Rand index 用来比较两种聚类结果，取值 0～1 | week13 | 论述 |
| 数据准备 | 数据类型、缺失值、离群点、标准化/归一化 "this concept, I think you should be clear" | week13 | 23-24 S1 整题考过 |

---

## 2. 考点地图（考过什么、还没考什么）

| 板块 | 已考（年份） | 还没考 / 可能轮到 |
|---|---|---|
| Fisher LDA | 六卷都有：设计 + 分类 + 距离/置信度；原理/优化表述（25-26 S1）；最佳适用条件、高维小样本失效（25-26 S2）；与 SVM 对比（23-24 S2） | 正则化 S_W+λI 的细节；"为什么用比值而不是均值差" |
| SVM | 可分原问题（23-24 S2）；软间隔原问题（24-25 S2）；ξ 与 C 的作用（25-26 S2）；由 α 求 w、b（四卷） | 对偶问题表述；为什么只有支持向量起作用；‖w‖² = Σα 自检 |
| Bayes / NB | Gaussian Bayes 二次判别（24-25 S1）；Gaussian NB（23-24 S1、24-25 S1）；Bernoulli（24-25 S2、25-26 S1）；Multinomial（25-26 S1） | GMM/EM 概念；拉普拉斯平滑论述；Gaussian Bayes 与 LDA 的关系 |
| 决策树 / RF | Gini 选根节点（25-26 S1） | **熵 / 信息增益计算**、停止分裂与剪枝、随机森林原理 |
| 回归 | 从未考 | OLS、Ridge vs Lasso、R² 与调整 R² |
| 评估 | 混淆矩阵四指标（23-24 S2）；评估方法与指标（21-22 S2）；k 折交叉验证原理（22-23 S2） | ROC/AUC、灵敏度/特异度、留一法、重复 k 折 |
| 特征选择 | Peaking + 三类方法（23-24 S2）；filter/wrapper 框图（21-22 S2）；为数据集推荐方法（22-23 S2） | 维数灾难、个体特征评价（Fisher ratio、互信息）、SFS/SBE 步骤 |
| 数据准备 | 23-24 S1 整题 | — |
| 聚类 | single（4 次）、complete、centroid linkage；K-means 理论与手算；选 K；四类方法（2 次）；DBSCAN vs K-means；指标（5 个 / Silhouette / Dunn） | **average linkage**、DBSCAN 手算、GMM 聚类、DB / CH / Rand 的解释 |

---

## 3. 计算题公式全集

> 七个专题里计算题用到的公式全部收在这里，按考试顺序排列。凡是课件版本和教科书不同的（Gini 带 ½、Silhouette 的 b、CH 没有系数、方差除以 N），都**按课件写**，并标了"课件版"。
> 每节结构：公式块 → 判据 / 置信度 → 易错点。详细推导和例题见对应专题讲义。

### 3.0 通用数学工具（每道计算题都会用到）

```
2×2 行列式：   det[a b; c d] = ad − bc
2×2 求逆：     [a b; c d]⁻¹ = 1/(ad − bc) · [d −b; −c a]      （主对角互换、副对角变号）
二次型展开：   设 Δ = x − m = (Δ₁, Δ₂)，A = [p q; q r]（对称）
              ΔᵀAΔ = p·Δ₁² + 2q·Δ₁Δ₂ + r·Δ₂²                 （交叉项乘 2）
内积 / 范数：  wᵀx = w₁x₁ + w₂x₂ + …     ‖w‖ = √(w₁² + w₂² + …)

均值向量：     m = (1/N) Σ x
散度矩阵：     S = Σ (x − m)(x − m)ᵀ                         （不除以 N）
协方差（MLE）：C = S / N                                     （课件和考试都除以 N）
2 维散度矩阵的元素：
              S₁₁ = Σ(x₁ − m₁)²   S₂₂ = Σ(x₂ − m₂)²   S₁₂ = S₂₁ = Σ(x₁ − m₁)(x₂ − m₂)

距离：
  Minkowski   d(x, x') = (Σₖ |xₖ − x'ₖ|^q)^(1/q)，q ≥ 1
  Euclidean   q = 2：d = √(Σ (xₖ − x'ₖ)²)                    （考试默认）
  Manhattan   q = 1：d = Σ |xₖ − x'ₖ|
点到超平面的有符号距离：r = g(x) / ‖w‖，g(x) = wᵀx + w₀
原点到超平面的距离：     r₀ = w₀ / ‖w‖

对数换底：     log₂ x = ln x / ln 2 ；约定 0·log₂0 = 0
组合数：       C(n, 2) = n(n − 1)/2                           （n 个点有多少对）
```

- 中间结果至少保留 4 位有效数字。LDA 的 w 常在 10⁻² ～ 10⁻³ 量级，舍入太早会让 g(x) 的符号算错。

---

### 3.1 Fisher LDA

```
类内散度：     Sᵢ = Σ_{x∈ωᵢ} (x − mᵢ)(x − mᵢ)ᵀ
              S_W = S₁ + S₂
类间散度：     S_B = (m₁ − m₂)(m₁ − m₂)ᵀ                      （秩为 1）
Fisher 准则：  J(w) = (m̃₁ − m̃₂)² / (s̃₁² + s̃₂²) = wᵀS_B w / wᵀS_W w
              其中 m̃ᵢ = wᵀmᵢ，s̃ᵢ² = wᵀSᵢw
最优条件：     S_B w = λ S_W w   →   S_W⁻¹S_B w = λw            （λ = J(w)）
最优方向：     w = S_W⁻¹ (m₁ − m₂)                             （考试用这个）
阈值：         w₀ = −½ wᵀ(m₁ + m₂)                             （中点在决策面上）
判别函数：     g(x) = wᵀx + w₀ ；g > 0 → ω₁，g < 0 → ω₂
距离：         r = |g(x)| / ‖w‖
置信度参照：   r₁ = |g(m₁)| / ‖w‖，r₂ = |g(m₂)| / ‖w‖           （类中心到边界的距离）
正则化：       S_W ← S_W + βI，β 取小正数（如 0.001）
```

**给的不是散度矩阵时怎么换算**

```
给 covariance Σᵢ（MLE，除以 Nᵢ）：  Sᵢ = Nᵢ · Σᵢ
给无偏 covariance（除以 Nᵢ − 1）：  Sᵢ = (Nᵢ − 1) · Σᵢ
只有 N₁ = N₂ 时，直接用 Σ₁ + Σ₂ 才不改变 w 的方向
```

**先验不等时**（课件外延伸）：中点阈值相当于先验相等。先验不等时阈值还要加 ln[P(ω₁)/P(ω₂)] 的修正，边界往先验小的那一类移。考试仍按课件用中点法算，答卷上加一句说明即可（23-24 S1 样本数 600 : 400，参考答案就是这样处理的）。

- **置信度判据**：r 远小于 r₁、r₂ → 靠近边界，置信度低；r 接近或超过类中心距离 → 置信度高。**不能只看 |g| 的大小**，必须换成距离再比。
- 坑：(m₁ − m₂) 写反，整个判决反号；给 scatter 却除以 N（LDA 不需要）。
- 技巧：先算 adj(S_W)(m₁ − m₂)，最后再除以 det。det > 0，方向不变。

---

### 3.2 线性 SVM

```
标签：         d(i) ∈ {+1, −1}
规范化约束：   d(i)[wᵀx(i) + b] ≥ 1                          （硬间隔）

硬间隔 primal：min  J(w) = ½ wᵀw
              s.t. d(i)[wᵀx(i) + b] ≥ 1，i = 1, …, N

软间隔 primal：min  J(w, ξ) = ½ wᵀw + C Σ ξ(i)
              s.t. d(i)[wᵀx(i) + b] ≥ 1 − ξ(i)，ξ(i) ≥ 0

Lagrangian：   L(w, b, α) = ½ wᵀw − Σ α(i){d(i)[wᵀx(i) + b] − 1}，α(i) ≥ 0
∂L/∂w = 0：    w = Σ α(i) d(i) x(i)                          （只有 α > 0 的支持向量参与）
∂L/∂b = 0：    Σ α(i) d(i) = 0                               （先用它验算题目给的 α）
KKT：          α(i){d(i)[wᵀx(i) + b] − 1} = 0                （α > 0 ⇒ 恰在间隔线上）
软间隔 KKT：   β(i) = C − α(i)，(C − α(i)) ξ(i) = 0
Dual：         max Q(α) = Σ α(i) − ½ ΣΣ α(i)α(j)d(i)d(j) x(i)ᵀx(j)
              s.t. Σ α(i)d(i) = 0，0 ≤ α(i) ≤ C              （硬间隔时没有上界 C）

求 b：         b = d(k) − wᵀx(k)，再对可用的支持向量取平均 ✅️
              b* = (1/N_s) Σ [d(k) − wᵀx(k)]
              硬间隔：所有 α > 0 的支持向量都能用
              软间隔：只用 0 < α < C 的支持向量

分类器：       g(x) = wᵀx + b ；g > 0 → +1 类，g < 0 → −1 类
距离：         r = |g(x)| / ‖w‖
间隔：         半宽 1/‖w‖，总宽 ρ = 2/‖w‖
松弛变量：     ξ(i) = max(0, 1 − d(i)g(x(i)))
```

**α 的四种情况**

| α(i) | ξ(i) | 位置 | 算 w | 算 b |
|---|---|---|---|---|
| 0 | 0 | 间隔外，分对 | 否 | 否 |
| 0 < α < C | 0 | 恰在间隔线上，d·g = 1 | 是 | **是** |
| C | 0 < ξ ≤ 1 | 间隔内，分对 | 是 | 否 |
| C | ξ > 1 | 越过边界，**分错** | 是 | 否 |

- **置信度判据**：|g(x)| > 1 → 在间隔外，置信度高；|g(x)| < 1 → 在间隔内，置信度低。等价于 r 和 1/‖w‖ 比。
- 坑：漏乘 d(i)（负类要带负号）；软间隔用了 α = C 的点算 b；题目说 separable 却写了 ξ 和 C。

---

### 3.3 Bayes 与朴素贝叶斯

**Bayes 定理与决策规则**

```
后验：         P(ωⱼ|x) = p(x|ωⱼ) P(ωⱼ) / p(x)
证据：         p(x) = Σⱼ p(x|ωⱼ) P(ωⱼ)
先验：         P(ωⱼ) = Nⱼ / N
决策：         判给 P(ωⱼ|x) 最大的类
等价判别函数： gⱼ(x) = p(x|ωⱼ)P(ωⱼ)                         （不用算 p(x)）
              gⱼ(x) = ln p(x|ωⱼ) + ln P(ωⱼ)
归一化后验：   P(ωⱼ|x) = gⱼ / Σₖ gₖ                         （用来说置信度）
最小错误率：   在 x 处的错误概率 = 1 − maxⱼ P(ωⱼ|x)
```

**高斯密度与参数估计**

```
一元：         p(x|ω) = 1/(√(2π) σ) · exp(−(x − μ)² / (2σ²))
多元（d 维）： p(x|ωⱼ) = 1/((2π)^(d/2) |Cⱼ|^(1/2)) · exp(−½ (x − mⱼ)ᵀCⱼ⁻¹(x − mⱼ))
MLE：          μ̂ = (1/n) Σ xₖ
              Ĉ = (1/n) Σ (xₖ − μ̂)(xₖ − μ̂)ᵀ                （除以 n，不是 n − 1）
参数个数：     完整高斯每类 d + d(d + 1)/2；Gaussian NB 每类 2d
```

**完整 Gaussian Bayes 的对数判别函数**

```
gᵢ(x) = −½ (x − mᵢ)ᵀCᵢ⁻¹(x − mᵢ) − ½ ln|Cᵢ| + ln P(ωᵢ)
        （已去掉各类共有的 −(d/2) ln 2π）
Cᵢ 不同：  边界是二次曲线，−½ ln|Cᵢ| 不能省
Cᵢ 相同：  边界是直线，法向量 C⁻¹(m₁ − m₂)，和 LDA 方向一致
P 不同：   ln P(ωᵢ) 不能省
```

**三种朴素贝叶斯**

```
朴素假设：     p(x|ωⱼ) = Πᵢ p(xᵢ|ωⱼ)                         （给定类别，特征条件独立）
判别：         gⱼ(x) = P(ωⱼ) · Πᵢ p(xᵢ|ωⱼ)

Gaussian NB：  p(xᵢ|ωⱼ) = 1/(√(2π) σᵢⱼ) · exp(−(xᵢ − μᵢⱼ)² / (2σᵢⱼ²))   （每个特征在每一类内服从一元高斯）
（连续特征）     μᵢⱼ = 第 j 类第 i 维的均值
              σᵢⱼ² = 第 j 类第 i 维的方差 = Sⱼ 的第 i 个对角元 / Nⱼ
              对数形式：ln p = −ln(√(2π) σ) − (x − μ)² / (2σ²)

Bernoulli NB： p(xᵢ = v | ωⱼ) = (ωⱼ 中 xᵢ = v 的样本数) / Nⱼ
（二值特征）     特征取 No / 0 时，也要乘"取 No 的概率"

Multinomial NB：（计数特征（词频））
  不平滑       p(w|ωⱼ) = count(w, ωⱼ) / Σₖ count(wₖ, ωⱼ)
  Laplace 平滑 p(w|ωⱼ) = (count(w, ωⱼ) + 1) / (Σₖ count(wₖ, ωⱼ) + d)
              d = 全部训练文本中不同词的个数（两类共用）
              测试句中一个词出现几次，就乘几次
```

- **置信度判据**：写出归一化后验，例如 0.64 : 0.36 → 判 ω₁ 但把握一般；0.95 : 0.05 → 置信度高。
- 坑：scatter 当方差用（Gaussian NB 要先除以 N，和 LDA 正好相反）；σ 和 σ² 混淆；漏乘先验；完整 Bayes 漏掉 −½ ln|C|；Multinomial 分母加 1 而不是加 d。

---

### 3.4 决策树

```
P(ωⱼ) = 节点中第 j 类样本的比例

Entropy：               i_E(N) = −Σⱼ P(ωⱼ) log₂ P(ωⱼ)
Gini（课件版）：        i_G(N) = ½ [1 − Σⱼ P(ωⱼ)²] = Σ_{j<k} P(ωⱼ)P(ωₖ)  不要忘了按分支样本数加权；选最小	值。

二叉划分的不纯度下降：
  Δi(N) = i(N) − P_L·i(N_L) − P_R·i(N_R)，P_L = N_L/N，P_R = N_R/N
多叉划分（B 支）：
  Δi(N) = i(N) − Σₖ Pₖ·i(Nₖ)
  信息增益 = 用 entropy 时的 Δi
Gain ratio（增益率）：
  Δi'(N) = Δi(N) / (−Σₖ Pₖ log₂ Pₖ)                         （分母是划分本身的熵）

选划分：       Δi 最大  ⇔  加权子节点不纯度 Σₖ Pₖ·i(Nₖ) 最小
连续特征：     排序 → 相邻不同取值取中点 t = (x₍ₖ₎ + x₍ₖ₊₁₎)/2 → 逐个算加权不纯度 → 取最小✅️
叶节点标签：   多数投票（样本最多的那一类）
```

- 坑：没按样本数加权；选成最大的加权不纯度；不同度量的数值拿来互相比较。

---

### 3.5 分类器评估

```
混淆矩阵：     TP、FP、FN、TN；N = TP + FP + FN + TN

Accuracy       = (TP + TN) / N
Error rate     = (FP + FN) / N = 1 − Accuracy
Sensitivity    = TP / (TP + FN) = Recall = TPR 实际里对的（错很多）
Specificity    = TN / (TN + FP)
Precision      = TP / (TP + FP)  预测里对的（漏很多）
F1             = 2·P·R / (P + R) = 2TP / (2TP + FP + FN)
FPR            = FP / (FP + TN) = 1 − Specificity

ROC：          横轴 FPR，纵轴 TPR；每个阈值一个点
              手画：分数从高到低，遇正样本向上走 1/N₊，遇负样本向右走 1/N₋
AUC（竖条法）：AUC = Σ (FPR 区间宽度 × 该区间的 TPR 高度)
AUC（配对法）：AUC = (正样本分数 > 负样本分数的对数 + ½ × 打平对数) / (N₊ · N₋)
              1 完美；0.5 随机；0 完全反了

k 折交叉验证： 性能 = (1/k) Σ 第 i 折的性能；LOOCV 即 k = N
重复 k 折：    重复 r 次 k 折，共训练 r·k 个模型，取全部平均
```

- **先看表头**：行是预测还是真实（课件和 23-24 S2 都是"行 = 预测，列 = 真实"）。看反了 precision 和 recall 互换，accuracy 和 F1 不变。
- 类别不平衡时 accuracy 会骗人，要报 precision、recall、F1 或 AUC。

---

### 3.6 特征选择

```
Fisher ratio（连续特征）：
  R = (m₁ − m₂)² / (σ₁² + σ₂²)                               （越大越有判别力）

熵：           E(x) = −Σᵢ p(x = uᵢ) log p(x = uᵢ)
联合熵：       E(x, y) = −Σᵢ Σⱼ p(x = uᵢ, y = cⱼ) log p(x = uᵢ, y = cⱼ)
Mutual information（离散特征）：
  MI(x, y) = E(x) + E(y) − E(x, y)                           （≥ 0；独立时为 0；越大越相关）

子集个数：     d 个特征的非空子集共 2ᵈ − 1 个（穷举的代价）
              恰好选 m 个：C(d, m) = d! / (m!(d − m)!)
SFS 评估次数： 从 d 个选到 m 个，共 d + (d − 1) + … + (d − m + 1) 次
```

- 坑：Fisher ratio 只看均值差是不够的，分母的类内方差同样重要；MI 计算时联合概率要用联合频数 / 总数。

---

### 3.7 K-means

```
簇均值：       mᵢ = (1/nᵢ) Σ_{x∈Dᵢ} x
分配：         x 分给 argminⱼ ‖x − mⱼ‖                      （欧氏距离）
更新：         mⱼ ← 本轮分到第 j 簇的样本的均值
目标：         WCSS = Σᵢ Σ_{x∈Dᵢ} ‖x − mᵢ‖²                  （用更新后的中心算）
收敛：         分配不再变化（WCSS 每轮不增，只保证局部最优）
```

- 手算用表格写每个样本到每个中心的距离，保留 4 位小数。比较大小时可以用距离平方，省去开方。
- 题目要两轮就写两轮，没变化也要写出来并说明已收敛。

---

### 3.8 凝聚层次聚类

```
Single：       d(A, B) = min_{a∈A, b∈B} d(a, b)
Complete：     d(A, B) = max_{a∈A, b∈B} d(a, b)
Average：      d(A, B) = (1/(n_A n_B)) Σ_{a∈A} Σ_{b∈B} d(a, b)
Centroid：     d(A, B) = ‖c_A − c_B‖，c = 簇内全部原始样本的均值

合并 A、B 后，新簇到其他簇 K 的距离：（比mxn次）
  Single       d(A∪B, K) = min(d(A, K), d(B, K))
  Complete     d(A∪B, K) = max(d(A, K), d(B, K))
  Average      d(A∪B, K) = [n_A·d(A, K) + n_B·d(B, K)] / (n_A + n_B)   （按大小加权）
  Centroid     先算 c_{A∪B} = (n_A·c_A + n_B·c_B)/(n_A + n_B)，用原始坐标重新算质心
```

- 每一步都合并**距离最小**的两个簇（complete 只是用 max 定义距离，不是合并最远的）。
- average 不能直接把两个旧距离平均，除非两簇大小相同；centroid 要用全部成员重新求均值，不是"旧质心 + 新点"再除以 2。
- 每步写：合并了谁、合并高度、更新后的距离矩阵。最后画树状图并**标注高度**。

---

### 3.9 DBSCAN 与 GMM

**DBSCAN**

```
ε 邻域：       N_ε(p) = { q : d(p, q) ≤ ε }                  （含 p 自己）
核心点：       |N_ε(p)| ≥ MinPts                             （本讲义约定：含自己）
边界点：       不是核心点，但落在某个核心点的 ε 邻域内
噪声点：       既不是核心点也不是边界点
```

- 课件原文是 "more than MinPts"，常见实现是 "≥ MinPts 且含自己"。**答卷上写明约定**；题目给了定义就照题目。
- 只有核心点能扩展簇。两个非核心点互为邻居也不构成簇。

**GMM**（知道即可，至今没考过手算）

```
混合密度：     p(x) = Σᵢ αᵢ N(x | μᵢ, Σᵢ)，Σᵢ αᵢ = 1
E 步：         γₖᵢ = αᵢ N(xₖ | μᵢ, Σᵢ) / Σⱼ αⱼ N(xₖ | μⱼ, Σⱼ)   （样本 k 属于分量 i 的后验）
M 步：         Nᵢ = Σₖ γₖᵢ
              αᵢ = Nᵢ / N
              μᵢ = (1/Nᵢ) Σₖ γₖᵢ xₖ
              Σᵢ = (1/Nᵢ) Σₖ γₖᵢ (xₖ − μᵢ)(xₖ − μᵢ)ᵀ
聚类：         把样本分给后验 γₖᵢ 最大的分量
```

---

### 3.10 聚类评价指标（课件版）

```
Silhouette（样本 xᵢ 属于簇 D_p）：
  aᵢ = (1/(n_p − 1)) Σ_{xⱼ∈D_p, j≠i} d(i, j)                （到同簇其他点的平均距离）
  bᵢ = min_{xⱼ∉D_p} d(i, j)                                  （课件版：到其他簇样本的最小距离）
  SCᵢ = (bᵢ − aᵢ) / max(aᵢ, bᵢ)
  SC  = (1/N) Σ SCᵢ ∈ [−1, 1]                                 越大越好
  教科书 / sklearn 版：bᵢ = 到各个其他簇的平均距离中最小的那个

Dunn：
  簇直径     δ_p = max_{xᵢ, xⱼ∈D_p} d(i, j)
  簇间距离   d_pq = min_{xᵢ∈D_p, xⱼ∈D_q} d(i, j)
  DI = min_{p≠q} d_pq / max_p δ_p                             越大越好

Davies-Bouldin：
  簇内离散   δᵢ = √( (1/nᵢ) Σ_{x∈Dᵢ} ‖x − mᵢ‖² )
  簇间距离   d(i, j) = ‖mᵢ − mⱼ‖
  相似度     Sᵢⱼ = (δᵢ + δⱼ) / d(i, j)
  Sᵢ = max_{j≠i} Sᵢⱼ
  DB = (1/K) Σᵢ Sᵢ                                            越小越好（唯一一个）

Calinski-Harabasz（课件版，没有 (N − K)/(K − 1) 系数）：
  Cᵢ = Σ_{x∈Dᵢ} ‖x − mᵢ‖²       Sᵢ = ‖mᵢ − m‖²，m = 全局均值
  CH = Σᵢ nᵢSᵢ / Σᵢ Cᵢ                                       越大越好

Rand index（比较两种划分）：
  RI = (N_s + N_d) / C(N, 2)
  N_s = 两种划分都放同簇的样本对数；N_d = 都放不同簇的样本对数     范围 [0, 1]
```

- Dunn 分子取 min、分母取 max，不要写反。
- 只有 DB 越小越好；其余越大越好。

---

### 3.11 数据预处理

```
均值：         x̄ = (1/N) Σ x(k)                              （对离群点敏感）
中位数：       排序后中间的值；N 为偶数时取中间两个的平均
众数：         出现次数最多的值
方差（课件用 1/N）：
  var(X) = (1/N) Σ x²(k) − ((1/N) Σ x(k))² = (1/N) Σ (x(k) − x̄)²
  σ = √var
四分位数：     Q₂ = 整体中位数；Q₁ = 前半部分的中位数；Q₃ = 后半部分的中位数
五数概括：     min、Q₁、Q₂、Q₃、max
IQR：          IQR = Q₃ − Q₁
栅栏：         下 Q₁ − 1.5·IQR，上 Q₃ + 1.5·IQR；栅栏外的是离群点
须线：         下须 = 下栅栏以内的最小数据值；上须 = 上栅栏以内的最大数据值
z-score 判离群：|z| = |x − x̄| / σ > 3

Min-max 归一化：x' = (x − x_min) / (x_max − x_min)            → [0, 1]
z-score 标准化：x' = (x − μ) / σ                               → 均值 0，标准差 1（σ 用 1/N）
梯度下降更新： θᵢ(k + 1) = θᵢ(k) − η · ∂L(k)/∂θᵢ(k)
```

- 缩放参数（min、max 或 μ、σ）**只从训练集估计**，测试集用同一组参数变换。
- 分组插补：按相似样本（如同性别、同缸数）分组，用组内均值 / 中位数填。

---

### 3.12 回归

```
模型：         y = θ₀ + θ₁x₁ + … + θₘxₘ + ε
矩阵形式：     y = Φθ + ε，Φ 是 N × (m + 1)，第一列全是 1
OLS 损失：     J(θ) = ½ Σ εᵢ² = ½ (y − Φθ)ᵀ(y − Φθ)
正规方程：     ΦᵀΦ θ = Φᵀy
OLS 解：       θ̂ = (ΦᵀΦ)⁻¹ Φᵀy

一元回归时：   ΦᵀΦ = [N  Σx; Σx  Σx²]，Φᵀy = [Σy; Σxy]
  等价捷径：   θ̂₁ = S_xy / S_xx，θ̂₀ = ȳ − θ̂₁ x̄
              S_xy = Σxy − N x̄ ȳ，S_xx = Σx² − N x̄²
  （例：x = 1…5，y = 2,3,5,4,6：S_xy = 69 − 60 = 9，S_xx = 55 − 45 = 10，θ̂₁ = 0.9，θ̂₀ = 4 − 2.7 = 1.3）

Ridge（L2）：  J = ½ [(y − Φθ)ᵀ(y − Φθ) + λ θᵀθ]
              θ̂ = (ΦᵀΦ + λI)⁻¹ Φᵀy                         （课件版连截距一起罚）
Lasso（L1）：  J = ½ [(y − Φθ)ᵀ(y − Φθ) + λ Σ |θⱼ|]          （没有解析解，数值求解）

评价指标：
  RSS = Σ (ŷᵢ − yᵢ)²      TSS = Σ (yᵢ − ȳ)²
  MSE  = RSS / N
  RMSE = √MSE                                                （和 y 同单位）
  MAE  = (1/N) Σ |ŷᵢ − yᵢ|
  R²   = 1 − RSS / TSS                                       越大越好
  Adjusted R² = 1 − (1 − R²)(n − 1) / (n − m − 1)            m = 自变量个数
```

- 自检：有截距的 OLS，残差之和为 0。
- 坑：Φ 漏了全 1 的第一列；adjusted R² 的分母是 n − m − 1。

---

### 3.13 置信度判据汇总（每道计算题最后一句）

| 方法 | 算什么 | 参照 | 结论怎么写 |
|---|---|---|---|
| LDA | r = \|g(x)\| / ‖w‖ | 类中心到边界的距离 \|g(mᵢ)\| / ‖w‖ | r 远小于参照 → close to the boundary, low confidence |
| SVM | \|g(x)\|，或 r = \|g(x)\| / ‖w‖ | 1，或半间隔 1/‖w‖ | \|g\| < 1 → inside the margin, low confidence |
| Bayes / NB | 归一化后验 P(ωⱼ\|x) | 0.5（两类） | 接近 0.5 → low confidence；接近 1 → high confidence |
| K-means / 聚类 | 到两个中心的距离差、SCᵢ | SCᵢ 接近 0 或为负 | 在两簇边界上 / 可能分错 |

---

## 4. 论述题专项（重点）

### 4.1 论述题题库（按命中概率排序，★ 越多越可能考）

> 每题给"得分要点"和"可以直接写的英文答案"，英文后面附中文对照。**加粗部分是得分要点**：背的时候先背加粗的骨架，再练着用自己的话把句子补完整。

---

#### A. LDA

**A1 ★★★ Formulate Fisher LDA as an optimization problem and explain its rationale.（25-26 S1 考过，12 分）**
（把 Fisher LDA 表述成一个优化问题，并解释其原理。）

要点：投影 y = wᵀx → 投影后均值与散度 → J(w) 定义 → S_B、S_W 定义 → 解 w = S_W⁻¹(m₁−m₂) → 为什么用比值 → 只有方向有意义 → w₀ 的确定。

> LDA seeks a **projection direction w** that maps a d-dimensional sample to a scalar **y = wᵀx** such that the two classes are **as separated as possible after projection**.
> - **Projected means**: m̃ᵢ = wᵀmᵢ. **Projected within-class scatter**: s̃ᵢ² = Σ_{x∈Dᵢ}(wᵀx − m̃ᵢ)² = wᵀSᵢw.
> - **Fisher criterion:** maximize **J(w) = (m̃₁ − m̃₂)² / (s̃₁² + s̃₂²) = (wᵀS_B w)/(wᵀS_W w)**,
>   where **S_B = (m₁−m₂)(m₁−m₂)ᵀ** is the between-class scatter matrix and **S_W = S₁+S₂**, Sᵢ = Σ(x−mᵢ)(x−mᵢ)ᵀ, is the within-class scatter matrix.
> - Setting ∂J/∂w = 0 gives the **generalized eigenvalue problem S_B w = λS_W w**. Since S_B w is always in the direction of (m₁−m₂), the solution is **w = S_W⁻¹(m₁ − m₂)**.
> - **Rationale:** the **numerator measures the separation of the projected class means** (between-class); the **denominator measures how compact each projected class is** (within-class). Maximizing the mean difference alone is not enough: if the within-class variance along that direction is also large, the classes still overlap. **The ratio balances both, giving the direction with minimum overlap.** S_W⁻¹ corrects the mean-difference direction by the within-class spread.
> - J(w) is invariant to scaling of w, so **only the direction matters**.
> - The classifier is g(x) = wᵀx + w₀ with **w₀ = −½wᵀ(m₁+m₂)**, i.e. **the midpoint of the two projected means lies on the decision boundary** (assuming **similar class spreads and equal priors**).

> 【中文】LDA 寻找一个**投影方向 w**，把 d 维样本映射成标量 **y = wᵀx**，使两类**投影后尽可能分开**。
> - **投影后的均值**：m̃ᵢ = wᵀmᵢ。**投影后的类内散度**：s̃ᵢ² = Σ(wᵀx − m̃ᵢ)² = wᵀSᵢw。
> - **Fisher 准则**：最大化 **J(w) = (m̃₁ − m̃₂)² / (s̃₁² + s̃₂²) = wᵀS_B w / wᵀS_W w**，其中 **S_B = (m₁−m₂)(m₁−m₂)ᵀ** 是类间散度矩阵，**S_W = S₁+S₂** 是类内散度矩阵。
> - 令 ∂J/∂w = 0，得到**广义特征值问题 S_B w = λS_W w**。S_B w 总与 (m₁−m₂) 同方向，所以解为 **w = S_W⁻¹(m₁ − m₂)**。
> - **原理**：**分子衡量投影后两类均值的分离程度**（类间）；**分母衡量投影后每一类的紧凑程度**（类内）。只最大化均值差不够：如果该方向上类内方差也大，两类仍会重叠。**用比值同时兼顾两者，得到重叠最少的方向。** S_W⁻¹ 用类内散度对均值差方向做了修正。
> - J(w) 对 w 的缩放不变，所以**只有方向有意义**。
> - 分类器 g(x) = wᵀx + w₀，**w₀ = −½wᵀ(m₁+m₂)**，即**两类投影均值的中点落在决策边界上**（前提是**两类散度相近、先验相等**）。

**A2 ★★★ Under what conditions does LDA perform best?（25-26 S2）**
（LDA 在什么条件下表现最好？）

> 1. Each class is **Gaussian with equal covariance matrices**. Then the Bayes-optimal boundary is linear and **LDA coincides with the Bayes classifier** (minimum error rate).
> 2. The classes **differ mainly in their means**. If the means are close and only the variances differ, the numerator of J(w) ≈ 0 and LDA fails.
> 3. Each class is **unimodal** and the classes are **(nearly) linearly separable**. LDA cannot handle multimodal classes or nonlinear boundaries (e.g. XOR, rings).
> 4. **The number of samples is much larger than the dimension (N ≫ d)**, so S_W is estimated accurately and is invertible.
> 5. **No severe outliers**, since means and scatter matrices are sensitive to them.
> 6. With the midpoint threshold, **equal priors** are implicitly assumed; otherwise w₀ should be adjusted.

> 【中文】
> 1. 每一类都是**高斯分布且协方差矩阵相同**。这时 Bayes 最优边界是线性的，**LDA 与 Bayes 分类器一致**（错误率最小）。
> 2. 两类**主要在均值上不同**。如果均值接近、只有方差不同，J(w) 的分子 ≈ 0，LDA 失效。
> 3. 每一类是**单峰**的，两类**（近似）线性可分**。LDA 处理不了多峰的类或非线性边界（如异或、圆环）。
> 4. **样本数远大于维数（N ≫ d）**，S_W 才估计得准、可逆。
> 5. **没有严重的离群点**，因为均值和散度矩阵对离群点敏感。
> 6. 用中点阈值时隐含了**先验相等**的假设；否则 w₀ 要调整。

**A3 ★★★ Why does LDA fail for high-dimensional data with few samples? How to fix it?（25-26 S2）**
（为什么 LDA 在高维小样本数据上会失效？如何补救？）

> **Cause:** LDA requires **S_W⁻¹**. S_W is a d×d matrix formed by summing N outer products, so **its rank is at most N − c** (N − 2 for two classes). **When d > N − 2, S_W is singular and cannot be inverted.** Even when d is slightly smaller than N, S_W is **ill-conditioned**: d(d+1)/2 parameters are estimated from too few samples, small eigenvalues are underestimated and amplified by inversion, so **w becomes very sensitive to noise → good training accuracy but poor test accuracy** (**overfitting**, curse of dimensionality / peaking phenomenon).
> **Remedies:**
> - **Regularization:** use **S_W + λI** instead of S_W; the matrix becomes positive definite and invertible; **λ chosen by cross-validation**. (课件第 4 章原话 "regularization technique")
> - **Reduce dimension first with PCA** (to ≤ N − c), or **feature selection**.
> - Use the **pseudo-inverse** of S_W, or **assume S_W is diagonal**.
> - **Collect more training samples.**

> 【中文】
> **原因**：LDA 需要 **S_W⁻¹**。S_W 是 N 个外积相加得到的 d×d 矩阵，**秩最多为 N − c**（两类时为 N − 2）。**d > N − 2 时 S_W 奇异，不可逆。** 即使 d 略小于 N，S_W 也是**病态**的：d(d+1)/2 个参数只用很少的样本估计，小特征值被低估，求逆时又被放大，于是 **w 对噪声极其敏感 → 训练精度高、测试精度差**（**过拟合**，即维数灾难 / 峰值现象）。
> **补救：**
> - **正则化**：用 **S_W + λI** 代替 S_W，矩阵变成正定、可逆；**λ 用交叉验证选**。
> - **先用 PCA 降维**（降到 ≤ N − c），或做**特征选择**。
> - 用 S_W 的**伪逆**，或**假设 S_W 是对角阵**。
> - **增加训练样本。**

**A4 ★★ Main differences between LDA and linear SVM.（23-24 S2）**
（LDA 和线性 SVM 的主要区别。）

| 维度 | LDA | Linear SVM |
|---|---|---|
| Criterion（准则） | **Maximize Fisher ratio** (between/within scatter of projections)；最大化 Fisher 比（投影后类间 / 类内散度） | **Maximize the margin of separation 2/‖w‖**；最大化分类间隔 |
| Samples used（用到的样本） | **All samples**, through class means and scatter matrices；全部样本，通过均值和散度矩阵 | **Only the support vectors** near the boundary；只用边界附近的支持向量 |
| Assumption（假设） | Implicitly **Gaussian classes with equal covariance** (then Bayes-optimal)；隐含高斯、协方差相同（此时 Bayes 最优） | **No distributional assumption**; purely geometric；无分布假设，纯几何 |
| Solution（求解） | **Closed form** w = S_W⁻¹(m₁−m₂)；闭式解 | **Convex quadratic programming** (dual problem, Lagrange multipliers)；凸二次规划（对偶、拉格朗日乘子） |
| Outliers（离群点） | Far-away outliers **still shift means/scatter**；远处离群点仍会拉动均值和散度 | Correctly classified points far from the boundary **have no effect**；远离边界且分对的点没有影响 |
| Non-separable / nonlinear（不可分 / 非线性） | No special mechanism；没有专门机制 | **Slack variables; kernel trick**；松弛变量、核技巧 |
| High-dim small sample（高维小样本） | **S_W singular, needs regularization**；S_W 奇异，需正则化 | **Margin maximization acts as regularization**；最大化间隔本身就是正则化 |

> One sentence: **LDA is a statistics-based method using all samples' means and scatters; SVM is a geometry-based method determined only by the boundary samples.**
>
> 【中文】一句话总结：**LDA 是基于统计的方法，用全部样本的均值和散度；SVM 是基于几何的方法，只由边界上的样本决定。**

**A5 ★ How to decide the confidence of a classification result?（每年计算题附带）**
（如何判断分类结果的置信度？）

> The **distance from x to the decision boundary is r = |g(x)|/‖w‖**. A sample **far from the boundary is classified with high confidence**; a sample **close to it may flip class** under small perturbations of the data or estimation errors of w. **For LDA, compare r with the distance of the class mean |g(mᵢ)|/‖w‖; for SVM, compare |g(x)| with 1** (i.e. r with the half-margin 1/‖w‖).

> 【中文】**x 到决策边界的距离 r = |g(x)|/‖w‖**。**离边界远的样本，分类置信度高**；**离边界近的样本**，数据稍有扰动或 w 有估计误差，**类别就可能翻转**。**LDA 把 r 和类均值到边界的距离 |g(mᵢ)|/‖w‖ 比较；SVM 把 |g(x)| 和 1 比较**（即 r 和半间隔 1/‖w‖ 比较）。

---

#### B. SVM

**B1 ★★★ Formulate the primal problem of the linear SVM (separable case) and justify it.（23-24 S2，8 分）**
（写出线性 SVM 在可分情形下的原问题，并说明理由。）

> **Constraints.** If the samples are separable, there exists a hyperplane wᵀx + b = 0 separating them. Since w and b can be scaled freely, we fix the scale so that **the closest samples satisfy |wᵀx + b| = 1 (canonical hyperplane)**. Then wᵀxᵢ + b ≥ 1 for dᵢ = +1 and ≤ −1 for dᵢ = −1, i.e. **dᵢ(wᵀxᵢ + b) ≥ 1, i = 1…N**. This requires **every training sample to be classified correctly** ("correctly").
> **Objective.** The distance of a sample to the hyperplane is |wᵀx+b|/‖w‖, so **the margin of separation is 2/‖w‖**. **Maximizing the margin ⇔ minimizing ‖w‖ ⇔ minimize Φ(w) = ½wᵀw** ("confidently"; the ½ and the square make it differentiable and a convex QP).
> **Justification.** Among the infinitely many separating hyperplanes, **the one with the largest margin is farthest from both classes, so it is the most robust to noise and generalizes best**. The problem is a **convex quadratic program with linear constraints, so it has a unique global optimum**; it is usually solved via the dual with Lagrange multipliers, and **only samples with αᵢ > 0 (support vectors) determine w**.

> 【中文】
> **约束。** 样本可分时，存在超平面 wᵀx + b = 0 把它们分开。w 和 b 可以任意缩放，所以固定尺度，使**离超平面最近的样本满足 |wᵀx + b| = 1（规范超平面）**。于是 dᵢ = +1 时 wᵀxᵢ + b ≥ 1，dᵢ = −1 时 ≤ −1，合写为 **dᵢ(wᵀxᵢ + b) ≥ 1，i = 1…N**。这要求**每个训练样本都分对**（"correctly"）。
> **目标。** 样本到超平面的距离是 |wᵀx+b|/‖w‖，所以**分类间隔是 2/‖w‖**。**最大化间隔 ⇔ 最小化 ‖w‖ ⇔ 最小化 Φ(w) = ½wᵀw**（"confidently"；½ 和平方使它可导，成为凸二次规划）。
> **理由。** 能分开两类的超平面有无数个，**间隔最大的那个离两类都最远，对噪声最稳健，泛化最好**。这是**带线性约束的凸二次规划，有唯一的全局最优解**；通常用拉格朗日乘子通过对偶问题求解，**只有 αᵢ > 0 的样本（支持向量）决定 w**。

**B2 ★★★ Primal problem for non-separable data.（24-25 S2，10 分）**
（写出不可分情形下的原问题。）

> **Minimize Φ(w, ξ) = ½wᵀw + C Σ ξᵢ subject to dᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ and ξᵢ ≥ 0.**
> Then explain ξ and C as in the model answer of §4.3, and add: **ξᵢ ≥ 0 prevents rewarding well-classified samples with negative slack; the problem is still a convex QP.**

> 【中文】**最小化 Φ(w, ξ) = ½wᵀw + C Σ ξᵢ，约束为 dᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ 且 ξᵢ ≥ 0。** 然后照 §4.3 的好答案解释 ξ 和 C，再补一句：**ξᵢ ≥ 0 防止用负的松弛"奖励"分得好的样本；问题仍是凸二次规划。**

**B3 ★★★ Roles of ξ and C**（ξ 和 C 的作用）：见 §4.3 示范答案（已附中文）。

**B4 ★★ Why can only support vectors with 0 < α < C be used to compute b?（老师 week9 强调）**
（为什么只能用 0 < α < C 的支持向量求 b？）

> By the **KKT conditions**, samples with **0 < αᵢ < C have ξᵢ = 0 and lie exactly on the margin hyperplanes**, so **dᵢ(wᵀxᵢ + b) = 1 holds with equality** and gives **b = dᵢ − wᵀxᵢ**. Samples with **αᵢ = C may lie inside the margin or be misclassified (ξᵢ > 0)**, so the equality does not hold and **they would give a wrong b**. In practice, **b is averaged over all margin support vectors** to reduce rounding errors.

> 【中文】由 **KKT 条件**，**0 < αᵢ < C 的样本 ξᵢ = 0，恰好落在间隔超平面上**，所以 **dᵢ(wᵀxᵢ + b) = 1 严格成立**，可得 **b = dᵢ − wᵀxᵢ**。**αᵢ = C 的样本可能在间隔内或被错分（ξᵢ > 0）**，等式不成立，**用它们会算出错误的 b**。实际中**对所有间隔线上的支持向量求出的 b 取平均**，以减小舍入误差。

**B5 ★ What are support vectors and why does the SVM depend only on them?**
（什么是支持向量？为什么 SVM 只依赖它们？）

> Support vectors are the training samples with **nonzero Lagrange multipliers αᵢ > 0**; they **lie on the margin** (or violate it in the non-separable case). Since **w = Σαᵢdᵢxᵢ and αᵢ = 0 for all other samples**, **removing non-support vectors does not change the classifier**. This makes SVM **sparse and robust to samples far from the boundary**.

> 【中文】支持向量是**拉格朗日乘子非零（αᵢ > 0）**的训练样本；它们**落在间隔线上**（不可分时也包括违反间隔的样本）。因为 **w = Σαᵢdᵢxᵢ，其他样本的 αᵢ 都为 0**，所以**删掉非支持向量不会改变分类器**。这使 SVM **具有稀疏性，并且对远离边界的样本稳健**。

---

#### C. Bayes / 朴素贝叶斯 / GMM

**C1 ★★★ What is the naive assumption? Why does Naive Bayes work well even if it is violated?**
（什么是朴素假设？为什么假设不成立时朴素贝叶斯仍然好用？）

> **Assumption:** given the class, **features are conditionally independent: p(x|ωᵢ) = Πⱼ p(xⱼ|ωᵢ)**. For Gaussian features this is equivalent to **setting the off-diagonal elements of the covariance matrix to zero**.
> **Why useful:** a full multivariate Gaussian needs a d-dim mean and **d(d+1)/2 covariance entries per class**; with few samples these cannot be estimated reliably (and the covariance may be singular). Under independence **only 2d parameters per class** are needed (d means and d variances), each estimated from 1-D data, with **no matrix inversion**. This **drastically reduces variance**, so NB often performs well in **small-sample or high-dimensional problems (e.g. text)** even though the assumption rarely holds exactly — **classification only needs the correct ranking of posteriors, not exact probabilities**.
> **Limitation:** **strongly correlated features are double-counted**; e.g. a test sample identical to a training sample may still be classified differently because NB uses only marginal statistics.

> 【中文】
> **假设**：给定类别后，**各特征条件独立：p(x|ωᵢ) = Πⱼ p(xⱼ|ωᵢ)**。对高斯特征，这等价于**把协方差矩阵的非对角元置为 0**。
> **为什么有用**：完整的多元高斯每类要估计 d 维均值和 **d(d+1)/2 个协方差元素**；样本少时估计不可靠（协方差还可能奇异）。独立假设下**每类只需 2d 个参数**（d 个均值、d 个方差），每个都用一维数据估计，**不需要求逆**。这**大幅降低了估计的方差**，所以即使假设很少严格成立，NB 在**小样本或高维问题（如文本）**上常常表现很好——**分类只要求后验的排序正确，不要求概率值准确**。
> **局限**：**强相关的特征会被重复计数**；例如一个与某训练样本完全相同的测试样本，仍可能被判成别的类，因为 NB 只用了边缘统计量。

**C2 ★★ Which Naive Bayes variant for which data?**
（哪种朴素贝叶斯适用于哪种数据？）

> **Gaussian NB for continuous features** (each feature modelled by a 1-D normal density per class); **Bernoulli NB for binary/categorical two-valued features** (P(xⱼ|ω) estimated by frequency counts, **including the probability of the feature being absent**); **Multinomial NB for word counts in text classification** (**P(w|ω) = (count+1)/(N_ω+d)**).

> 【中文】**Gaussian NB 用于连续特征**（每类每个特征用一个一元正态密度建模）；**Bernoulli NB 用于二值 / 二类别特征**（P(xⱼ|ω) 用频数估计，**特征"不出现"的概率也要算进去**）；**Multinomial NB 用于文本分类中的词频**（**P(w|ω) = (count+1)/(N_ω+d)**）。

**C3 ★★ Why Laplace smoothing?**
（为什么要用 Laplace 平滑？）

> If a feature value (word) **never appears with a class in training, its estimated probability is 0** and **the product of likelihoods becomes 0 regardless of other evidence**, so classes cannot be compared. **Laplace (add-one) smoothing adds 1 to every count and d (vocabulary size) to the denominator**, so **every probability is positive** while the estimates **remain close to the frequencies when counts are large**.

> 【中文】如果某个特征值（词）**在训练集中从没和某类一起出现，它的估计概率就是 0**，**似然连乘结果变成 0，其他证据再强也没用**，各类无法比较。**Laplace（加一）平滑给每个计数加 1、分母加 d（词表大小）**，使**每个概率都为正**，而**计数大时估计值仍接近原频率**。

**C4 ★★ Bayes decision rule & how it relates to LDA.**
（Bayes 决策规则，以及它与 LDA 的关系。）

> **Assign x to ωᵢ if P(ωᵢ|x) > P(ωⱼ|x) for all j ≠ i**, equivalently **p(x|ωᵢ)P(ωᵢ) is the largest (p(x) is common)**. With Gaussian class-conditional densities, **gᵢ(x) = −½ln|Σᵢ| − ½(x−μᵢ)ᵀΣᵢ⁻¹(x−μᵢ) + lnP(ωᵢ)**, giving a **quadratic boundary**. **If all classes share the same covariance, the quadratic terms cancel and the boundary becomes linear — the same direction as Fisher LDA.** It **minimizes the probability of error** when the densities are correct.

> 【中文】**若对所有 j ≠ i 都有 P(ωᵢ|x) > P(ωⱼ|x)，就把 x 判给 ωᵢ**；等价于 **p(x|ωᵢ)P(ωᵢ) 最大（p(x) 各类相同）**。类条件密度为高斯时，**gᵢ(x) = −½ln|Σᵢ| − ½(x−μᵢ)ᵀΣᵢ⁻¹(x−μᵢ) + lnP(ωᵢ)**，边界是**二次曲线**。**若各类协方差相同，二次项抵消，边界变成直线，方向与 Fisher LDA 相同。** 密度估计正确时，它**使错误概率最小**。

**C5 ★ What is a Gaussian mixture model and how is it trained (EM)?**
（什么是高斯混合模型？如何用 EM 训练？）

> When samples of one class (or one cluster) **come from more than one distribution**, **p(x) = Σₖ πₖ N(x|μₖ, Σₖ)**. The parameters are estimated by the **EM algorithm**: initialize (μ, Σ, π); **E-step**: compute the **responsibility γₖ(x) = probability that x belongs to component k**; **M-step**: **update μₖ, Σₖ, πₖ as γ-weighted means/covariances/fractions**; **repeat until the log-likelihood converges**. **The number of components M is a hyperparameter** (老师 week7 强调 "we don't know M in practice").

> 【中文】当一类（或一个簇）的样本**来自不止一个分布**时，**p(x) = Σₖ πₖ N(x|μₖ, Σₖ)**。参数用 **EM 算法**估计：初始化 (μ, Σ, π)；**E 步**：计算**责任度 γₖ(x)，即 x 属于第 k 个分量的概率**；**M 步**：**用 γ 加权更新 μₖ、Σₖ、πₖ**（加权均值、加权协方差、加权比例）；**重复直到对数似然收敛**。**分量个数 M 是超参数**。

---

#### D. 决策树 / 随机森林 / 回归

**D1 ★★ How is a classification tree built? How to choose a split?**
（分类树是怎么建的？如何选择划分？）

> At each node, **consider every feature and every candidate threshold** (for continuous features, **midpoints of sorted adjacent values**). Choose the split that gives the **minimum weighted impurity of the children** (**maximum impurity reduction**), measured by **entropy −ΣP log₂P or Gini** (½(1−ΣP²) in the course slides; 1−ΣP² in textbooks). **Recursively split the children until a stopping criterion is met; label each leaf by the majority class.**

> 【中文】在每个节点上，**考察每个特征和每个候选阈值**（连续特征取**排序后相邻取值的中点**）。选使**子节点加权不纯度最小**（**不纯度下降最大**）的划分，不纯度用**熵 −ΣP log₂P 或 Gini**（课件版 ½(1−ΣP²)，教科书版 1−ΣP²）衡量。**对子节点递归划分，直到满足停止准则；每个叶节点按多数类标注。**

**D2 ★★ When to stop splitting / why prune?**
（何时停止划分？为什么要剪枝？）

> **A fully grown tree** (each leaf pure, possibly one sample) **overfits** — it is just a lookup table of the training data; **stopping too early underfits**. **Stopping criteria: impurity reduction below a threshold, node has fewer than a minimum number of samples, maximum depth, or validation error stops decreasing (cross-validation).** **Pruning: grow the full tree first, then merge neighbouring leaves whose removal causes only a small increase in impurity/error**, giving a **smaller tree with better generalization**. (课件：先 grow 后 prune 两阶段)

> 【中文】**完全长成的树**（每个叶子都纯，可能只有一个样本）**会过拟合**——它只是训练数据的查找表；**停得太早则欠拟合**。**停止准则：不纯度下降小于阈值、节点样本数少于下限、达到最大深度，或验证误差不再下降（交叉验证）。** **剪枝：先把整棵树长满，再合并那些去掉后不纯度 / 误差只略微增加的相邻叶子**，得到**更小、泛化更好的树**。

**D3 ★★ Explain random forest.（老师强调 diversity）**
（解释随机森林。）

> Random forest is an **ensemble of decision trees** based on **bagging (bootstrap aggregating)**. (1) **Bootstrapping:** each tree is trained on a **bootstrap sample drawn with replacement** from the training set. (2) **Random feature subsets:** each tree (split) uses **only a random subset of the features**. These two sources of randomness make the trees **diverse / weakly correlated**. (3) **Aggregating:** the final class is decided by **majority vote** of the trees. **Averaging many diverse, low-bias but high-variance trees reduces variance**, so a random forest **overfits less and is more accurate and robust than a single tree**; it also gives **feature importance**. The cost is **loss of interpretability and higher computation**.

> 【中文】随机森林是基于 **bagging（bootstrap aggregating）** 的**决策树集成**。(1) **自助采样**：每棵树用从训练集中**有放回抽取的 bootstrap 样本**训练。(2) **随机特征子集**：每棵树（每次划分）**只用随机选出的一部分特征**。这两处随机性使各棵树**多样、相关性弱**。(3) **聚合**：最终类别由各棵树**多数投票**决定。**把许多多样的、低偏差高方差的树平均，可以降低方差**，所以随机森林**比单棵树更不容易过拟合，更准确、更稳健**；还能给出**特征重要性**。代价是**失去可解释性、计算量更大**。

**D4 ★★ Ridge vs Lasso regression.（从未考，老师总复习专门提）**
（岭回归与 Lasso 回归的比较。）

> Both **add a penalty to the OLS loss to avoid overfitting (regularization)**, useful when **predictors are many or collinear (XᵀX nearly singular)**.
> - **Ridge (L2):** minimize Σ(yᵢ − ŷᵢ)² + **λ‖β‖₂²**. **Shrinks coefficients toward zero but not exactly zero**; has a **closed-form solution β = (XᵀX + λI)⁻¹Xᵀy**; **handles multicollinearity**.
> - **Lasso (L1):** minimize Σ(yᵢ − ŷᵢ)² + **λ‖β‖₁**. Shrinks some coefficients **exactly to zero**, so it performs **variable (feature) selection** and gives a **sparse, interpretable model**; **no analytic solution, solved numerically**.
> - **λ controls the strength** (λ = 0 → OLS; large λ → stronger shrinkage); **chosen by cross-validation**.

> 【中文】两者都**在 OLS 损失上加惩罚项来防止过拟合（正则化）**，适用于**自变量很多或共线（XᵀX 接近奇异）**的情况。
> - **Ridge（L2）**：最小化 Σ(yᵢ − ŷᵢ)² + **λ‖β‖₂²**。**把系数向 0 收缩，但不会恰好为 0**；有**闭式解 β = (XᵀX + λI)⁻¹Xᵀy**；**能处理多重共线性**。
> - **Lasso（L1）**：最小化 Σ(yᵢ − ŷᵢ)² + **λ‖β‖₁**。会把部分系数**压缩到恰好为 0**，因此能做**变量（特征）选择**，得到**稀疏、可解释的模型**；**没有解析解，要数值求解**。
> - **λ 控制正则化强度**（λ = 0 即 OLS；λ 越大收缩越强）；**用交叉验证选**。

**D5 ★ R² and adjusted R².**
（R² 与调整 R²。）

> **R² = 1 − SS_res/SS_tot is the fraction of variance of y explained by the model.** It **never decreases when variables are added, even insignificant ones**; **adjusted R² penalizes the number of predictors** and **increases only if a new variable improves the model more than expected by chance**.

> 【中文】**R² = 1 − SS_res/SS_tot，是模型解释的 y 的方差比例。** **增加变量时它只增不减，哪怕变量毫无用处**；**调整 R² 对自变量个数进行惩罚**，**只有新变量带来的改善超过偶然水平时才会上升**。

---

#### E. 分类器评估

**E1 ★★★ Discuss procedures and metrics for evaluating a classifier.（21-22 S2 考过）**
（讨论评估分类器的过程和指标。）

> **Procedures:**
> 1. ***Hold-out:*** split data into **training and test sets** (e.g. 2/3 : 1/3). Simple, but **the estimate depends on one random split and wastes data**; **stratified sampling keeps class proportions**.
> 2. ***Repeated hold-out:*** **repeat random splits several times and average** → less variance.
> 3. ***k-fold cross-validation:*** divide data into k folds; **each fold is used once as test set and the other k−1 for training; average the k results**. **Every sample is used for both training and testing.**
> 4. ***Leave-one-out (LOOCV):*** **k = N**; nearly unbiased but **expensive** and high variance.
> 5. ***Repeated k-fold CV:*** **repeat k-fold with different random partitions and average**, reducing the noise of a single k-fold run.
>
> **Metrics:** **accuracy / error rate**; **sensitivity (recall, TPR) and specificity (TNR)**; **precision**; **F-score** (harmonic mean of precision and recall, **good for imbalanced data**); **ROC curve** (TPR vs FPR as the decision threshold varies) and **AUC** (area under ROC; 1 = perfect, 0.5 = random; **threshold-independent**).
> **Choose by application**: in cancer diagnosis **recall matters more** (missing a patient is costly); **with class imbalance accuracy is misleading, use F-score or AUC**.

> 【中文】
> **评估过程：**
> 1. ***留出法***：把数据分成**训练集和测试集**（如 2/3 : 1/3）。简单，但**结果依赖一次随机划分，且浪费数据**；**分层抽样可以保持各类比例**。
> 2. ***重复留出法***：**多次随机划分再取平均** → 方差更小。
> 3. ***k 折交叉验证***：把数据分成 k 折，**每一折轮流当一次测试集，其余 k−1 折训练，k 个结果取平均**。**每个样本既参与训练也参与测试。**
> 4. ***留一法（LOOCV）***：**k = N**；几乎无偏，但**计算量大**，方差也大。
> 5. ***重复 k 折交叉验证***：**用不同的随机划分重复做 k 折再取平均**，减小单次 k 折结果的波动。
>
> **评估指标**：**准确率 / 错误率**；**灵敏度（召回率、TPR）和特异度（TNR）**；**精确率**；**F 值**（精确率和召回率的调和平均，**适合类别不平衡数据**）；**ROC 曲线**（决策阈值变化时 TPR 对 FPR 的曲线）和 **AUC**（ROC 下面积；1 为完美，0.5 为随机；**与阈值无关**）。
> **按应用选指标**：癌症诊断中**召回率更重要**（漏诊代价大）；**类别不平衡时准确率会误导，要用 F 值或 AUC**。

**E2 ★★ Describe k-fold cross-validation and its rationale.（22-23 S2）**
（描述 k 折交叉验证及其原理。）

> Procedure as above. **Rationale:** **the performance on training data is optimistic; what matters is performance on unseen data.** **A single hold-out estimate depends heavily on which samples happen to be in the test set and wastes data** when the dataset is small. **k-fold CV uses every sample for testing exactly once and for training k−1 times**, so the estimate is **more reliable (lower variance) and data-efficient**. It is also used to **select hyperparameters** (C in SVM, λ in ridge, tree size). **Typical k = 5 or 10; larger k → less bias but more computation.** (老师打比方：考一次试不能代表真实水平，要多次取平均)

> 【中文】步骤同上。**原理**：**训练集上的性能偏乐观，真正重要的是在没见过的数据上的性能。** **单次留出法的结果很依赖哪些样本恰好落进测试集**，数据少时还浪费数据。**k 折交叉验证让每个样本恰好当一次测试样本、当 k−1 次训练样本**，所以估计**更可靠（方差更小）、数据利用更充分**。它还用来**选超参数**（SVM 的 C、Ridge 的 λ、树的大小）。**常用 k = 5 或 10；k 越大偏差越小，但计算量越大。**

**E3 ★★ Precision vs recall / why accuracy can be misleading; ROC & AUC.**
（精确率与召回率；为什么准确率会误导；ROC 与 AUC。）

> With 95% negatives, **a classifier that predicts everything negative has 95% accuracy but recall 0**. **Precision = of the samples predicted positive, how many are truly positive; recall = of the truly positive samples, how many are found.** **Raising the threshold usually increases precision and decreases recall.** **The ROC curve plots TPR against FPR over all thresholds; the closer it is to the top-left corner, the better; AUC summarizes it as a single threshold-independent number.**

> 【中文】负类占 95% 时，**把所有样本都判为负类的分类器准确率有 95%，但召回率为 0**。**精确率 = 预测为正的样本中真正为正的比例；召回率 = 真正为正的样本中被找出来的比例。** **提高阈值通常使精确率上升、召回率下降。** **ROC 曲线画出所有阈值下 TPR 对 FPR 的关系，越靠近左上角越好；AUC 把它概括成一个与阈值无关的数。**

---

#### F. 维数灾难与特征选择

**F1 ★★★ Explain the peaking phenomenon and how it guides data collection and feature selection.（23-24 S2，8 分；老师强调）**
（解释峰值现象，以及它对数据采集和特征选择的指导意义。）

> **What:** **For a fixed number of training samples N, as the number of features d increases, the classification performance first improves, then after a critical value d\* it deteriorates** — the error curve has a minimum ("peak" in performance).
> **Why:** more features add information, but **the number of classifier parameters grows with d** (e.g. d(d+1)/2 for a covariance matrix) **while N is fixed, so parameter estimates become unreliable**; **the data become sparse in high dimensions and the classifier starts fitting noise (overfitting)** — a manifestation of the **curse of dimensionality**.
> **Guidance for data collection:** **the number of training samples must be large relative to d** (rule of thumb: several, e.g. **5–10, samples per feature per class**). **With more samples, the peak shifts to larger d and the minimum error is lower**, so **if many features are to be used, more data must be collected**. **Adding measurements without adding samples does not help.**
> **Guidance for feature selection:** **with limited data, using all available features is not optimal**; **perform feature selection/extraction to reduce d to around the optimum**, keeping **discriminative, non-redundant features (max relevance, min redundancy)**. **The best subset size can be found by cross-validation.** This also **reduces computation and measurement cost**.

> 【中文】
> **是什么**：**训练样本数 N 固定时，随着特征数 d 增加，分类性能先变好，超过某个临界值 d\* 后反而变差**——错误率曲线有一个最低点（性能的"峰值"）。
> **为什么**：特征多了信息会增加，但**分类器的参数个数随 d 增长**（如协方差矩阵有 d(d+1)/2 个），**而 N 不变，参数估计变得不可靠**；**高维空间中数据变得稀疏，分类器开始拟合噪声（过拟合）**——这是**维数灾难**的一种表现。
> **对数据采集的指导**：**训练样本数相对 d 必须足够多**（经验法则：**每类每个特征 5～10 个样本**）。**样本越多，峰值点越往大 d 移，最低错误率也越低**，所以**要用很多特征，就必须采集更多数据**。**只增加测量项而不增加样本没有帮助。**
> **对特征选择的指导**：**数据有限时，用上全部特征并不是最优的**；**要做特征选择 / 提取，把 d 降到最优值附近**，保留**有判别力、不冗余的特征（最大相关、最小冗余）**。**最佳子集大小可以用交叉验证确定。** 这样还能**降低计算量和测量成本**。

**F2 ★★ What is the curse of dimensionality?**
（什么是维数灾难？）

> Difficulties arising with high-dimensional data. Two main aspects: (1) **data sparsity** — **the volume grows exponentially with d, so a fixed number of samples covers the space very sparsely** and not enough data are observed; (2) **distance concentration** — **all pairwise distances become nearly equal, so distance-based methods (KNN, K-means) lose discriminative power**. Consequences: **more parameters to estimate, overfitting, high computation**. Remedies: **feature selection/extraction, regularization, more data**.

> 【中文】处理高维数据时遇到的种种困难。主要有两方面：(1) **数据稀疏**——**空间体积随 d 指数增长，固定数量的样本覆盖空间非常稀疏**，观测到的数据不够；(2) **距离集中**——**所有两两距离变得几乎相等，基于距离的方法（KNN、K-means）失去区分能力**。后果：**要估计的参数更多、过拟合、计算量大**。补救：**特征选择 / 提取、正则化、增加数据**。

**F3 ★★ Feature selection vs feature extraction.**
（特征选择与特征提取的区别。）

> **Selection chooses a subset of the original features** (**keeps physical meaning, interpretable, reduces measurement cost**, but **discards the information of removed features**). **Extraction derives new features as functions (e.g. linear combinations) of all original features, e.g. PCA**; it **compresses information more efficiently** but **the new features lose physical meaning and all original features must still be measured**. Note: **information preserved is not necessarily discriminative information (PCA is unsupervised).**

> 【中文】**特征选择是从原始特征中挑一个子集**（**保留物理意义、可解释、降低测量成本**，但**丢掉了被删特征的信息**）。**特征提取是把全部原始特征变换（如线性组合）成新特征，例如 PCA**；它**压缩信息更高效**，但**新特征没有物理意义，而且所有原始特征仍然都要测量**。注意：**保留下来的信息不一定是有判别力的信息（PCA 是无监督的）。**

**F4 ★★★ Three types of feature subset selection methods with pros and cons.（23-24 S2，9 分）**
（三类特征子集选择方法及其优缺点。）

| Method | How | Pros | Cons |
|---|---|---|---|
| **Filter（过滤法）** | **No classifier in the loop**; evaluates subsets by **class-separability measures** (Mahalanobis distance, scatter-based trace(S_W⁻¹S_B), Fisher ratio, mutual information)；不把分类器放进循环，用类可分性度量评价子集 | **Fast, scalable to high dimensions, independent of classifier, less overfitting**；快、能处理高维、与分类器无关、不易过拟合 | **Not optimized for the final classifier**; individual-feature filters **ignore redundancy and interactions**; distribution assumptions may not hold；不针对最终分类器优化，单特征过滤忽略冗余和交互，分布假设可能不成立 |
| **Wrapper（包裹法）** | **Classifier in the loop**; evaluates each candidate subset by **(validation/CV) classification accuracy**; combined with a **search strategy (SFS, SBE, GA)**；分类器在循环里，用验证集 / 交叉验证的分类精度评价子集，配合搜索策略 | **Tailored to the classifier, accounts for feature interactions, usually highest accuracy**；针对分类器、考虑特征交互、精度通常最高 | **Very expensive** (train many classifiers); **may overfit the validation set** with few samples; **subset depends on the classifier**；计算量很大、样本少时会对验证集过拟合、选出的子集依赖分类器 |
| **Embedded（嵌入法）** | **Selection happens inside model training** (**Lasso L1 penalty sets coefficients to zero**; decision tree/random forest split choice)；在模型训练过程中完成选择（Lasso 把系数压到 0；决策树 / 随机森林选划分） | **Cheaper than wrapper (one training), considers interactions**；比包裹法便宜（只训练一次），考虑交互 | **Specific to the learning algorithm**; **needs tuning of regularization parameter**；依赖特定学习算法，需要调正则化参数 |

> Also mention: **a selection algorithm = search algorithm** (exhaustive / sequential forward selection / sequential backward elimination) **+ evaluation criterion** (filter vs wrapper). Don't confuse these two classifications.
>
> 【中文】再补一句：**特征选择算法 = 搜索算法**（穷举 / 顺序前向选择 / 顺序后向消除）**+ 评价准则**（filter 还是 wrapper）。不要把这两种分类方式混为一谈。

**F5 ★★ Draw block diagrams of filter and wrapper.（21-22 S2）**（画出 filter 和 wrapper 的框图。）考场上画两个框图：

```
Filter:   Full feature set → [Search: generate subset] → [Separability measure] → stop? —Yes→ Selected subset
                                    ↑______________________ No ________________|
Wrapper:  Full feature set → [Search: generate subset] → [Train classifier + validate accuracy] → stop? —Yes→ Selected subset
                                    ↑______________________ No ______________________________|
```
中文对照：全部特征 → [搜索：生成候选子集] → [Filter：可分性度量 / Wrapper：训练分类器并验证精度] → 是否停止？——是 → 选定子集；否 → 回到搜索。
**停止条件**：达到预定特征数；准则不再提升；（wrapper）验证集精度达到最优。**两图唯一的区别是评价那一格**：filter 用可分性度量，wrapper 用分类器精度。

**F6 ★★ Describe sequential forward selection (or backward elimination).**
（描述顺序前向选择，或顺序后向消除。）

> **SFS (bottom-up): start with an empty set.** Step 1: **evaluate each feature alone** (separability or classifier accuracy); **select the best**. Step k: **for each remaining feature, evaluate the current subset plus that feature; add the one giving the largest improvement.** **Stop when the desired number of features is reached or the criterion no longer improves.** **SBE (top-down): start with all features; at each step remove the feature whose removal least degrades the criterion.** Both are **sub-optimal (greedy, cannot undo a choice — "nesting effect")** but **much cheaper than exhaustive search (2^d subsets)**.

> 【中文】**SFS（自底向上）：从空集开始。** 第 1 步：**逐个单独评价每个特征**（可分性或分类精度），**选出最好的**。第 k 步：**对每个剩余特征，评价"当前子集 + 该特征"，加入提升最大的那个。** **达到预定特征数或准则不再提升时停止。** **SBE（自顶向下）：从全部特征开始，每步删去"删掉后准则下降最少"的那个特征。** 两者都是**次优的（贪心，选了就不能撤回——"嵌套效应"）**，但**比穷举（2^d 个子集）便宜得多**。

**F7 ★ Individual feature evaluation: Fisher ratio and mutual information.**
（单特征评价：Fisher 比与互信息。）

> **Fisher ratio FR = (μ₁−μ₂)²/(σ₁²+σ₂²)**: **large between-class difference, small within-class variance → more discriminative**; **for continuous features**. **Mutual information I(X;C) = H(X) + H(C) − H(X,C)**: **how much knowing the feature reduces uncertainty about the class**; **for discrete features**. Limitation: **ranking features individually ignores redundancy — top-ranked features may be highly correlated.**

> 【中文】**Fisher 比 FR = (μ₁−μ₂)²/(σ₁²+σ₂²)**：**类间差异大、类内方差小 → 判别力强**；**用于连续特征**。**互信息 I(X;C) = H(X) + H(C) − H(X,C)**：**知道这个特征后，对类别的不确定性减少了多少**；**用于离散特征**。局限：**逐个给特征排名忽略了冗余——排名靠前的几个特征可能高度相关。**

**F8 ★★ "Suggest a feature selection method for this dataset and describe the procedure."（22-23 S2）**
（为这个数据集推荐一种特征选择方法，并描述步骤。）

> 套路：**根据数据规模选**。**特征维数高、样本多 → filter**（或 filter 预筛 + wrapper 精选）；**维数中等、要求精度 → wrapper + SFS**；**线性模型 → Lasso（embedded）**。然后写 F6 的步骤，并说明**评价准则、停止条件**，以及**用交叉验证避免过拟合**。
>
> 英文句式：*"Since the dataset has a large number of features, I suggest a **filter method for pre-screening followed by a wrapper with sequential forward selection**. The evaluation criterion is the **cross-validated accuracy** of the classifier, and the search stops when **adding a feature no longer improves** the accuracy."*（由于特征很多，建议先用 filter 预筛，再用 wrapper + 顺序前向选择；评价准则是分类器的交叉验证精度，加入特征不再提升精度时停止。）

---

#### G. 数据准备（23-24 S1 整题，近年没考，背清单即可）

**G1 Data types（数据类型）**
> **Categorical / Nominal**: labels with **no order** (gender, colour). **Ordinal**: **ordered, but intervals are not meaningful** (satisfaction level, grade). **Interval**: **differences are meaningful, no true zero** (temperature in °C). **Ratio**: **true zero, ratios are meaningful** (height, income). Numerical data can also be **discrete or continuous**. **The data type determines which operations are valid** — e.g. **no mean for nominal data, use the mode; use the median for ordinal data**.
>
> 【中文】**类别 / 名义型**：标签**无序**（性别、颜色）。**序数型**：**有序但间隔无意义**（满意度、成绩等级）。**区间型**：**差值有意义、没有绝对零点**（摄氏温度）。**比率型**：**有绝对零点、比值有意义**（身高、收入）。数值型又分**离散和连续**。老师强调：**知道类型才知道能做什么运算**——如 **nominal 不能求均值，用众数；ordinal 用中位数**。

**G2 Outliers（离群点）**
> **Definition**: samples that **deviate markedly from the others**; they may be **errors or genuine rare values**. **Detection**: **z-score > 3, IQR rule (outside Q₁ − 1.5·IQR or Q₃ + 1.5·IQR), box plot, distance/density-based methods.** **Treatment**: **first identify the cause**; **remove**; **cap / winsorize**; **treat as missing and impute**; **log transform**; **use robust methods such as the median**; **keep natural outliers** (e.g. in fraud detection).
>
> 【中文】**定义**：**明显偏离其他样本**的值，可能是**错误，也可能是真实的罕见值**。**检测**：**z-score > 3、IQR 规则（小于 Q₁ − 1.5·IQR 或大于 Q₃ + 1.5·IQR）、箱线图、基于距离 / 密度的方法。** **处理**：**先判断成因**；**删除**；**截尾 / 缩尾**；**当作缺失值插补**；**对数变换**；**用中位数等稳健方法**；**自然产生的离群点要保留**（如欺诈检测）。

**G3 Missing values（缺失值）**
> **Eliminate samples** (few and random missing values); **eliminate the feature** (high missing ratio); **impute with mean / median / mode** (can be computed **within each class or group of similar samples**); **model-based imputation** (KNN, regression); **add a "missing" indicator variable**. **Imputation statistics must be computed on the training set only.**
>
> 【中文】**删除样本**（缺失少且随机）；**删除特征**（缺失比例高）；**用均值 / 中位数 / 众数插补**（可**按类别或相似样本分组**分别算）；**用模型插补**（KNN、回归）；**增加"是否缺失"的指示变量**。**插补用的统计量只能在训练集上算。**

**G4 Different ranges/units（量纲不同）**
> **Why**: **features with large ranges dominate distances**; **distance-based and gradient-based methods are affected**. **Min-max normalization (x − min)/(max − min) → [0, 1]** (**sensitive to outliers**). **z-score standardization (x − μ)/σ → mean 0, variance 1.** **Scaling parameters are estimated on the training set only.** **Tree models do not need scaling.**
>
> 【中文】**为什么**：**取值范围大的特征主导距离**；**基于距离和梯度的方法受影响**。**min-max 归一化 (x − min)/(max − min) → [0, 1]**（**对离群点敏感**）。**z-score 标准化 (x − μ)/σ → 均值 0、方差 1。** **缩放参数只从训练集估计。** **树模型不需要缩放。**

---

#### H. 聚类（每年都有一道，论述占 5～12 分）

**H1 ★★★ List and describe four types of clustering methods and their application scenarios.（23-24 S2、25-26 S1）**
（列出并描述四类聚类方法及其应用场景。）

> 1. **Centroid-based (partitioning), e.g. K-means.** **Predefine K; each cluster is represented by its centroid; alternately assign samples to the nearest centroid and update centroids to minimize WCSS.** **Fast and simple**, but **needs K, sensitive to initialization and outliers, finds only convex, spherical, similar-sized clusters**. *Scenario:* **large numerical datasets with compact clusters** — customer segmentation, image colour quantization.
> 2. **Hierarchical (agglomerative), e.g. single/complete/average/centroid linkage.** **Start with each sample as a cluster and repeatedly merge the two closest clusters, giving a dendrogram that can be cut at any level.** **No need to predefine K, shows hierarchy**, but **O(n²) memory/time and merges cannot be undone**. *Scenario:* **small datasets where hierarchy matters** — gene expression, taxonomy, document topics.
> 3. **Density-based, e.g. DBSCAN.** **Connects high-density regions; points in low-density regions are noise.** **Finds arbitrarily shaped clusters, determines the number of clusters automatically, robust to noise**; **sensitive to ε and MinPts and to varying densities**. *Scenario:* **spatial/GPS data, anomaly detection, ring- or moon-shaped clusters**.
> 4. **Distribution-based, e.g. Gaussian mixture model with EM.** **Assumes data are generated by a mixture of distributions; gives soft (probabilistic) assignments** and can model **elliptical, overlapping clusters of different sizes**; **needs the number of components and may converge to local optima**. *Scenario:* **overlapping clusters needing probability outputs** — speaker modelling, image segmentation.

> 【中文】
> 1. **基于质心（划分式），如 K-means。** **预先设定 K；每个簇用质心代表；交替地把样本分给最近的质心、再更新质心，使 WCSS 最小。** **快速简单**，但**需要给定 K，对初值和离群点敏感，只能找凸的、球状的、大小相近的簇**。*场景*：**簇紧凑的大规模数值数据**——客户细分、图像颜色量化。
> 2. **层次式（凝聚），如 single / complete / average / centroid linkage。** **开始时每个样本自成一簇，反复合并最近的两个簇，得到可以在任意高度切开的树状图。** **不需要预先给定 K，能展示层次结构**，但**需要 O(n²) 的内存和时间，合并之后不能撤回**。*场景*：**层次结构有意义的小数据集**——基因表达、生物分类、文档主题。
> 3. **基于密度，如 DBSCAN。** **把高密度区域连成簇，低密度区域的点是噪声。** **能找任意形状的簇，自动确定簇数，对噪声稳健**；**对 ε 和 MinPts 敏感，处理不了密度差别大的簇**。*场景*：**空间 / GPS 数据、异常检测、环形或月牙形的簇**。
> 4. **基于分布，如用 EM 训练的高斯混合模型。** **假设数据由若干分布混合生成；给出软的（概率）分配**，能刻画**椭圆形、相互重叠、大小不同的簇**；**需要给定分量个数，可能收敛到局部最优**。*场景*：**需要概率输出的重叠簇**——说话人建模、图像分割。

**H2 ★★★ Describe the DBSCAN procedure.（21-22 S2、23-24 S1）**
（描述 DBSCAN 的步骤。）

> Parameters: **neighbourhood radius ε and MinPts**.
> 1. For each point, count the points within its ε-neighbourhood. **Core point: at least MinPts points** (the slides say "more than MinPts" — state your convention); **border point: fewer than MinPts but within the ε-neighbourhood of a core point**; **noise: neither**.
> 2. **Pick an unvisited core point and start a new cluster.**
> 3. **Add all points in its ε-neighbourhood; for each newly added core point, add its neighbours as well (density-reachable), until the cluster cannot grow.**
> 4. **Repeat 2–3 until all core points are assigned.**
> 5. **Border points join the cluster of a neighbouring core point; noise points belong to no cluster.**
> Advantages: **arbitrary shapes, no K needed, identifies noise**. Limitations: **sensitive to ε/MinPts, struggles with clusters of different densities** and very high dimensions.

> 【中文】参数：**邻域半径 ε 和 MinPts**。
> 1. 对每个点数出它 ε 邻域内的点数。**核心点：至少 MinPts 个点**（课件写 "more than MinPts"，答题时写明你的约定）；**边界点：不足 MinPts 个，但落在某个核心点的 ε 邻域内**；**噪声点：两者都不是**。
> 2. **任选一个未访问的核心点，建立一个新簇。**
> 3. **把它 ε 邻域内的点全部加入；对新加入的核心点，再把它们的邻居也加入（密度可达），直到簇不再扩大。**
> 4. **重复 2–3，直到所有核心点都被分配。**
> 5. **边界点归入相邻核心点所在的簇；噪声点不属于任何簇。**
> 优点：**任意形状、不需要 K、能识别噪声**。局限：**对 ε / MinPts 敏感，处理不好密度不同的簇**，也不适合维数很高的数据。

**H3 ★★★ Choose a clustering algorithm for the data in the figure and justify.（23-24 S1）**
（为图中的数据选择聚类算法并说明理由。）

> 判断三件事：**簇的形状**（球状 → K-means/GMM；环形、月牙、细长 → DBSCAN 或 single linkage）；**有没有噪声**（有 → DBSCAN）；**簇数是否已知**（已知且簇紧凑 → K-means）。答题结构：选择 → **结合图的特征讲理由** → 写算法步骤。
>
> 英文示例：*"The data form **two concentric rings**: the clusters are **non-convex** and their **centroids nearly coincide**, so **K-means would cut each ring in half**. **DBSCAN grows clusters along dense regions** and can follow the ring shape, and it also **labels the scattered points as noise**. Therefore DBSCAN is preferred."*
> （数据是**两个同心圆环**：簇是**非凸**的，**两个质心几乎重合**，所以 **K-means 会把每个环切成两半**。**DBSCAN 沿高密度区域扩展簇**，能顺着环形走，还能**把零散的点标为噪声**。因此选 DBSCAN。）

**H4 ★★★ Describe K-means and its limitations; can it be used for irregular-shaped clusters?（24-25 S2）**
（描述 K-means 及其局限；它能用于形状不规则的簇吗？）

> Steps: **choose K; initialize K centroids (random samples); assign each sample to the nearest centroid; recompute each centroid as the mean of its samples; repeat until assignments no longer change.** Objective: **minimize WCSS**. **Converges to a local optimum, depends on initialization** (**run several times, keep the lowest WCSS**).
> Limitations: **K must be given; sensitive to initialization and outliers** (means are pulled); **assumes convex, spherical, similar-sized clusters because the boundary between two centroids is their perpendicular bisector (Voronoi partition)**. Therefore **for moon-shaped, ring-shaped or elongated clusters it generally fails**; **use DBSCAN, single-linkage hierarchical clustering, or GMM for elliptical clusters**.

> 【中文】步骤：**选定 K；初始化 K 个质心（随机挑样本）；把每个样本分给最近的质心；把每个质心重新算成该簇样本的均值；重复直到分配不再变化。** 目标：**使 WCSS 最小**。**只收敛到局部最优，结果依赖初值**（**多跑几次，取 WCSS 最小的一次**）。
> 局限：**必须给定 K；对初值和离群点敏感**（均值会被拉偏）；**默认簇是凸的、球状的、大小相近，因为两个质心之间的分界是它们的中垂线（Voronoi 划分）**。所以**对月牙形、环形或细长的簇一般会失败**；**应改用 DBSCAN、single linkage 层次聚类，椭圆形簇可用 GMM**。

**H5 ★★★ How to determine K?（24-25 S2、25-26 S2）**
（如何确定 K？）

> 1. **Elbow method:** **run K-means for K = 1…K_max, compute WCSS, plot WCSS vs K.** WCSS decreases monotonically; **the "elbow" where the decrease sharply slows indicates a suitable K**.
> 2. **Validity indices:** **compute the Silhouette coefficient (or Dunn, CH — larger better; DB — smaller better) for each K and choose the best**; more objective when the elbow is unclear.
> 3. **Domain knowledge** may fix K.
> **Do not choose the K with minimum WCSS: WCSS = 0 when K = N.**

> 【中文】
> 1. **肘部法**：**对 K = 1…K_max 分别跑 K-means，算出 WCSS，画 WCSS–K 曲线。** WCSS 单调下降；**下降速度突然变慢的"肘部"就是合适的 K**。
> 2. **评价指标**：**对每个 K 算 Silhouette 系数（或 Dunn、CH，越大越好；DB，越小越好），取最优的 K**；肘部不明显时更客观。
> 3. **领域知识**可能直接规定 K。
> **不能选 WCSS 最小的 K：K = N 时 WCSS = 0。**

**H6 ★★★ List five clustering evaluation metrics / explain Silhouette / Dunn.（23-24 S1、24-25 S1、25-26 S1）**
（列出五个聚类评价指标；解释 Silhouette / Dunn。）

> 五个指标用 §3.10 的公式，每个指标写：**定义（一句话或公式）+ 衡量 cohesion / separation 的方式 + 越大还是越小越好**。总结一句（老师 week13 原话的意思）：
> *"**All internal indices combine within-cluster cohesion and between-cluster separation into one number, but use different measures; Rand index is an external index comparing two partitions.**"*
> （**所有内部指标都把簇内紧凑度和簇间分离度合成一个数，只是度量方式不同；Rand 指数是比较两种划分的外部指标。**）

> **Silhouette:** **aᵢ = mean distance to the other samples in the same cluster (cohesion)**; **bᵢ = minimum distance to the samples of other clusters (separation)** (course definition; textbooks use the smallest mean distance to another cluster); **sᵢ = (bᵢ − aᵢ)/max(aᵢ, bᵢ) ∈ [−1, 1]**; **≈ 1 well clustered, ≈ 0 on the boundary, < 0 probably in the wrong cluster**; **average over all samples**. Uses: **evaluate a clustering, choose K, compare algorithms, find badly assigned samples**. Limitation: **O(n²) distances, favours convex clusters**.
>
> 【中文】**Silhouette**：**aᵢ = 到同簇其他样本的平均距离（紧凑度）**；**bᵢ = 到其他簇样本的最小距离（分离度）**（课件定义；教科书用"到最近邻簇的平均距离"）；**sᵢ = (bᵢ − aᵢ)/max(aᵢ, bᵢ) ∈ [−1, 1]**；**≈ 1 分得好，≈ 0 在边界上，< 0 很可能分错了簇**；**对所有样本取平均**。用途：**评价聚类结果、选 K、比较算法、找出分错的样本**。局限：**要算 O(n²) 个距离，偏好凸形簇**。

> **Dunn:** **DI = min inter-cluster distance (closest pair from different clusters) / max intra-cluster diameter (farthest pair within a cluster)**; **measures the worst case**; **larger is better**; **sensitive to noise since it uses extreme values**.
>
> 【中文】**Dunn**：**DI = 最小簇间距离（不同簇间最近的一对）/ 最大簇内直径（同簇内最远的一对）**；**衡量最坏情况**；**越大越好**；**只用极值，对噪声敏感**。

**H7 ★★ Advantages/disadvantages of hierarchical clustering; compare single and complete linkage.**
（层次聚类的优缺点；比较 single 和 complete linkage。）

> Advantages: **no need to specify K; dendrogram shows nested structure; any distance measure**. Disadvantages: **O(n²) memory and at least O(n²) time, not suitable for large data; greedy merges cannot be undone; sensitive to noise**. **Single linkage (minimum distance) can find elongated/non-convex clusters but suffers from the chaining effect** and noise sensitivity; **complete linkage (maximum distance) gives compact, similar-diameter clusters but is sensitive to outliers** and may split large clusters; **average and centroid linkage are compromises**.

> 【中文】优点：**不需要指定 K；树状图展示嵌套结构；可用任意距离度量**。缺点：**需要 O(n²) 内存和至少 O(n²) 时间，不适合大数据；贪心合并不能撤回；对噪声敏感**。**Single linkage（最小距离）能找细长 / 非凸的簇，但有链式效应**，对噪声敏感；**complete linkage（最大距离）得到紧凑、直径相近的簇，但对离群点敏感**，还可能把大簇拆开；**average 和 centroid linkage 是两者的折中**。

---

### 4.5 论述题临场清单（考前 10 分钟看）

- [ ] 题目动词圈出来了吗（justify / rationale / how does it guide / state assumptions）？
- [ ] 公式写出来后，每一项都解释了吗？
- [ ] 要点数 ≥ 分值 ÷ 2 吗？
- [ ] 有没有一句话的优缺点或适用条件？
- [ ] 结合了题目给的图 / 数据吗？
- [ ] 计算题最后写了"判为哪一类 + 置信度高低 + 理由"吗？

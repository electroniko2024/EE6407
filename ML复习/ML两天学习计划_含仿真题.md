# EE6407 ML 两天冲刺计划（含仿真题）

> 配套讲义：[ML复习讲义_重点与论述题.md](ML复习讲义_重点与论述题.md)（下文"讲义 §x"指它的章节）
> 真题在 `历年期末卷/`（扫描版），答案在 `历年期末卷/参考答案/`。
> **规则：2025-26 S1 和 2025-26 S2 两份卷子在 Day 2 下午之前一眼都不要看**，留作最后的真题模考。前四份卷子拆开穿插在学习过程中做。

## 总体安排

| | 上午 | 下午 | 晚上 |
|---|---|---|---|
| **Day 1 监督学习** | LDA → SVM（真题穿插） | 仿真 M1、M2 → Bayes/NB/树 → 仿真 M3 → 真题限时 | 评估 / 特征选择 / 回归论述 → 仿真 M5（纯论述） |
| **Day 2 聚类 + 模考** | 层次聚类、K-means 计算 → 聚类论述 → 仿真 M4 | **真题模考 25-26 S1 Q2–Q4** | **真题模考 25-26 S2 Q2–Q4** → 错题复盘 |

计时标准：每道 25 分大题按 **35～40 分钟**练（4 题的卷子 GA 往往最费时，ML 要给 GA 省时间）。

---

## Day 1：监督学习 + 论述

| 时间 | 任务 | 做什么题 | 产出 |
|---|---|---|---|
| 09:00–09:30 | 读讲义 §0–§2：卷型、老师强调表、考点地图 | — | 心里有张地图 |
| 09:30–10:30 | **LDA**：讲义 §3.1 + 论述 A1–A5 | 真题 **24-25 S2 Q3(b)(c)**（LDA 设计 + 距离置信度，15 分，25 min） | 对答案，错了就照讲义 §3.1 的"坑"逐条查 |
| 10:30–11:00 | LDA 论述闭卷默写 | 不看讲义写 **A1**（12 分题，限 12 min），再写 **A3**（7 分，7 min），对照讲义打分 | 记下漏掉的要点 |
| 11:00–11:15 | 休息 | | |
| 11:15–12:45 | **SVM**：讲义 §3.2 + 论述 B1–B5 + 示范 §4.3 | 真题 **23-24 S2 Q2 整题**（可分原问题 + 求 w,b + LDA vs SVM，25 分，**限时 40 min**） | 2(c) 对照讲义 A4 的表格查漏 |
| 12:45–13:45 | 午饭 | | |
| 13:45–14:30 | **仿真 M1**（LDA + Gaussian NB + 高维小样本，25 分，限时 38 min） | 见下文 | 对答案 |
| 14:30–15:00 | **仿真 M2**（软间隔 SVM 求 b，25 分，限时 30 min） | 见下文 | 重点检查有没有用错 α = C 的点 |
| 15:00–16:15 | **Bayes / NB / 决策树**：讲义 §3.3、§3.4 + 论述 C1–C5、D1–D3 | 真题 **24-25 S1 Q2 整题**（Gaussian Bayes + Gaussian NB，25 分，限时 40 min）；真题 **24-25 S2 Q3(a)**（Bernoulli NB，10 分，12 min） | 注意 scatter / covariance 的区别 |
| 16:15–16:45 | **仿真 M3**（熵 / Gini 选根节点 + Multinomial NB + 平滑论述） | 见下文 | 熵还没考过，这道必做 |
| 16:45–17:00 | 休息 | | |
| 17:00–18:15 | 综合限时 | 真题 **24-25 S1 Q3 整题**（LDA + SVM，限时 38 min）；真题 **24-25 S2 Q2 整题**（软间隔原问题 + 求 w,b，限时 38 min） | 两题都对完答案 |
| 18:15–19:15 | 晚饭 | | |
| 19:15–20:15 | **评估 / 特征选择 / 回归 / 数据准备**：读讲义论述 D4–D5、E1–E3、F1–F8、G1–G4 | 真题 **23-24 S2 Q3 整题**（混淆矩阵 + peaking + 三类特征选择，限时 38 min） | 论述按讲义要点逐条给自己打分 |
| 20:15–20:30 | 数据准备只列提纲 | 真题 **23-24 S1 Q2**：四小问每问只写要点骨架，共 10 min | 近年没考，不在这上面花太多时间 |
| 20:30–21:30 | **仿真 M5**（纯论述 6 题，闭卷，限时 45 min） | 见下文 | 对照讲义打分，把拿分少的题目编号记下来 |
| 21:30–22:00 | **默写公式页**：一张纸写出 LDA、SVM（w, b, 间隔）、三种 NB、熵/Gini、混淆矩阵全部公式 | — | 明早第一件事就是核对它 |

---

## Day 2：聚类 + 真题模考

| 时间 | 任务 | 做什么题 | 产出 |
|---|---|---|---|
| 09:00–09:20 | 核对昨晚默写的公式页，错的抄三遍 | — | |
| 09:20–10:45 | **聚类计算**：讲义 §3.5–§3.7 | 真题 **23-24 S2 Q4(b)**（single linkage，4 维，13 分，20 min）；真题 **24-25 S1 Q4 整题**（single + complete + Silhouette，限时 38 min） | 每步写距离矩阵和合并高度 |
| 10:45–11:00 | 休息 | | |
| 11:00–12:15 | **聚类论述**：讲义 H1–H7 | 真题 **24-25 S2 Q4 整题**（K-means 理论 + 选 K + 不规则形状 + single linkage，限时 38 min）；真题 **23-24 S1 Q4 整题**（按图选算法 + 5 个指标，限时 35 min） | 论述按讲义要点打分 |
| 12:15–13:00 | **仿真 M4**（average linkage + K-means + Silhouette + DBSCAN，限时 38 min） | 见下文 | average linkage 和 DBSCAN 都没考过，这道必做 |
| 13:00–14:00 | 午饭 | | |
| 14:00–14:30 | 看讲义 §1（老师强调表）、§4.5（临场清单），过一遍 Day 1 的错题记录 | — | |
| 14:30–16:25 | **真题模考 ①：2025-26 S1 的 Q2、Q3、Q4**，闭卷，计时 **115 min** | 卷子在 `历年期末卷/EE6407 2025-2026 Semester 1.pdf` | 不查任何资料，计算器可用 |
| 16:25–17:00 | 对照 `参考答案/EE6407 2025-2026 Semester 1 参考答案.pdf` 打分 | | 算出 ML 75 分里拿了多少，论述单独统计 |
| 17:00–17:15 | 休息 | | |
| 17:15–19:10 | **真题模考 ②：2025-26 S2 的 Q2、Q3、Q4**，闭卷，计时 **115 min** | `历年期末卷/EE6407 2025-2026 Semester 2.pdf` | |
| 19:10–19:45 | 对答案打分 | `参考答案/EE6407 2025-2026 Semester 2 参考答案.pdf` | |
| 19:45–20:30 | 晚饭 | | |
| 20:30–21:40 | **错题复盘**：计算错的从头重算一遍；论述丢分的按 D-E-E-C 模板重写一遍 | — | |
| 21:40–22:10 | 最后一遍：讲义 §4.4 里所有 ★★★ 题只背要点骨架（每题 3～5 个关键词） | — | 早点睡 |

**如果时间不够**，按这个顺序砍：数据准备（G）→ 回归 D5 → 仿真 M5 的后两题 → 23-24 S1 Q4。**不能砍**：LDA、SVM 求 b、层次聚类、两套真题模考。

**如果还有富余时间**：`1.Exam/` 文件夹里是更早几年（20-S2 ～ 24-S1）的手写解答，可以挑 ROC、交叉验证、DBSCAN 相关的题看；`6407历年考试题.docx` 里列了 21-22 S2、22-23 S2 的题目概要（评估方法与指标、k 折交叉验证、filter/wrapper 框图、DBSCAN 步骤），都可以拿来当论述练习。

---

## 仿真题

> 按老师的出题习惯仿写：计算题后面跟置信度 / 解释，论述小问 5～8 分。数值都用程序算过。**先做完再翻到文末看答案。**

### 仿真 M1（25 分）LDA + Gaussian NB

A two-class problem has 50 training samples in each class. The sample mean vectors and **scatter matrices** are:

m₁ = [2.0, 1.2]ᵀ, S₁ = [[40, 8], [8, 30]];  m₂ = [0.5, 3.0]ᵀ, S₂ = [[36, −4], [−4, 45]]

(a) Design a Fisher linear discriminant classifier. (8 marks)
(b) Classify the test sample x = [1.4, 2.0]ᵀ. Compute its distance to the decision boundary and state your confidence. (5 marks)
(c) Design a Gaussian naïve Bayes classifier using the same data (equal priors). State all assumptions and classify x. (8 marks)
(d) Suppose each sample instead had 200 features while the number of samples stayed the same. What problem would occur in (a)? Suggest one remedy. (4 marks)

### 仿真 M2（25 分）软间隔 SVM

A linear SVM is trained on non-separable data with **C = 2**. The support vectors and their Lagrange multipliers are:

| SV | x | d | α |
|---|---|---|---|
| 1 | [2.5, 1.0]ᵀ | +1 | 2.000 |
| 2 | [1.0, 2.5]ᵀ | +1 | 1.825 |
| 3 | [0.5, 1.5]ᵀ | −1 | 0.950 |
| 4 | [2.0, 0.5]ᵀ | −1 | 0.875 |
| 5 | [1.8, 1.6]ᵀ | −1 | 2.000 |

(a) Verify that the multipliers satisfy the equality constraint of the dual problem. (2 marks)
(b) Determine w and b. Explain which support vectors you use to compute b and why. (10 marks)
(c) Classify x = [2.2, 1.5]ᵀ, find its distance to the decision boundary and state your confidence. (6 marks)
(d) For SV 1 and SV 5, determine whether each lies inside the margin or is misclassified, and give the corresponding value of the slack variable ξ. (7 marks)

### 仿真 M3（25 分）决策树（熵）+ Multinomial NB

(a) 16 training samples (8 from ω₁, 8 from ω₂) have three binary features. The class counts for each feature value are:

| Feature | value = 1 (ω₁, ω₂) | value = 0 (ω₁, ω₂) |
|---|---|---|
| A | (6, 2) | (2, 6) |
| B | (5, 5) | (3, 3) |
| C | (7, 3) | (1, 5) |

Using **entropy impurity**, determine which feature should be used at the root node. Would the choice change if Gini impurity were used? (10 marks)

(b) Training documents:
- ω₁ (Sports): "ball goal team", "goal goal win", "team win"
- ω₂ (Politics): "vote team law", "law vote win vote"

Using multinomial naïve Bayes with Laplace smoothing, classify the document "goal team vote". (10 marks)

(c) Explain why Laplace smoothing is needed in (b). (5 marks)

### 仿真 M4（25 分）聚类

Five 2-D samples: A = (1, 1), B = (1.5, 2), C = (5, 5), D = (6, 5.4), E = (3, 3.5).

(a) Perform agglomerative hierarchical clustering using **average linkage** (Euclidean distance). Show the distance matrix at each step and sketch the dendrogram. (9 marks)
(b) Apply K-means with K = 2, using A and C as initial centroids. Perform two iterations. (8 marks)
(c) Compute the Silhouette coefficient of sample E for the final K-means clustering and interpret the result. (4 marks)
(d) One-dimensional data: 1.0, 1.6, 2.1, 2.5, 4.0, 6.0, 6.8, 7.5, 8.1, 10.0. Apply DBSCAN with ε = 1.0 and MinPts = 3 (a point counts itself; core point if its ε-neighbourhood contains ≥ MinPts points). Identify core, border and noise points and the resulting clusters. (4 marks)

### 仿真 M5（纯论述，闭卷 45 分钟）

1. Describe the k-fold cross-validation procedure and discuss its rationale. Compare it with the hold-out method and leave-one-out cross-validation. (8 marks)
2. What are the ROC curve and AUC? Why can accuracy be misleading on imbalanced data? (6 marks)
3. Compare ridge regression and lasso regression. If you suspect that only a few of 200 predictors are relevant, which would you use and why? (7 marks)
4. Explain how a random forest is constructed and why it usually outperforms a single decision tree. (7 marks)
5. What is the curse of dimensionality? Describe its two main aspects. (5 marks)
6. In a soft-margin SVM, why can a support vector with α = C not be used to compute the bias b? (5 marks)

---
---

## 仿真题答案（做完再看）

### M1 答案

**(a)**
S_W = S₁ + S₂ = [[76, 4], [4, 75]]，det = 76×75 − 16 = 5684
S_W⁻¹ = (1/5684)·[[75, −4], [−4, 76]] = [[0.013195, −0.000704], [−0.000704, 0.013371]]
m₁ − m₂ = [1.5, −1.8]ᵀ
w = S_W⁻¹(m₁ − m₂) = (1/5684)·[75×1.5 + 4×1.8, −4×1.5 − 76×1.8]ᵀ = [119.7, −142.8]ᵀ/5684 = **[0.021059, −0.025123]ᵀ**
m₁ + m₂ = [2.5, 4.2]ᵀ，w₀ = −½(0.021059×2.5 − 0.025123×4.2) = −½(0.052648 − 0.105517) = **0.026435**
g(x) = 0.021059x₁ − 0.025123x₂ + 0.026435；g > 0 → ω₁，g < 0 → ω₂

**(b)** g(x) = 0.029483 − 0.050246 + 0.026435 = **0.005672 > 0 → ω₁**
‖w‖ = 0.032782，距离 r = 0.005672/0.032782 = **0.173**
参照：类均值 m₁ 到边界的距离 = |g(m₁)|/‖w‖ = 0.038405/0.032782 = 1.172。测试样本的距离只有它的约 15%，样本很靠近决策面，**置信度低**。

**(c)** 假设：① 给定类别后两个特征条件独立（忽略协方差的非对角元）；② 每个特征在每一类内服从一维高斯；③ 方差 = scatter 对角元 / 样本数（**不能直接用 scatter**）；④ 先验相等 0.5。
- ω₁：σ² = (40/50, 30/50) = (0.8, 0.6)；p(1.4|ω₁) = 0.3562，p(2.0|ω₁) = 0.3021，乘积 = 0.1076
- ω₂：σ² = (36/50, 45/50) = (0.72, 0.9)；p(1.4|ω₂) = 0.2679，p(2.0|ω₂) = 0.2413，乘积 = 0.0646
- 0.1076 > 0.0646 → **ω₁**，后验 P(ω₁|x) ≈ 0.625，和 LDA 结论一致。
- 坑：如果忘了除以 50，方差变成 40、30……密度几乎是平的，结果没有意义。

**(d)** d = 200，N = 100：S_W 的秩至多 N − 2 = 98 < 200，**S_W 奇异不可逆**，w 无法计算；即使勉强求伪逆也会严重过拟合。补救：正则化 S_W + λI（λ 用交叉验证选），或先 PCA 降到 98 维以下，或做特征选择。（讲义 A3）

### M2 答案

**(a)** Σαᵢdᵢ = 2 + 1.825 − 0.95 − 0.875 − 2 = **0** ✓

**(b)**
正类贡献：2×[2.5, 1.0] + 1.825×[1.0, 2.5] = [5, 2] + [1.825, 4.5625] = [6.825, 6.5625]
负类贡献：0.95×[0.5, 1.5] + 0.875×[2.0, 0.5] + 2×[1.8, 1.6] = [0.475, 1.425] + [1.75, 0.4375] + [3.6, 3.2] = [5.825, 5.0625]
**w = [6.825 − 5.825, 6.5625 − 5.0625] = [1.0, 1.5]ᵀ**

b 只能用 **0 < α < C = 2** 的支持向量：SV 2、3、4（它们在间隔边界上，ξ = 0，dᵢ(wᵀxᵢ + b) = 1 严格成立）。SV 1、5 的 α = C，可能在间隔内或被错分，不能用。
- SV 2：b = 1 − (1.0 + 3.75) = −3.75
- SV 3：b = −1 − (0.5 + 2.25) = −3.75
- SV 4：b = −1 − (2.0 + 0.75) = −3.75
- 平均 **b = −3.75**。（若误用 SV 1：b = 1 − 4 = −3，错。）

g(x) = x₁ + 1.5x₂ − 3.75

**(c)** g(x) = 2.2 + 2.25 − 3.75 = **0.70 > 0 → 类 +1**
‖w‖ = √3.25 = 1.8028，距离 = 0.70/1.8028 = **0.388**；间隔半宽 1/‖w‖ = 0.555。
|g| = 0.70 < 1，样本落在**间隔内部**，比正类间隔边界离决策面还近 → **置信度低**。

**(d)**
- SV 1：g = 2.5 + 1.5 − 3.75 = 0.25，d·g = 0.25 ∈ (0, 1) → 分类正确但**落在间隔内**，ξ = 1 − 0.25 = **0.75**
- SV 5：g = 1.8 + 2.4 − 3.75 = 0.45，d = −1，d·g = −0.45 < 0 → **被错分**，ξ = 1 + 0.45 = **1.45 > 1**
- 两者都是 α = C，符合"α = C 对应间隔内或错分样本"。

### M3 答案

**(a)** 根节点熵 = 1（8:8）。

| 特征 | 分支熵 | 加权熵 | 信息增益 | 分支 Gini | 加权 Gini |
|---|---|---|---|---|---|
| A | (6,2): 0.8113；(2,6): 0.8113 | 0.8113 | 0.1887 | 0.375；0.375 | 0.3750 |
| B | (5,5): 1；(3,3): 1 | 1.0000 | 0 | 0.5；0.5 | 0.5000 |
| C | (7,3): 0.8813；(1,5): 0.6500 | (10/16)(0.8813) + (6/16)(0.6500) = **0.7946** | **0.2054** | 0.42；0.2778 | **0.3667** |

示例：H(7,3) = −0.7log₂0.7 − 0.3log₂0.3 = 0.3602 + 0.5211 = 0.8813。
**选 C 做根节点**（加权熵最小、信息增益最大）。用 Gini 也选 C（0.3667 < 0.3750 < 0.5），结论不变。B 两个分支的类别比例和父节点一样（1:1），完全没有信息量。

**(b)** 词表 {ball, goal, team, win, vote, law}，d = 6。
ω₁ 词频：ball 1，goal 3，team 2，win 2，总数 N₁ = 8；ω₂：vote 3，team 1，law 2，win 1，总数 N₂ = 7。先验 3/5、2/5。

| 词 | P(·\|ω₁) = (c+1)/(8+6) | P(·\|ω₂) = (c+1)/(7+6) |
|---|---|---|
| goal | 4/14 | 1/13 |
| team | 3/14 | 2/13 |
| vote | 1/14 | 4/13 |

ω₁：0.6 × (4·3·1)/14³ = 0.6 × 12/2744 = **2.62 × 10⁻³**
ω₂：0.4 × (1·2·4)/13³ = 0.4 × 8/2197 = **1.46 × 10⁻³**
→ **判为 ω₁ (Sports)**，归一化后验 ≈ 0.643 : 0.357。

**(c)** "vote"在 ω₁ 中、"goal"在 ω₂ 中出现次数为 0，不平滑的话两类的似然都是 0，无法比较；零概率会一票否决其他所有证据。拉普拉斯平滑分子加 1、分母加 d，保证所有概率为正，计数大时估计值仍接近频率。（讲义 C3）

### M4 答案

**初始距离矩阵**

| | A | B | C | D | E |
|---|---|---|---|---|---|
| A | 0 | 1.1180 | 5.6569 | 6.6603 | 3.2016 |
| B | | 0 | 4.6098 | 5.6400 | 2.1213 |
| C | | | 0 | 1.0770 | 2.5000 |
| D | | | | 0 | 3.5511 |

**(a) average linkage**
1. 最小 1.0770，合并 **{C,D}**。更新：d(CD,A) = (5.6569+6.6603)/2 = 6.1586；d(CD,B) = (4.6098+5.6400)/2 = 5.1249；d(CD,E) = (2.5+3.5511)/2 = 3.0256
2. 最小 1.1180，合并 **{A,B}**。更新：d(AB,E) = (3.2016+2.1213)/2 = 2.6614；d(AB,CD) = (5.6569+6.6603+4.6098+5.6400)/4 = 5.6418
3. 最小 2.6614，**E 并入 {A,B}**。更新：d(ABE,CD) = 6 个点对的平均 = 28.6181/6 = 4.7697
4. 在 **4.7697** 处合并全部。

```
4.7697 ─      ┌──────────────┴──────────────┐
2.6614 ─  ┌───┴─────┐                       │
1.1180 ─ ┌┴┐        │                       │
1.0770 ─ │ │        │                  ┌────┴────┐
         A B        E                  C         D
```
注意第 3 步的 4.7697 必须用原始 6 个点对平均，不能把 5.6418 和 3.0256 简单平均。

**(b) K-means**，初始中心 c₁ = A(1,1)，c₂ = C(5,5)

第 1 轮：

| 样本 | 到 c₁ | 到 c₂ | 归属 |
|---|---|---|---|
| A | 0 | 5.6569 | 1 |
| B | 1.1180 | 4.6098 | 1 |
| C | 5.6569 | 0 | 2 |
| D | 6.6603 | 1.0770 | 2 |
| E | 3.2016 | 2.5000 | 2 |

更新：c₁ = (A+B)/2 = (1.25, 1.5)；c₂ = (C+D+E)/3 = (4.6667, 4.6333)

第 2 轮：

| 样本 | 到 c₁ | 到 c₂ | 归属 |
|---|---|---|---|
| A | 0.5590 | 5.1619 | 1 |
| B | 0.5590 | 4.1185 | 1 |
| C | 5.1296 | 0.4955 | 2 |
| D | 6.1459 | 1.5380 | 2 |
| E | 2.6575 | 2.0155 | 2 |

分配不变，中心不变，**已收敛**：{A,B}、{C,D,E}。

**(c)** E 在簇 {C,D,E}：a = (2.5 + 3.5511)/2 = 3.0256；b = 到 {A,B} 的平均距离 = (3.2016 + 2.1213)/2 = 2.6614
s(E) = (2.6614 − 3.0256)/3.0256 = **−0.120**
解释：s < 0，说明按平均距离 E 离另一个簇更近，它是两簇之间的边界点，很可能被"分错"。K-means 看的是到**质心**的距离（E 离 c₂ 更近），Silhouette 看的是到簇内**各点的平均距离**，而 (a) 的层次聚类把 E 和 A、B 放在一起——不同准则对边界点给出不同答案。

**(d)** 各点 ε 邻域内点数（含自身）：1.0:2，1.6:4，2.1:3，2.5:3，4.0:1，6.0:2，6.8:3，7.5:3，8.1:2，10.0:1
- **核心点**：1.6，2.1，2.5，6.8，7.5
- **边界点**：1.0（在 1.6 的邻域内），6.0（在 6.8 的邻域内），8.1（在 7.5 的邻域内）
- **噪声**：4.0，10.0
- 簇：{1.0, 1.6, 2.1, 2.5} 和 {6.0, 6.8, 7.5, 8.1}

### M5 答案

逐题对照讲义：1 → E1 + E2；2 → E3；3 → D4（选 **Lasso**：L1 会把无关变量的系数压到 0，自动做变量选择，得到稀疏、可解释的模型）；4 → D3；5 → F2；6 → B4。
打分方法：每个讲义要点算 1 分，没写"为什么"的要点只算半分。

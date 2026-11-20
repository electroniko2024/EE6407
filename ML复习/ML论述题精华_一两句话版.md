# EE6407 ML 论述题精华 · 一两句话版（English + 中文）

> **用法**：每题只留 **1～2 句核心要点**，英文是可以直接写在答卷上的"骨架句"，中文是对应译文。**加粗的部分就是核心采分点**——考场上至少要把这些词/短语写出来。先写这一两句保底，再按 D-E-E-C 模板（定义 → 解释 → 评价 → 结论）补充细节。完整展开版见 [ML复习讲义_重点与论述题.md](ML复习讲义_重点与论述题.md) §4 和各专题的"论述题范文"。
>
> **范围（共 41 题）**：
> - **遗传算法（GA）论述题已按要求删除**。
> - **历年真题的论述题（除 GA 外）全部保留**，共 28 题。
> - **自拟 / 讲义 / 专题 / 模拟卷里的论述题留 13 题**：之前保留的 8 题（LDA 的 w₀ 假设、分类置信度、为什么只用 0<α<C 求 b、Laplace 平滑、随机森林、Ridge vs Lasso、准确率为何不够 + ROC/AUC、层次聚类 linkage 比较），再补回最重要的 5 题（Bayes 分类器与 LDA 的关系、为什么最大化间隔、SVM 为什么用对偶问题、决策树剪枝、K-means vs DBSCAN）。其余自拟题的答案仍在讲义 §4.4 和专题里。
>
> **约定**：课件版公式（Gini = ½(1−ΣP²)；Silhouette 的 b 取到其他簇样本的最小距离；LDA 的 w₀ = −½wᵀ(m₁+m₂)）。数学符号用普通 Unicode，不用 LaTeX。
>
> **来源缩写**：真题 = 历年卷（如"25-26 S1 4(a)"表示 2025-26 第一学期 Q4(a)；"旧卷"表示 21-22、22-23 等更早的卷）；讲义 = 复习讲义题库编号；专题 = ML专题0x 的 E 编号；模拟 = 模拟卷 / 仿真题。

---

## 目录

| 组 | 主题 | 题号 |
|---|---|---|
| A | LDA | 1–7 |
| B | SVM | 8–13 |
| C | Bayes / 朴素贝叶斯 | 14–16 |
| D | 决策树 / 随机森林 / 回归 | 17–20 |
| E | 分类器评估与特征选择 | 21–27 |
| F | 数据准备 | 28–31 |
| G | 聚类（Q4） | 32–41 |

---

## A. LDA（几乎每年的 Q2，论述每次换角度）

### 1. Formulate Fisher LDA as an optimisation problem and explain its rationale（把 Fisher LDA 表述成优化问题并解释原理）
*来源：真题 25-26 S1 2(a)；讲义 A1；专题 01 E1*

**EN:** LDA finds the projection y = wᵀx that **maximises J(w) = (m̃₁−m̃₂)²/(s̃₁²+s̃₂²) = wᵀS_B w / wᵀS_W w**, i.e. a **large separation of the projected means relative to the projected within-class scatter**, which gives **w = S_W⁻¹(m₁−m₂)**. The **ratio** is needed because a large mean difference alone does not help if the within-class spread is also large.

**中文：** LDA 寻找投影 y = wᵀx，使 **J(w) = (m̃₁−m̃₂)²/(s̃₁²+s̃₂²) = wᵀS_B w / wᵀS_W w 最大**，即**投影后均值相距远、同时类内散度小**，解为 **w = S_W⁻¹(m₁−m₂)**。必须用**比值**，因为只拉大均值差而类内方差也大时，两类仍会重叠。

### 2. How is the threshold w₀ determined, what assumption is behind it, and what if it fails?（w₀ 怎么定、隐含什么假设、不成立怎么办）
*来源：专题 01 E3；模拟卷 2 Q2(c)；预测文档（LDA 论述押题，约 65%）*

**EN:** **w₀ = −½wᵀ(m₁+m₂)** puts the **midpoint of the two projected means** on the boundary, which **assumes similar class spreads (and implicitly equal priors)**. If this fails, **use it as a starting value and try nearby thresholds, keeping the one with the best training/validation accuracy**, or add ln[P(ω₁)/P(ω₂)] and shift the threshold towards the more compact class.

**中文：** **w₀ = −½wᵀ(m₁+m₂)** 使**两类投影均值的中点**落在边界上，隐含**假设两类散度相近（以及先验相等）**。不成立时**把它当起点，试几个附近的值，取训练/验证精度最高的**，或加上 ln[P(ω₁)/P(ω₂)] 并把阈值移向更紧凑的一类。

### 3. Under what conditions does LDA perform best?（LDA 在什么条件下最好）
*来源：真题 25-26 S2 2(a)；讲义 A2；专题 01 E4*

**EN:** When **each class is unimodal Gaussian with a common covariance matrix** (then LDA coincides with the Bayes classifier), the classes **differ mainly in their means and are nearly linearly separable**, **N ≫ d**, and there are **no severe outliers**.

**中文：** **每类是单峰高斯且协方差相同**（此时 LDA 即 Bayes 分类器），两类**主要在均值上不同且近似线性可分**，**样本数远大于维数（N ≫ d）**，且**没有严重离群点**。

### 4. Why does LDA fail for high-dimensional data with few samples, and how can it be fixed?（高维小样本下 LDA 为何失效、如何补救）
*来源：真题 25-26 S2 2(c)；讲义 A3；专题 01 E5；仿真 M1(d)*

**EN:** **S_W has rank at most N−2, so it is singular when d > N−2** (and ill-conditioned otherwise), which makes w extremely noise-sensitive and overfitted. Fix it by **regularisation (S_W + λI, λ by cross-validation)**, by **reducing the dimension first (PCA or feature selection)**, or by **collecting more samples**.

**中文：** **S_W 的秩至多为 N−2，d > N−2 时奇异不可逆**（略小时也病态），w 对噪声极其敏感、过拟合。补救：**正则化（S_W + λI，λ 用交叉验证选）**、**先降维（PCA 或特征选择）**、**增加样本**。

### 5. Main differences between LDA and linear SVM（LDA 与线性 SVM 的主要区别）
*来源：真题 23-24 S2 2(c)；讲义 A4；专题 01 E7、专题 02 E6*

**EN:** LDA **maximises the Fisher ratio using all samples through means and scatter matrices** (closed form, implicit Gaussian equal-covariance assumption), whereas SVM **maximises the margin 2/‖w‖ and is determined only by the support vectors** (convex QP, no distribution assumption, with soft margin and kernels for non-separable data).

**中文：** LDA **最大化 Fisher 比，通过均值和散度矩阵用到全部样本**（闭式解，隐含高斯且协方差相同的假设）；SVM **最大化间隔 2/‖w‖，只由支持向量决定**（凸二次规划，无分布假设，不可分时有软间隔和核）。

### 6. How to judge the confidence of a classification result（如何判断分类结果的置信度）
*来源：每年计算题附带；讲义 A5；专题 01 E6；专题 02 E7*

**EN:** The **distance to the decision boundary r = |g(x)|/‖w‖** is the confidence: far from the boundary means high confidence, close means the label may flip. For **LDA compare r with the distance of the class mean to the boundary**, for **SVM compare |g(x)| with 1** (r with the half-margin 1/‖w‖).

**中文：** **到决策边界的距离 r = |g(x)|/‖w‖** 就是置信度：离得远置信度高，离得近类别可能翻转。**LDA 把 r 与类均值到边界的距离比较**，**SVM 把 |g(x)| 与 1 比较**（即 r 与半间隔 1/‖w‖ 比较）。

### 7. Relationship between the Bayes classifier and LDA（Bayes 分类器与 LDA 的关系）
*来源：讲义 C4；专题 01 E8、专题 03 E9*

**EN:** If **both classes are Gaussian with the same covariance, the quadratic terms of the Bayes discriminant cancel** and the boundary is linear with normal vector Σ⁻¹(μ₁−μ₂), the **same direction as Fisher's w, so LDA is Bayes-optimal**. The only difference is the threshold (Bayes adds ln[P(ω₁)/P(ω₂)]); with **unequal covariances the Bayes boundary is quadratic** and LDA is no longer optimal.

**中文：** 两类为**高斯且协方差相同**时，**Bayes 判别函数的二次项抵消**，边界是法向量为 Σ⁻¹(μ₁−μ₂) 的直线，**与 Fisher 的 w 同方向，所以 LDA 是 Bayes 最优的**。区别只在阈值（Bayes 要加 ln[P(ω₁)/P(ω₂)]）；**协方差不同时 Bayes 边界是二次的**，LDA 不再最优。

---

## B. SVM

### 8. Formulate the primal problem of the linear SVM for separable data and justify it（可分情形的原问题与理由）
*来源：真题 23-24 S2 2(a)；讲义 B1；专题 02 E1*

**EN:** **Minimise ½wᵀw subject to dᵢ(wᵀxᵢ+b) ≥ 1 for all i**: the canonical scaling makes the constraints demand that every sample is classified correctly, and **minimising ‖w‖ maximises the margin 2/‖w‖**. The maximum-margin hyperplane is the most robust and generalises best, and the problem is a **convex QP with a unique global optimum**.

**中文：** **最小化 ½wᵀw，约束 dᵢ(wᵀxᵢ+b) ≥ 1（所有 i）**：规范化尺度使约束要求每个样本都分对，**最小化 ‖w‖ 就是最大化间隔 2/‖w‖**。最大间隔超平面最稳健、泛化最好，且问题是**凸二次规划，有唯一全局最优**。

### 9. Why does the SVM maximise the margin?（为什么 SVM 要最大化间隔）
*来源：专题 02 E5*

**EN:** A classifier should be both correct and confident: among the infinitely many separating hyperplanes, the **maximum-margin one is farthest from both classes**, so **small perturbations of test samples are unlikely to push them across the boundary**, which gives the **best generalisation**.

**中文：** 分类器既要分对又要有信心：在无数个能分开两类的超平面中，**最大间隔的那个离两类都最远**，所以**测试样本稍有扰动也不容易越界**，**泛化最好**。

### 10. Formulate the primal problem for non-separable data and explain each term（不可分情形的原问题）
*来源：真题 24-25 S2 2(a)；讲义 B2；专题 02 E2；模拟卷 1 Q2(a)*

**EN:** **Minimise ½wᵀw + C·Σξᵢ subject to dᵢ(wᵀxᵢ+b) ≥ 1−ξᵢ and ξᵢ ≥ 0**: the first term **maximises the margin**, the second **penalises margin violations and misclassifications**, and ξᵢ ≥ 0 keeps the problem a convex QP.

**中文：** **最小化 ½wᵀw + C·Σξᵢ，约束 dᵢ(wᵀxᵢ+b) ≥ 1−ξᵢ 且 ξᵢ ≥ 0**：第一项**最大化间隔**，第二项**惩罚越过间隔和错分的样本**，ξᵢ ≥ 0 保证问题仍是凸二次规划。

### 11. Roles of the slack variables ξ and the parameter C（ξ 与 C 的作用）
*来源：真题 25-26 S2 3(a)；讲义 B3；专题 02 E3*

**EN:** **ξᵢ measures how far sample i violates the margin** (ξ = 0 on or outside the margin, 0 < ξ ≤ 1 inside the margin but correctly classified, ξ > 1 misclassified), and **C trades margin width against training errors**: **large C gives a narrow margin and few violations (overfitting risk)**, **small C gives a wide margin and more violations (underfitting risk)**.

**中文：** **ξᵢ 衡量样本 i 违反间隔的程度**（ξ = 0 在间隔线上或外侧；0 < ξ ≤ 1 在间隔内但分对；ξ > 1 被错分）；**C 在间隔宽度与训练误差之间权衡**：**C 大则间隔窄、违反少（易过拟合）**，**C 小则间隔宽、违反多（易欠拟合）**。

### 12. What are support vectors? Why does SVM depend only on them? Why can only those with 0 < α < C be used to compute b?（支持向量；为什么只用 0<α<C 的求 b）
*来源：讲义 B4、B5；专题 02 E4；仿真 M5 第 6 题；模拟卷 1 Q2(b)；老师 week9 强调*

**EN:** **Support vectors are the samples with α > 0 and w = Σαᵢdᵢxᵢ, so all other samples have no influence** on the classifier. **Only vectors with 0 < α < C have ξ = 0 and lie exactly on the margin**, so dᵢ(wᵀxᵢ+b) = 1 holds and gives **b = dᵢ − wᵀxᵢ**; vectors with **α = C may be inside the margin or misclassified, so the equality fails** and would give a wrong b (average b over the valid ones).

**中文：** **支持向量是 α > 0 的样本，w = Σαᵢdᵢxᵢ，其余样本对分类器没有影响**。**只有 0 < α < C 的样本 ξ = 0、恰好落在间隔线上**，dᵢ(wᵀxᵢ+b) = 1 成立，才能用 **b = dᵢ − wᵀxᵢ** 求 b；**α = C 的样本可能在间隔内或被错分，等式不成立**，会算出错误的 b（对有效的取平均）。

### 13. Why is the dual problem used?（为什么用对偶问题）
*来源：专题 02 E8*

**EN:** The dual has **only the multipliers α as unknowns with simple constraints (Σαᵢdᵢ = 0, 0 ≤ αᵢ ≤ C)** and is a standard QP (solved by SMO for large data); its solution is **sparse and identifies the support vectors**, and the **data appear only through inner products, which allows the kernel trick**.

**中文：** 对偶问题**只有拉格朗日乘子 α 一种未知数，约束简单（Σαᵢdᵢ = 0，0 ≤ αᵢ ≤ C）**，是标准二次规划（大数据用 SMO）；解是**稀疏的、直接给出支持向量**，且**数据只以内积形式出现，可以用核技巧**。

---

## C. Bayes / 朴素贝叶斯

### 14. Prior, likelihood, posterior and joint probability, and the Bayes decision rule（四种概率与 Bayes 决策规则）
*来源：专题 03 E1；真题 24-25 S1 2(a)*

**EN:** The **prior P(ω)** is the belief before seeing x, the **class-conditional density p(x|ω)** is how x is distributed in class ω, the **joint is p(x|ω)P(ω)**, and the **posterior P(ω|x) is the joint divided by p(x)**. **Bayes rule: assign x to the class with the largest posterior, i.e. the largest p(x|ωᵢ)P(ωᵢ)**, which minimises the error probability.

**中文：** **先验 P(ω)** 是看到 x 之前的信念，**类条件密度 p(x|ω)** 是 x 在该类中的分布，**联合概率是 p(x|ω)P(ω)**，**后验 P(ω|x) 是联合概率除以 p(x)**。**Bayes 规则：把 x 判给后验最大（即 p(x|ωᵢ)P(ωᵢ) 最大）的类**，这使错误概率最小。

### 15. What is the naive assumption, and why does Naive Bayes still work when it is violated?（朴素假设；为什么不成立时仍好用）
*来源：真题 24-25 S1 2(b)；讲义 C1；专题 03 E3；模拟卷 1 Q3(b)*

**EN:** **Given the class, features are conditionally independent, p(x|ω) = Πⱼ p(xⱼ|ω)** (a diagonal covariance for Gaussian features), which **cuts the parameters per class from d(d+1)/2 covariance entries to d variances** and so greatly reduces estimation variance with few samples. It still works because **classification only needs the correct ranking of the posteriors**, but strongly correlated features get double-counted.

**中文：** **给定类别，各特征条件独立，p(x|ω) = Πⱼ p(xⱼ|ω)**（高斯特征即协方差矩阵为对角阵），**把每类需估计的协方差元素从 d(d+1)/2 个降到 d 个方差**，小样本时大幅降低估计方差。假设不成立仍然好用，因为**分类只要求后验的排序正确**；但强相关特征会被重复计数。

### 16. Why is Laplace smoothing needed? What happens without it?（为什么需要 Laplace 平滑）
*来源：讲义 C3；专题 03 E5；仿真 M3(c)；模拟卷 2 Q3(a)(ii)*

**EN:** A word never seen with a class gets **probability 0, which wipes out the whole product of likelihoods** whatever the other evidence, so no decision can be made; **adding 1 to every count and d to the denominator makes all probabilities positive** while staying close to the frequencies for large counts. A result built mostly on smoothed zero counts has low confidence.

**中文：** 某词从未和某类一起出现，其概率为 0，**连乘后无论其他证据多强结果都是 0**，无法判决；**每个计数加 1、分母加 d，使所有概率为正**，计数大时仍接近原频率。结果主要由平滑后的零计数决定时，置信度低。

---

## D. 决策树 / 随机森林 / 回归

### 17. How is a classification tree built and how is a split chosen?（怎么建树、怎么选划分）
*来源：讲义 D1；专题 04 E2；真题 25-26 S1 3(c)*

**EN:** At each node try every feature and candidate threshold and **choose the split with the largest impurity drop (lowest weighted child impurity, by entropy or Gini)**, **split recursively until a stopping criterion is met**, and **label each leaf with the majority class**.

**中文：** 每个节点上尝试所有特征和候选阈值，**选不纯度下降最大（子节点加权不纯度最小，用熵或 Gini）的划分**，**递归划分直到满足停止准则**，**每个叶子按多数类标注**。

### 18. When to stop splitting and why prune?（何时停止划分、为什么剪枝）
*来源：讲义 D2；专题 04 E4*

**EN:** A **fully grown tree memorises the training data (overfitting)** while stopping too early underfits; **stop when the impurity drop is small, nodes are small, the maximum depth is reached or validation error stops falling**, or **grow the full tree and then prune leaves whose removal barely increases the error**.

**中文：** **长满的树会记住训练数据（过拟合）**，停得太早又会欠拟合；可在**不纯度下降很小、节点样本太少、达到最大深度或验证误差不再下降时停止**，或**先长满再剪掉去掉后误差只略增的叶子**。

### 19. Explain how a random forest is built and used, and why it usually outperforms a single tree（随机森林的构造，以及为什么通常优于单棵树）
*来源：讲义 D3；专题 04 E5、E6；模拟卷 2 Q3(c)；仿真 M5 第 4 题（预测约 40%）*

**EN:** A random forest is a **bagging ensemble of trees, each trained on a bootstrap sample with a random subset of features, combined by majority vote**; **averaging many diverse, low-bias but high-variance trees reduces variance**, so it is more accurate and robust than a single overfitting tree, at the cost of training time and interpretability.

**中文：** 随机森林是 **bagging 的树集成：每棵树用自助样本和随机特征子集训练，最后多数投票**；**把许多多样、低偏差、高方差的树平均，降低方差**，因此比单棵易过拟合的树更准确、更稳健，代价是训练更慢、可解释性变差。

### 20. Compare ridge and lasso regression（岭回归与 Lasso 的比较）
*来源：讲义 D4；专题 07 E6；仿真 M5 第 3 题*

**EN:** Both **add a penalty to the least-squares loss to reduce overfitting** and handle collinearity: **ridge (L2, λ‖β‖²) shrinks all coefficients towards zero** and has the closed form (ΦᵀΦ+λI)⁻¹Φᵀy, whereas **lasso (L1, λ‖β‖₁) sets some coefficients exactly to zero, so it selects features** and is the better choice when only a few of many predictors matter (λ is chosen by cross-validation).

**中文：** 两者都**在最小二乘损失上加惩罚来减轻过拟合**、处理共线性：**岭回归（L2，λ‖β‖²）把所有系数向 0 收缩**，有闭式解 (ΦᵀΦ+λI)⁻¹Φᵀy；**Lasso（L1，λ‖β‖₁）会把一些系数压成恰好为 0，具有特征选择作用**，所以在大量预测变量中只有少数有用时更合适（λ 用交叉验证选）。

---

## E. 分类器评估与特征选择

### 21. Discuss the procedures and metrics for evaluating a classifier（评估分类器的方法与指标）
*来源：讲义 E1；专题 05 E1；真题 21-22 S2（旧卷）*

**EN:** Procedures: **hold-out, repeated hold-out, k-fold cross-validation, leave-one-out and repeated k-fold** (stratified sampling keeps class ratios); metrics: **accuracy/error rate, sensitivity and specificity, precision, recall, F-score and ROC/AUC**, chosen according to the application (e.g. recall matters most in cancer diagnosis, and accuracy misleads on imbalanced data).

**中文：** 方法：**留出法、重复留出法、k 折交叉验证、留一法、重复 k 折**（分层抽样保持类别比例）；指标：**准确率/错误率、灵敏度和特异度、精确率、召回率、F 值、ROC/AUC**，按应用选择（如癌症诊断更看重召回率，类别不平衡时准确率会误导）。

### 22. Describe k-fold cross-validation and its rationale; compare with hold-out and leave-one-out（k 折交叉验证及其与留出法、留一法的比较）
*来源：讲义 E2；专题 05 E2；仿真 M5 第 1 题；真题 22-23 S2（旧卷）*

**EN:** **Split the data into k folds, use each fold once as the test set and the other k−1 for training, and average the k results**, so **every sample is used for both training and testing** and the estimate has lower variance than a single hold-out split. **Leave-one-out (k = N) is nearly unbiased but expensive**, and **hold-out is cheap but depends on one random split and wastes data**.

**中文：** **把数据分成 k 折，每折轮流做测试集、其余 k−1 折训练，k 次结果取平均**，**每个样本都既参与训练又参与测试**，估计的方差比一次留出划分小。**留一法（k = N）几乎无偏但开销大**；**留出法便宜，但依赖一次随机划分且浪费数据**。

### 23. Why is accuracy not enough? Explain precision, recall, ROC and AUC（准确率为何不够；精确率、召回率、ROC、AUC）
*来源：讲义 E3；专题 05 E8；仿真 M5 第 2 题；模拟卷 1 Q2(c)*

**EN:** With **imbalanced classes a classifier predicting everything as the majority class has high accuracy but zero recall**; **precision = TP/(TP+FP)** and **recall = TP/(TP+FN)** describe the class of interest, and the **ROC curve (TPR vs FPR over all thresholds) with its threshold-independent AUC (1 = perfect, 0.5 = random)** summarises performance.

**中文：** **类别不平衡时，把一切判成多数类的分类器准确率很高、召回率却为 0**；**精确率 = TP/(TP+FP)**、**召回率 = TP/(TP+FN)** 刻画关注的那一类；**ROC 曲线（各阈值下 TPR 对 FPR）及与阈值无关的 AUC（1 为完美，0.5 为随机）**给出整体性能。

### 24. Explain the peaking phenomenon and how it guides data collection and feature selection（峰值现象及其对数据采集和特征选择的指导）
*来源：真题 23-24 S2 3(b)；讲义 F1；专题 05 E3*

**EN:** For a fixed number of training samples, **performance first improves as features are added and then deteriorates** because the **number of parameters grows with d and the estimates become unreliable (overfitting)**. Hence **collect more samples when many features are used (the peak moves to larger d)**, and with limited data **select or extract features to bring d close to the optimum**.

**中文：** 训练样本数固定时，**随特征数增加性能先升后降**，因为**参数个数随 d 增长、估计变得不可靠（过拟合）**。因此**用很多特征就要多采样（峰值右移）**；数据有限时要**做特征选择/提取，把 d 降到接近最优值**。

### 25. Three types of feature subset selection methods: filter, wrapper, embedded（特征子集选择的三类方法）
*来源：真题 23-24 S2 3(c)；讲义 F4；专题 05 E5*

**EN:** **Filter ranks or selects features with a criterion independent of the classifier** (fast, general, but may not suit the final classifier); **wrapper evaluates each subset by the accuracy of the classifier itself** (accurate but expensive and prone to overfitting); **embedded selects features during training** (e.g. lasso, decision trees), a compromise in cost and accuracy.

**中文：** **Filter 用与分类器无关的准则排序/选择特征**（快、通用，但未必适合最终分类器）；**Wrapper 用分类器本身的精度评价每个子集**（准确但开销大、易过拟合）；**Embedded 在训练过程中完成选择**（如 Lasso、决策树），在开销与精度间折中。

### 26. Draw the block diagrams of the filter and wrapper methods（画出 filter 与 wrapper 的框图）
*来源：讲义 F5；专题 05 E4；真题 21-22 S2（旧卷）*

**EN:** Both loop **"all features → search for a subset → evaluate → stop? → selected subset → classifier"**; the **only difference is the evaluation box**, which uses a **separability criterion in filter** and the **accuracy of a trained classifier (with feedback to the search) in wrapper**.

**中文：** 两者都是循环**"全部特征 → 搜索子集 → 评价 → 是否停止 → 选出子集 → 分类器"**，**唯一区别在评价框**：**filter 用可分性准则**，**wrapper 用训练分类器得到的精度（并反馈给搜索）**。

### 27. Suggest a feature selection method for a given dataset and describe the procedure（为给定数据集推荐特征选择方法并描述步骤）
*来源：讲义 F8；真题 22-23 S2（旧卷）*

**EN:** For many features, **pre-screen with a filter (Fisher ratio or mutual information) and then run a wrapper with sequential forward selection**, using **cross-validated classifier accuracy as the criterion** and **stopping when adding a feature no longer improves it**.

**中文：** 特征很多时，**先用 filter（Fisher 比或互信息）预筛，再用 wrapper + 顺序前向选择精选**，评价准则取**分类器的交叉验证精度**，**加入特征不再提升时停止**。

---

## F. 数据准备（23-24 S1 整题，之后没再考，背清单即可）

### 28. Data types in machine learning（数据类型）
*来源：真题 23-24 S1 2(a)；讲义 G1；专题 07 E1*

**EN:** **Nominal** (labels without order), **ordinal** (ordered but intervals not meaningful), **interval** (differences meaningful, no true zero) and **ratio** (true zero, ratios meaningful); the **type decides which operations and statistics are valid** (e.g. mode for nominal, median for ordinal).

**中文：** **名义型**（无顺序的标签）、**有序型**（有顺序但间隔无意义）、**区间型**（差有意义、无真正的零）、**比率型**（有真正的零、比值有意义）；**数据类型决定哪些运算和统计量有效**（如名义型用众数，有序型用中位数）。

### 29. How to handle outliers（离群点的处理）
*来源：真题 23-24 S1 2(b)；讲义 G2；专题 07 E2*

**EN:** **Detect** them with the **z-score (> 3), the IQR rule (outside Q₁−1.5·IQR or Q₃+1.5·IQR) or a box plot**, find the cause, then **remove, cap (winsorise), treat as missing, transform, or use robust statistics such as the median**; keep genuine outliers when they matter (e.g. fraud).

**中文：** **检测**用 **z 分数（> 3）、IQR 规则（落在 Q₁−1.5·IQR 或 Q₃+1.5·IQR 之外）或箱线图**，先查明原因，再选择**删除、截尾（winsorize）、当缺失值处理、变换，或用中位数等稳健统计量**；真实且重要的离群点（如欺诈）应保留。

### 30. How to handle missing values（缺失值的处理）
*来源：真题 23-24 S1 2(c)；讲义 G3；专题 07 E3*

**EN:** **Delete the samples (few, random missing) or the feature (high missing ratio)**, or **impute with the mean/median/mode (preferably within each class), with a model (KNN or regression), or add a "missing" indicator**; the **imputation statistics must come from the training set only**.

**中文：** **删除样本（缺失少且随机）或删除特征（缺失比例高）**，或**用均值/中位数/众数填补（最好按类别分别计算）、用模型填补（KNN 或回归），或增加"缺失"指示变量**；**填补用的统计量只能来自训练集**。

### 31. Features with different ranges: normalisation and standardisation（不同量纲的特征：归一化与标准化）
*来源：真题 23-24 S1 2(d)；讲义 G4；专题 07 E4*

**EN:** **Large-range features dominate distance- and gradient-based methods**, so rescale them: **min-max (x−min)/(max−min) maps to [0,1] but is sensitive to outliers**, **z-score (x−μ)/σ gives mean 0 and variance 1**; **estimate the parameters on the training set only** (tree models need no scaling).

**中文：** **取值范围大的特征会主导基于距离和梯度的方法**，所以要缩放：**min-max (x−min)/(max−min) 映射到 [0,1]，对离群点敏感**；**z-score (x−μ)/σ 得到均值 0、方差 1**；**缩放参数只用训练集估计**（树模型不需要缩放）。

---

## G. 聚类（Q4，每份卷子都有，论述占 5～12 分）

### 32. List four types of clustering methods with an application scenario for each（四类聚类方法及应用场景）
*来源：真题 23-24 S2 4(a)、25-26 S1 4(a)；讲义 H1；专题 06 E1*

**EN:** **Centroid-based** (K-means: compact, similar-sized clusters in large numeric data, e.g. customer segmentation), **hierarchical** (linkage and dendrogram: small data where the hierarchy matters, e.g. taxonomy or gene expression), **density-based** (DBSCAN: arbitrary shapes and noise, e.g. spatial or GPS data) and **distribution-based** (GMM with EM: soft assignments of overlapping elliptical clusters, e.g. image segmentation).

**中文：** **基于质心**（K-means：大规模数值数据中紧凑、大小相近的簇，如客户细分）、**层次**（linkage 与树状图：需要层次结构的小数据，如分类学、基因表达）、**基于密度**（DBSCAN：任意形状和噪声，如空间/GPS 数据）、**基于分布**（GMM + EM：重叠的椭圆簇和软分配，如图像分割）。

### 33. Describe the DBSCAN procedure（DBSCAN 的步骤）
*来源：讲义 H2；专题 06 E4；真题 21-22 S2、23-24 S1（旧卷）*

**EN:** With **radius ε and MinPts**, label each point as **core (at least MinPts points in its ε-neighbourhood), border (within a core point's neighbourhood) or noise**; **start a cluster at an unvisited core point and keep adding all density-reachable points**, border points join a neighbouring core's cluster, and noise belongs to no cluster.

**中文：** 给定**半径 ε 和 MinPts**，把每个点标为**核心点（ε 邻域内至少 MinPts 个点）、边界点（落在某核心点的邻域内）或噪声**；**从未访问的核心点开始新簇，不断加入密度可达的点**，边界点归入相邻核心点的簇，噪声不属于任何簇。

### 34. Choose a clustering algorithm for the given data and justify it（为给定的数据选聚类算法并说明理由）
*来源：真题 23-24 S1 4(a)(b)；讲义 H3*

**EN:** For **ring-shaped (non-convex) data with noise choose DBSCAN**, because it **follows density connectivity whereas K-means cuts the plane by perpendicular bisectors and cannot separate concentric rings**; for **three compact, roughly spherical clusters choose K-means**, which is fast and fits convex clusters.

**中文：** **环形（非凸）且有噪声的数据选 DBSCAN**，因为它**按密度连通性聚类，而 K-means 用中垂线划分平面，分不开同心圆环**；**三个紧凑、近似球形的簇选 K-means**，速度快且适合凸簇。

### 35. Describe the K-means algorithm and its limitations; can it be used for irregularly shaped clusters?（K-means 及其局限，不规则形状能用吗）
*来源：真题 24-25 S2 4(a)(c)；讲义 H4；专题 06 E2*

**EN:** **Choose K and initial centroids, assign each sample to the nearest centroid, recompute the centroids as the means, and repeat until the assignments stop changing** (minimising WCSS, only a local optimum). It **needs K, is sensitive to initialisation and outliers, and assumes convex, similar-sized, spherical clusters**, so for irregular shapes use **DBSCAN, single linkage or GMM** instead.

**中文：** **选定 K 和初始质心，把每个样本分给最近的质心，用均值更新质心，重复到分配不再变化**（最小化 WCSS，只能得到局部最优）。它**需要给定 K，对初值和离群点敏感，假设簇是凸的、大小相近的球状簇**，所以不规则形状应改用 **DBSCAN、single linkage 或 GMM**。

### 36. K-means vs DBSCAN with outliers; limitations of DBSCAN and how to choose ε（有离群点时 K-means 与 DBSCAN；DBSCAN 局限与 ε 的选法）
*来源：模拟卷 2 Q4(b)*

**EN:** **K-means forces outliers into some cluster and drags the centroids, whereas DBSCAN labels them as noise.** DBSCAN **works poorly when cluster densities differ and is sensitive to ε and MinPts**; **choose ε at the elbow of the sorted k-distance plot (k = MinPts−1)**.

**中文：** **K-means 会把离群点硬分进某个簇并拉偏质心，而 DBSCAN 把它们标成噪声。** DBSCAN **在各簇密度差别大时效果差，且对 ε 和 MinPts 敏感**；**ε 取排序后的 k-distance 图的肘部（k = MinPts−1）**。

### 37. How to determine K（如何确定 K）
*来源：真题 24-25 S2 4(b)、25-26 S2 4(c)；讲义 H5；专题 06 E3*

**EN:** Run K-means for K = 1…Kmax and **pick the elbow of the WCSS-versus-K curve (never the minimum WCSS, which is 0 at K = N)**, or **choose the K that optimises a validity index (Silhouette, Dunn or CH largest, DB smallest)**, or use domain knowledge.

**中文：** 对 K = 1…Kmax 分别跑 K-means，**取 WCSS–K 曲线的肘部（不能取 WCSS 最小的 K，因为 K = N 时 WCSS = 0）**；或**选使有效性指标最优的 K（Silhouette、Dunn、CH 最大，DB 最小）**；或根据领域知识确定。

### 38. List five clustering evaluation metrics（五个聚类评价指标）
*来源：真题 23-24 S1 4(c)；讲义 H6；专题 06 E6*

**EN:** **Silhouette, Dunn index, Davies-Bouldin index, Calinski-Harabasz index** (internal indices that combine within-cluster cohesion and between-cluster separation; larger is better except DB) and the **Rand index** (an external index that compares a partition with the ground truth).

**中文：** **Silhouette、Dunn 指数、Davies-Bouldin 指数、Calinski-Harabasz 指数**（内部指标，综合簇内紧凑度与簇间分离度，除 DB 越小越好外都是越大越好）和 **Rand 指数**（外部指标，把划分与真实标签比较）。

### 39. Describe the Silhouette coefficient (and why it is useful)（Silhouette 系数及用途）
*来源：真题 24-25 S1 4(c)；模拟卷 1 Q4(c)；讲义 H6；专题 06 E5*

**EN:** **sᵢ = (bᵢ−aᵢ)/max(aᵢ, bᵢ) ∈ [−1, 1]**, where **aᵢ is the mean distance to the other samples of its own cluster and bᵢ is the minimum distance to samples of the other clusters** (course definition): **near 1 is well clustered, near 0 on a boundary, negative probably misassigned**; the average evaluates a clustering and helps choose K.

**中文：** **sᵢ = (bᵢ−aᵢ)/max(aᵢ, bᵢ) ∈ [−1, 1]**，其中 **aᵢ 是到本簇其他样本的平均距离，bᵢ 是到其他簇样本的最小距离**（课件定义）：**接近 1 说明聚得好，接近 0 在簇边界上，为负说明可能分错**；平均值可用来评价聚类和选 K。

### 40. Describe the Dunn index（Dunn 指数）
*来源：真题 25-26 S1 4(c)；讲义 H6；专题 06 E5*

**EN:** **Dunn = (smallest distance between samples of different clusters) / (largest within-cluster diameter)**; it **rewards compact, well-separated clusters (larger is better)** but looks only at extreme pairs and is therefore **sensitive to noise and outliers**.

**中文：** **Dunn = （不同簇样本间的最小距离）/（簇内最大直径）**；**奖励紧凑且分离的簇（越大越好）**，但只看极端样本对，**对噪声和离群点敏感**。

### 41. Advantages and disadvantages of hierarchical clustering; compare single and complete linkage（层次聚类优缺点；single 与 complete linkage 比较）
*来源：讲义 H7；专题 06 E7*

**EN:** Hierarchical clustering **needs no preset K and gives a dendrogram**, but **costs at least O(n²) and its greedy merges cannot be undone**. **Single linkage (minimum distance) follows elongated shapes but suffers from chaining and noise**, **complete linkage (maximum distance) gives compact clusters of similar diameter but is sensitive to outliers**, and average/centroid linkage lie in between.

**中文：** 层次聚类**不需要预设 K，并给出树状图**，但**至少 O(n²) 的开销，贪心合并不可撤销**。**Single linkage（最小距离）能跟随细长形状，但有链式效应、对噪声敏感**；**complete linkage（最大距离）得到紧凑、直径相近的簇，但对离群点敏感**；average/centroid linkage 介于两者之间。

---

## 附：被合并的重复题对照

| 同一问题 | 出现位置 | 本文题号 |
|---|---|---|
| ξ 和 C 的作用 | 25-26 S2 3(a)；模拟卷 1 Q2(a)；讲义 B3；专题 02 E3 | 11 |
| 为什么只用 0<α<C 求 b | 仿真 M5 第 6 题；模拟卷 1 Q2(b)；讲义 B4；专题 02 E4 | 12 |
| 四类聚类方法 | 23-24 S2 4(a)；25-26 S1 4(a)；讲义 H1；专题 06 E1 | 32 |
| 怎么选 K | 24-25 S2 4(b)；25-26 S2 4(c)；讲义 H5；专题 06 E3 | 37 |
| LDA 与 SVM 的区别 | 23-24 S2 2(c)；讲义 A4；专题 01 E7；专题 02 E6 | 5 |
| 随机森林（构造 + 为什么更好） | 讲义 D3；专题 04 E5/E6；模拟卷 2；仿真 M5 | 19 |
| k 折交叉验证 | 讲义 E2；专题 05 E2；仿真 M5 | 22 |
| 峰值现象 | 23-24 S2 3(b)；讲义 F1；专题 05 E3 | 24 |

> **没有收进来的**：纯计算题（LDA/SVM/NB 数值计算、层次聚类和 K-means 手算）不属于论述题；画框图和树状图只保留要点。**遗传算法（GA）的 11 道论述题已按要求从本文删除**（真题 23-24 S1/S2、24-25 S1/S2、25-26 S1/S2 的 Q1 论述），完整答案仍在各历年卷参考答案 md 里。其余被删的自拟论述题（如 LDA 的 S_W⁻¹ 作用、LDA vs PCA、GMM、curse of dimensionality 等）在讲义 §4.4 和专题里仍有完整答案，需要时回去查。

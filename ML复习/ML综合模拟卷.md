# EE6407 机器学习部分 综合模拟卷

> **说明**
> - 按真题结构出题：Q2、Q3 是监督学习，Q4 是聚类，每题 25 分，共 75 分（真卷另有 Q1 遗传算法 25 分）。
> - 建议用时 **1 小时 50 分钟**（真卷 4 题约 150 分钟，每题约 37 分钟）。不能用电脑，可以用计算器。
> - 题型覆盖历年真题的高频考点，也加了几种"还没考过、但很可能轮到"的变体：average linkage、entropy 选根节点、soft-margin SVM 中 α = C 的点被误分类。
> - 题目用英文，和真卷一致。答案在后半部分，数字都用程序核对过，附得分点。
> - 公式用普通符号书写，任何 md 阅读器都能直接看。

---

# PART 1 — QUESTIONS

## Question 2 (25 marks)

**(a)** Formulate the primal optimization problem of the linear soft-margin support vector machine. Explain the roles of the slack variables ξ(i) and the parameter C.
(7 marks)

**(b)** A linear SVM is trained with C = 2 and gives the following support vectors:

| i | x(i) | d(i) | α(i) |
|---|---|---|---|
| 1 | (2.9, −0.5) | −1 | 0.9713 |
| 2 | (2.0, 1.7) | −1 | 2 |
| 3 | (1.2, 2.6) | −1 | 0.3279 |
| 4 | (2.6, 0.8) | +1 | 2 |
| 5 | (2.7, 2.3) | +1 | 1.2992 |

(i) Determine the weight vector w and the bias b. Justify which support vectors you use to compute b.
(ii) Classify x_a = (2.5, 1.5) and x_b = (3.5, 3.0). For each, compute the distance to the decision boundary and comment on the confidence of the classification.
(iii) State where support vectors 2 and 4 lie relative to the margin and the decision boundary.
(12 marks)

**(c)** A fraud-detection classifier is tested on 2000 transactions (positive = fraud). The confusion matrix is (rows = predicted, columns = actual):

| | Actual Positive | Actual Negative |
|---|---|---|
| **Predicted Positive** | 45 | 35 |
| **Predicted Negative** | 15 | 1905 |

Compute the accuracy, precision, recall and F-score. Is accuracy an adequate measure here? Explain briefly.
(6 marks)

---

## Question 3 (25 marks)

Class ω₁ has N₁ = 50 training samples and class ω₂ has N₂ = 40. Their mean vectors and **scatter matrices** are:

- m₁ = [2.0, 3.0]ᵀ,  S₁ = [[60, 18], [18, 45]]
- m₂ = [4.0, 1.5]ᵀ,  S₂ = [[56, −10], [−10, 36]]

**(a)** Design a Fisher linear discriminant classifier. Use it to classify x = [3.0, 2.5]ᵀ, compute the distance of x to the decision boundary, and comment on the confidence of the result.
(10 marks)

**(b)** State the assumptions of the Gaussian Naive Bayes classifier. Using the same data, classify x = [3.0, 2.5]ᵀ with Gaussian Naive Bayes, and give the posterior probability of the chosen class.
(8 marks)

**(c)** A data set of 160 samples (80 in ω₁, 80 in ω₂) has three binary features A, B and C with the following sample distributions:

| Feature value | ω₁ | ω₂ |
|---|---|---|
| A = 1 | 50 | 10 |
| A = 0 | 30 | 70 |
| B = 1 | 40 | 40 |
| B = 0 | 40 | 40 |
| C = 1 | 60 | 30 |
| C = 0 | 20 | 50 |

Determine the root node of a classification tree based on the **entropy** impurity measure, and give the information gain of the selected feature.
(7 marks)

---

## Question 4 (25 marks)

Six samples are given:

P1 = (7, 9),  P2 = (8, 6),  P3 = (7, 1),  P4 = (1, 0),  P5 = (7, 2),  P6 = (4, 9)

**(a)** Cluster the six samples using the agglomerative hierarchical clustering algorithm with Euclidean distance and **average linkage**. Show the distance updates at each step and sketch the dendrogram.
(10 marks)

**(b)** Apply K-means clustering with K = 2 to the same samples, using P1 and P2 as the initial cluster centres. Perform the iterations until convergence, showing the assignment and the new centres in each iteration, and give the final within-cluster sum of squares (WCSS).
(9 marks)

**(c)** Describe the Silhouette coefficient. For the final K-means result in (b), compute the Silhouette coefficients of P2 and P4, and interpret the values.
(6 marks)

**— END OF PAPER —**

---
---

# PART 2 — ANSWERS AND MARKING

## Question 2

### 2(a) Soft-margin primal (7 marks)

**Formulation.** Given {x(i), d(i)}, i = 1, …, N, d(i) ∈ {+1, −1}, find w, b and ξ(i) that

  minimize J(w, ξ) = ½ wᵀw + C Σᵢ ξ(i)

  subject to d(i)[wᵀx(i) + b] ≥ 1 − ξ(i),  ξ(i) ≥ 0,  i = 1, …, N.

**ξ(i).** When the data are not linearly separable, the hard constraints d(i)[wᵀx(i) + b] ≥ 1 cannot all hold. The slack ξ(i) relaxes the constraint and measures how much sample i violates it:
- ξ = 0: on or outside the margin, correctly classified;
- 0 < ξ ≤ 1: inside the margin but on the correct side;
- ξ > 1: misclassified.

So Σξ(i) is an upper bound on the number of training errors.

**C.** C is a user-specified hyperparameter that controls the trade-off between margin and error:
- the term ½wᵀw maximizes the margin 2/‖w‖;
- CΣξ(i) penalizes violations.

A large C gives a narrow margin and few training errors, and risks overfitting. A small C gives a wide margin and tolerates more violations, and risks underfitting. In the dual, C becomes the upper bound 0 ≤ α(i) ≤ C.

> 得分点：目标函数 1.5 分；两条约束 1.5 分；ξ 的含义和三种取值 2 分；C 的作用（折中、大小的影响）2 分。

### 2(b) SVM calculation (12 marks)

**Check.** Σα(i)d(i) = (2 + 1.2992) − (0.9713 + 2 + 0.3279) = 3.2992 − 3.2992 = 0 ✓

**(i) w = Σ α(i)d(i)x(i)**

| i | α(i)d(i) | α(i)d(i)x(i) |
|---|---|---|
| 1 | −0.9713 | (−2.81677, 0.48565) |
| 2 | −2 | (−4.0, −3.4) |
| 3 | −0.3279 | (−0.39348, −0.85254) |
| 4 | +2 | (5.2, 1.6) |
| 5 | +1.2992 | (3.50784, 2.98816) |
| **Sum** | | **w = (1.4976, 0.8213)** |

**b.** Only support vectors with **0 < α < C** lie exactly on the margin (g = ±1), so only they can be used to compute b.
- Here those are SV 1, 3 and 5. SV 2 and SV 4 have α = C = 2 and are excluded.
- **Reason (KKT).** β(i) = C − α(i) and β(i)ξ(i) = 0. If α < C, then ξ = 0 and the sample is on the margin. If α = C, ξ may be positive, so g(x) is not ±1.

| i | d(i) | wᵀx(i) | b = d(i) − wᵀx(i) |
|---|---|---|---|
| 1 | −1 | 3.93238 | −4.93238 |
| 3 | −1 | 3.93241 | −4.93241 |
| 5 | +1 | 5.93241 | −4.93241 |

**b = −4.9324**（三个值一致，说明 w 正确）。

**g(x) = 1.4976 x₁ + 0.8213 x₂ − 4.9324**
- g > 0 → class +1；g < 0 → class −1。
- ‖w‖ = 1.7080，1/‖w‖ = 0.5855，margin 2/‖w‖ = 1.1710。

> 陷阱：如果误用 SV 2 或 SV 4（α = C）求 b，会得到 −5.3913 或 −3.5508，都是错的。

**(ii) Classification and confidence**

| Sample | g(x) | Class | r = \|g\|/‖w‖ | Compare with 1/‖w‖ = 0.5855 | Confidence |
|---|---|---|---|---|---|
| x_a = (2.5, 1.5) | +0.0435 | +1 | 0.0255 | \|g\| ≪ 1, inside the margin, almost on the boundary | **Low** |
| x_b = (3.5, 3.0) | +2.7730 | +1 | 1.6235 | \|g\| > 1, outside the margin, about 2.8 times farther than the support vectors | **High** |

**(iii)** Substitute into d·g:

| SV | α | d·g | ξ = 1 − d·g | Position |
|---|---|---|---|---|
| 2 | C | 0.5411 | 0.4589 | inside the margin, correct side (0 < ξ ≤ 1) |
| 4 | C | −0.3817 | 1.3817 | wrong side of the boundary, **misclassified** (ξ > 1) |

This is consistent with α = C ⇔ ξ > 0.

> 得分点：验算 Σαd 1 分；w 3 分；说明只用 0 < α < C 的支持向量及理由 2 分；b 2 分；两个测试点的类别、距离、置信度 3 分；SV 2、4 的位置 1 分。

### 2(c) Confusion matrix (6 marks)

TP = 45，FP = 35，FN = 15，TN = 1905（行是预测，列是真实）。

| Metric | Formula | Value |
|---|---|---|
| Accuracy | (TP + TN)/N = 1950/2000 | **0.975** |
| Precision | TP/(TP + FP) = 45/80 | **0.5625** |
| Recall | TP/(TP + FN) = 45/60 | **0.75** |
| F-score | 2PR/(P + R) = 2·45/(90 + 35 + 15) | **0.6429** |

**No, accuracy is not adequate.**
- The classes are highly imbalanced: only 60 of 2000 transactions are fraud.
- A trivial classifier that labels everything "not fraud" would already reach an accuracy of 1940/2000 = 0.97 while detecting no fraud at all.
- Here 25% of the frauds are missed (recall 0.75), and 44% of the alarms are false (precision 0.5625).
- Precision, recall and F-score (or the ROC/AUC) describe the performance on the class of interest much better.

> 得分点：四个指标各 1 分；类别不平衡的评论 2 分。易错：行列方向读反会把 precision 和 recall 互换。

---

## Question 3

### 3(a) Fisher LDA (10 marks)

1. **S_W** = S₁ + S₂ = [[116, 8], [8, 81]]
2. **Determinant**: det = 116 × 81 − 8² = 9396 − 64 = 9332
3. **Inverse**: S_W⁻¹ = (1/9332)[[81, −8], [−8, 116]]
4. **m₁ − m₂** = [−2, 1.5]ᵀ, so
   - w = (1/9332)[81(−2) − 8(1.5), −8(−2) + 116(1.5)]ᵀ = (1/9332)[−174, 190]ᵀ = **[−0.018646, 0.020360]ᵀ**
5. **m₁ + m₂** = [6, 4.5]ᵀ, so
   - w₀ = −½ wᵀ(m₁ + m₂) = −½(−0.111876 + 0.091620) = **0.010128**
6. **Classifier**: **g(x) = −0.018646 x₁ + 0.020360 x₂ + 0.010128**
   - g > 0 → ω₁；g < 0 → ω₂。
   - Boundary: x₂ = 0.9158 x₁ − 0.4974.
   - Self-check: g(m₁) = 0.033914 > 0, so ω₁ is on the positive side ✓

**Classify x = [3.0, 2.5]ᵀ**
- g(x) = −0.055938 + 0.050900 + 0.010128 = **+0.005090 > 0 → ω₁**
- ‖w‖ = 0.027608, so r = |g|/‖w‖ = **0.184**.
- For reference, the distance of m₁ to the boundary is 0.033914/0.027608 = 1.228.
- r is only about 15% of that, so x lies close to the decision boundary and the classification is made with **low confidence**: a small perturbation of x or an error in w could change the decision.

> 得分点：S_W 1 分；求逆 2 分；w 2 分；w₀ 1.5 分；判别函数和决策规则 1 分；分类 1 分；距离和带参照的置信度结论 1.5 分。
> 易错：题目给的是散度矩阵，LDA 里直接相加即可，**不要**除以样本数（这一步和 3(b) 相反）。

### 3(b) Gaussian Naive Bayes (8 marks)

**Assumptions**
1. Given the class, the features are conditionally independent: p(x|ωⱼ) = p(x₁|ωⱼ)p(x₂|ωⱼ). Only the diagonal elements are used; the off-diagonal elements 18 and −10 are ignored.
2. Each feature in each class follows a univariate normal distribution. Its variance is estimated by maximum likelihood: σ² = scatter diagonal element / Nⱼ.
3. The priors are the class proportions: P(ω₁) = 50/90 = 0.5556，P(ω₂) = 40/90 = 0.4444.

**Parameters**

| | μ (x₁) | σ² (x₁) | μ (x₂) | σ² (x₂) |
|---|---|---|---|---|
| ω₁ | 2.0 | 60/50 = 1.2 | 3.0 | 45/50 = 0.9 |
| ω₂ | 4.0 | 56/40 = 1.4 | 1.5 | 36/40 = 0.9 |

**Densities** at x = (3.0, 2.5)，p = exp[−(x − μ)²/(2σ²)] / √(2πσ²)：

| | p(x₁\|ωⱼ) | p(x₂\|ωⱼ) | Product | × Prior |
|---|---|---|---|---|
| ω₁ | e^(−1/2.4)/√(2π·1.2) = 0.2401 | e^(−0.25/1.8)/√(2π·0.9) = 0.3660 | 0.08787 | **0.04882** |
| ω₂ | e^(−1/2.8)/√(2π·1.4) = 0.2359 | e^(−1/1.8)/√(2π·0.9) = 0.2413 | 0.05692 | **0.02530** |

**Decision**: 0.04882 > 0.02530 → **ω₁**. Posterior P(ω₁|x) = 0.04882/(0.04882 + 0.02530) = **0.659**.

The result agrees with LDA in 3(a). Both methods place x on the ω₁ side, but neither strongly: LDA gives a small distance and NB gives a posterior of only about 0.66.

> 得分点：三条假设各 1 分；方差除以样本数 1 分；四个一元密度 2 分；乘先验并下结论 1 分（后验算对再加分）。
> 易错：把 60、45 直接当方差用，密度几乎是平的，结论没有意义。

### 3(c) Entropy root node (7 marks)

**Formula**: i_E(N) = −Σⱼ P(ωⱼ) log₂ P(ωⱼ)；加权不纯度 = (n_L/N) i(N_L) + (n_R/N) i(N_R)。
**Parent node**: 80 : 80，so i_E = 1。

| Feature | Branch | Counts | Entropy | Weighted entropy | Information gain |
|---|---|---|---|---|---|
| A | = 1 (60) | 50 : 10 | −(5/6)log₂(5/6) − (1/6)log₂(1/6) = 0.6500 | (60/160)(0.6500) + (100/160)(0.8813) = **0.7946** | **0.2054** |
| | = 0 (100) | 30 : 70 | 0.8813 | | |
| B | both (80) | 40 : 40 | 1 | **1.0000** | 0 |
| C | = 1 (90) | 60 : 30 | 0.9183 | (90/160)(0.9183) + (70/160)(0.8631) = **0.8942** | 0.1058 |
| | = 0 (70) | 20 : 50 | 0.8631 | | |

**Conclusion**: A has the lowest weighted entropy, i.e. the largest information gain (0.2054), so **A is the root node**.

B gives no gain, because both of its branches keep the 1 : 1 class ratio of the parent. Using the Gini impurity (with ½) gives the same choice: A 0.1833 < C 0.2143 < B 0.25.

> 得分点：公式 1 分；六个分支的熵 3 分；三个加权值 2 分；结论和信息增益 1 分。易错：忘记按分支样本数加权；log 没用以 2 为底。

---

## Question 4

### 4(a) Average linkage (10 marks)

**Distance matrix**（例：d(P1, P2) = √(1² + 3²) = √10）

| | P1 | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|
| P1 | 0 | 3.1623 | 8.0000 | 10.8167 | 7.0000 | 3.0000 |
| P2 | | 0 | 5.0990 | 9.2195 | 4.1231 | 5.0000 |
| P3 | | | 0 | 6.0828 | 1.0000 | 8.5440 |
| P4 | | | | 0 | 6.3246 | 9.4868 |
| P5 | | | | | 0 | 7.6158 |
| P6 | | | | | | 0 |

Average linkage: the distance between two clusters is the mean of **all** pairwise distances between their samples (n_A × n_B pairs).

**Step 1.** Minimum d(P3, P5) = **1.0000** → merge into {P3, P5}.
- d(P1, {35}) = (8 + 7)/2 = 7.5
- d(P2, {35}) = (5.0990 + 4.1231)/2 = 4.6111
- d(P4, {35}) = (6.0828 + 6.3246)/2 = 6.2037
- d(P6, {35}) = (8.5440 + 7.6158)/2 = 8.0799

**Step 2.** Minimum d(P1, P6) = **3.0000** → merge into {P1, P6}.
- d(P2, {16}) = (3.1623 + 5)/2 = 4.0811
- d(P4, {16}) = (10.8167 + 9.4868)/2 = 10.1517
- d({16}, {35}) = (8 + 7 + 8.5440 + 7.6158)/4 = 7.7899

**Step 3.** Candidates:
- d(P2, {16}) = **4.0811**
- d(P2, {35}) = 4.6111
- d(P4, {35}) = 6.2037
- d(P2, P4) = 9.2195
- ……

Merge P2 with {P1, P6} at **4.0811**.
- d(P4, {126}) = (10.8167 + 9.2195 + 9.4868)/3 = 9.8410
- d({126}, {35}) = (8 + 7 + 5.0990 + 4.1231 + 8.5440 + 7.6158)/6 = 6.7303

**Step 4.** Candidates: d(P4, {35}) = **6.2037**, d({126}, {35}) = 6.7303, d(P4, {126}) = 9.8410. Merge P4 with {P3, P5} at **6.2037**.

**Step 5.** d({126}, {345}) = mean of the 9 pairwise distances = **7.7672**. All samples are merged.

**Merging order**: (P3, P5) @ 1.0000 → (P1, P6) @ 3.0000 → +P2 @ 4.0811 → (P4 + {P3, P5}) @ 6.2037 → all @ 7.7672

**Dendrogram**（纵轴为合并距离）：

```
7.7672 ─      ┌─────────────┴──────────────┐
6.2037 ─      │                     ┌──────┴──────┐
4.0811 ─ ┌────┴─────┐               │             │
3.0000 ─ ┌─┴─┐      │               │             │
1.0000 ─ │   │      │               │          ┌──┴──┐
         P1  P6     P2              P4         P3    P5
```

Cutting the tree at a height of about 7 gives two clusters, {P1, P2, P6} and {P3, P4, P5}.

> 补充（体现理解）：如果用 **single linkage**，第 3 步 P2 会以 3.1623 并入 {P1, P6}，第 4 步 {P3, P5} 会以 4.1231 并入 {P1, P2, P6}，P4 最后以 6.0828 单独并入。这时 P4 像一个离群点，切成两簇得到的是 {P4} 和其余五个点。average linkage 不受这种"链式"影响。
> 得分点：距离矩阵 2 分；每步更新正确 5 分（平均要按样本对数）；树状图（标高度、不交叉）2 分；合并顺序 1 分。

### 4(b) K-means (9 marks)

Initial centres: c₁ = P1 = (7, 9)，c₂ = P2 = (8, 6)。

**Iteration 1**

| Sample | to c₁ | to c₂ | Cluster |
|---|---|---|---|
| P1 | 0 | 3.1623 | 1 |
| P2 | 3.1623 | 0 | 2 |
| P3 | 8.0000 | 5.0990 | 2 |
| P4 | 10.8167 | 9.2195 | 2 |
| P5 | 7.0000 | 4.1231 | 2 |
| P6 | 3.0000 | 5.0000 | 1 |

New centres:
- c₁ = (P1 + P6)/2 = **(5.5, 9)**
- c₂ = (P2 + P3 + P4 + P5)/4 = (23/4, 9/4) = **(5.75, 2.25)**
- WCSS = 56.0

**Iteration 2**

| Sample | to c₁ (5.5, 9) | to c₂ (5.75, 2.25) | Cluster |
|---|---|---|---|
| P1 | 1.5000 | 6.8648 | 1 |
| P2 | 3.9051 | 4.3732 | **1**（changed） |
| P3 | 8.1394 | 1.7678 | 2 |
| P4 | 10.0623 | 5.2559 | 2 |
| P5 | 7.1589 | 1.2748 | 2 |
| P6 | 1.5000 | 6.9732 | 1 |

New centres:
- c₁ = (P1 + P2 + P6)/3 = (19/3, 8) = **(6.3333, 8)**
- c₂ = (P3 + P4 + P5)/3 = **(5, 1)**

**Iteration 3**

| Sample | to c₁ (6.3333, 8) | to c₂ (5, 1) | Cluster |
|---|---|---|---|
| P1 | 1.2019 | 8.2462 | 1 |
| P2 | 2.6034 | 5.8310 | 1 |
| P3 | 7.0317 | 2.0000 | 2 |
| P4 | 9.6148 | 4.1231 | 2 |
| P5 | 6.0369 | 2.2361 | 2 |
| P6 | 2.5386 | 8.0623 | 1 |

The assignments are unchanged, so the algorithm has **converged**. Final clusters: **{P1, P2, P6}** and **{P3, P4, P5}**.

**WCSS**（final centres）：
- Cluster 1: P1 0.4444 + 1 = 1.4444；P2 2.7778 + 4 = 6.7778；P6 5.4444 + 1 = 6.4444 → 14.6667
- Cluster 2: P3 4 + 0 = 4；P4 16 + 1 = 17；P5 4 + 1 = 5 → 26
- **WCSS = 40.6667**, down from 56.0 after iteration 1.

The result agrees with the two-cluster cut of the average-linkage dendrogram in (a).

> 得分点：每轮距离表 2 分 × 2；新中心 2 分；第 3 轮说明收敛 1 分；WCSS 2 分。
> 易错：第 2 轮 P2 的两个距离只差 0.47，要算准；簇编号跟初始中心走；收敛后也要写出第 3 轮。

### 4(c) Silhouette coefficient (6 marks)

**Definition**（按课件）。For sample xᵢ in cluster D_p:
- aᵢ = average distance from xᵢ to the other samples of D_p (cohesion);
- bᵢ = minimum distance from xᵢ to the samples of the other clusters (separation);
- SCᵢ = (bᵢ − aᵢ) / max(aᵢ, bᵢ), in [−1, 1].

The overall SC is the average of SCᵢ over all samples. A larger SC means compact, well-separated clusters. SCᵢ near 1: well clustered; near 0: on the boundary between clusters; negative: probably assigned to the wrong cluster. SC can also be computed for different K to choose the number of clusters.

（Common textbook definition: bᵢ is the smallest *average* distance to another cluster. State which one you use.）

**Final clusters**: A = {P1, P2, P6}，B = {P3, P4, P5}。

| Sample | a | b | SC |
|---|---|---|---|
| P2 | (3.1623 + 5)/2 = 4.0811 | min(5.0990, 9.2195, 4.1231) = 4.1231 | (4.1231 − 4.0811)/4.1231 = **0.010** |
| P4 | (6.0828 + 6.3246)/2 = 6.2037 | min(10.8167, 9.2195, 9.4868) = 9.2195 | (9.2195 − 6.2037)/9.2195 = **0.327** |

**Interpretation.**
- **P2**: SC ≈ 0. P2 is almost as close to P5 in the other cluster (4.12) as it is, on average, to its own cluster (4.08). It lies on the boundary between the two clusters, which matches K-means, where P2 only switched clusters in iteration 2 and its two distances were close.
- **P4**: SC = 0.33, positive but moderate. P4 belongs to cluster B, but it lies rather far from its own cluster members (average 6.20), so it is a loosely attached, outlier-like member.

(With the textbook definition: P2 0.336, P4 0.370; overall SC with the lecture definition is 0.298.)

> 得分点：定义 a、b、SC 和取值范围 2 分；含义和用途 1 分；P2、P4 的计算 2 分；解释 1 分。

---

## 自评对照表

| 题 | 考点 | 对应讲义 | 满分 | 我的得分 |
|---|---|---|---|---|
| 2(a) | 软间隔原问题、ξ 和 C | 专题 02 §5 | 7 | |
| 2(b) | 由 α 求 w、b（只用 0 < α < C）、置信度 | 专题 02 §5.5、§7 | 12 | |
| 2(c) | 混淆矩阵、类别不平衡 | 专题 05 §2 | 6 | |
| 3(a) | Fisher LDA 设计、距离和置信度 | 专题 01 §7 | 10 | |
| 3(b) | Gaussian NB、散度矩阵要除以 N | 专题 03 §6.4 | 8 | |
| 3(c) | entropy 选根节点 | 专题 04 §4、§7 | 7 | |
| 4(a) | average linkage + 树状图 | 专题 06 §5 | 10 | |
| 4(b) | K-means 手算到收敛 | 专题 06 §3 | 9 | |
| 4(c) | Silhouette 定义与计算 | 专题 06 §8 | 6 | |
| | | | **75** | |

哪一小问丢分最多，就回到对应讲义的那一节，再看一遍例题。

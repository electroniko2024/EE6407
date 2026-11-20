# EE6407 机器学习部分 模拟卷 2（按 2026-27 S1 预测出题）

> **说明**
> - 按 [ML出题规律与预测.md](ML出题规律与预测.md) 的主预测出题：Q2 LDA，Q3 朴素贝叶斯 + 决策树，Q4 聚类。每题 25 分，共 75 分。
> - 建议用时 **1 小时 50 分钟**。不能用电脑，可以用计算器。
> - 专门覆盖第一张 [ML综合模拟卷.md](ML综合模拟卷.md) 没练到的预测考点：
>   - LDA：题目给协方差和不等样本数，要先换算成散度矩阵；w₀ 的论述。
>   - Multinomial NB：带 Laplace 平滑，测试句里有重复词。
>   - 决策树：连续特征和类别特征一起比较，用 entropy。
>   - 随机森林论述。
>   - DBSCAN 手算。
>   - Davies-Bouldin 指数的计算。
> - 题目用英文，和真卷一致。答案在后半部分，数字都用程序核对过，附得分点。
> - 公式用普通符号书写，任何 md 阅读器都能直接看。

---

# PART 1 — QUESTIONS

## Question 2 (25 marks)

A two-class problem has N₁ = 20 training samples in class ω₁ and N₂ = 80 in class ω₂. The sample means and the maximum-likelihood covariance matrices (normalised by Nᵢ) of the two classes are:

m₁ = [4, 3]ᵀ, Σ₁ = [2.0 0.6; 0.6 0.5]
m₂ = [1, 1]ᵀ, Σ₂ = [0.5 −0.1; −0.1 1.5]

**(a)** Design a Fisher linear discriminant classifier for this problem. Give the weight vector w, the threshold w₀ and the decision rule. Show all steps.
(11 marks)

**(b)** Classify the test samples x_A = [2, 3]ᵀ and x_B = [4, 1]ᵀ. For each, compute the distance to the decision boundary and comment on the confidence of the classification.
(6 marks)

**(c)** Explain how the threshold w₀ is determined in Fisher LDA. What assumption is behind this choice? Discuss whether the assumption is reasonable for this problem, and what you would do if it does not hold.
(8 marks)

---

## Question 3 (25 marks)

**(a)** A Multinomial Naive Bayes classifier is trained on the following five short documents:

| Document | Text | Class |
|---|---|---|
| 1 | "new phone camera" | Tech |
| 2 | "phone battery life" | Tech |
| 3 | "fast new chip" | Tech |
| 4 | "fresh pasta sauce" | Food |
| 5 | "new pasta recipe" | Food |

(i) Classify the test document "new phone recipe recipe" using Laplace (add-one) smoothing. Show the conditional probabilities you use.
(ii) What would happen without smoothing? Comment on the confidence of your result in (i).
(10 marks)

**(b)** The table below shows 10 training samples with one continuous feature (Temperature) and one binary feature (Windy). The class label is Play.

| Temperature | 18 | 20 | 22 | 24 | 26 | 27 | 29 | 31 | 33 | 35 |
|---|---|---|---|---|---|---|---|---|---|---|
| Windy | No | Yes | No | No | Yes | No | Yes | No | Yes | Yes |
| Play | Yes | Yes | Yes | Yes | No | Yes | No | Yes | No | No |

Using **entropy impurity**, find the best splitting threshold for Temperature, and determine which feature should be used at the root node of a classification tree. Show your calculations.
(8 marks)

**(c)** Describe how a random forest is constructed and used for classification, explaining both bootstrapping and aggregating. Why does a random forest usually outperform a single classification tree?
(7 marks)

---

## Question 4 (25 marks)

**(a)** Apply DBSCAN with ε = 1.5 and MinPts = 4 to the 14 two-dimensional points below. A point q is in the ε-neighbourhood of p if d(p, q) ≤ ε (Euclidean distance), and p is a core point if its ε-neighbourhood, **including p itself**, contains at least MinPts points.

| Point | A | B | C | D | E | F | G | H | I | J | K | L | M | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x | 1 | 2 | 1 | 2 | 3 | 6 | 7 | 6 | 7 | 8 | 4.5 | 5 | 9 | 10 |
| y | 1 | 1 | 2 | 2.5 | 3 | 6 | 6 | 7 | 7.5 | 8.5 | 6 | 6.8 | 2 | 2.5 |

All pairwise distances not exceeding 2.0 are listed below; every other pair is more than 2.0 apart.

| Pair | AB | AC | AD | BC | BD | CD | DE | FG | FH | FI | FK | FL | GH | GI | HI | HK | HL | IJ | KL | MN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d | 1.000 | 1.000 | 1.803 | 1.414 | 1.500 | 1.118 | 1.118 | 1.000 | 1.000 | 1.803 | 1.500 | 1.281 | 1.414 | 1.500 | 1.118 | 1.803 | 1.020 | 1.414 | 0.943 | 1.118 |

Identify the core, border and noise points, and give the resulting clusters.
(10 marks)

**(b)** (i) How would K-means with K = 2 treat points M and N, and how would they affect the result? (ii) State two limitations of DBSCAN, and explain how ε can be chosen in practice.
(6 marks)

**(c)** Six samples are P1(1, 1), P2(2, 1), P3(1, 3), P4(5, 4), P5(6, 5), P6(6, 3). Two candidate partitions are:

- Partition I: {P1, P2, P3} and {P4, P5, P6}
- Partition II: {P1, P2} and {P3, P4, P5, P6}

Define the Davies-Bouldin (DB) index, compute it for both partitions (use the root-mean-square distance to the centroid as the within-cluster scatter), and state which partition is better.
(9 marks)

---

# PART 2 — ANSWERS AND MARKING

## Question 2

### 2(a) Fisher LDA with covariance matrices (11 marks)

**Step 1: convert covariance to scatter.** The covariances are normalised by Nᵢ, so Sᵢ = Nᵢ·Σᵢ.

S₁ = 20 × [2.0 0.6; 0.6 0.5] = [40 12; 12 10]
S₂ = 80 × [0.5 −0.1; −0.1 1.5] = [40 −8; −8 120]
S_W = S₁ + S₂ = [80 4; 4 130]

**Step 2: invert S_W.**

det(S_W) = 80 × 130 − 4 × 4 = 10384
S_W⁻¹ = (1/10384) · [130 −4; −4 80]

**Step 3: weight vector.** m₁ − m₂ = [3, 2]ᵀ.

adj(S_W)(m₁ − m₂) = [130×3 − 4×2, −4×3 + 80×2] = [382, 148]
w = [382, 148] / 10384 = **[0.036787, 0.014253]ᵀ**

**Step 4: threshold.** m₁ + m₂ = [5, 4]ᵀ.

w₀ = −½ wᵀ(m₁ + m₂) = −½ (0.036787×5 + 0.014253×4) = −½ (0.183937 + 0.057011) = **−0.120474**

**Decision rule.** g(x) = 0.036787·x₁ + 0.014253·x₂ − 0.120474. Assign x to ω₁ if g(x) > 0, otherwise to ω₂.

**Trap.** If Σ₁ + Σ₂ = [2.5 0.5; 0.5 2.0] is used directly, w ∝ [1.053, 0.737]. This direction is quite different, because the two classes have very different sizes (20 vs 80). It also classifies x_A into the wrong class (see 2(b)). Adding covariances directly is valid only when N₁ = N₂.

> 得分点：
> - 换算 Sᵢ = NᵢΣᵢ，2 分。漏了这一步、直接用 Σ₁ + Σ₂，这一问最多给 6 分。
> - 求 S_W，1 分。
> - 写出 det 和逆矩阵，2 分。
> - 求出 w，2 分。
> - 写出 w₀ 的公式并算出数值，2 分。
> - 写出 g(x) 和决策规则，2 分。

### 2(b) Classification and confidence (6 marks)

‖w‖ = √(0.036787² + 0.014253²) = 0.039452
Reference: g(m₁) = 0.069434, so the distance from each class mean to the boundary is |g(mᵢ)|/‖w‖ = **1.760**. By the midpoint rule, the two means are equally far from the boundary.

| Sample | g(x) | Class | r = \|g\|/‖w‖ | r / 1.760 | Confidence |
|---|---|---|---|---|---|
| x_A = [2, 3] | 0.073575 + 0.042758 − 0.120474 = **−0.004141** | **ω₂** | **0.105** | 6% | **Low** |
| x_B = [4, 1] | 0.147149 + 0.014253 − 0.120474 = **0.040928** | **ω₁** | **1.037** | 59% | **Fairly high** |

- x_A lies almost on the decision boundary, only 6% of the mean-to-boundary distance away. A small change in the data or in w could flip its label, so the classification is not reliable. With the wrong scatter matrix it would even be assigned to ω₁ (g = +0.21).
- x_B is on the ω₁ side, more than half-way from the boundary to the class mean, so the classification is reasonably confident.

> 得分点：每个样本分类 1 分、算出距离 1 分；和类中心距离比较并下结论，2 分。

### 2(c) Threshold w₀ and its assumption (8 marks)

> **How it is determined.** w₀ is chosen so that the **midpoint of the two class means lies on the decision boundary**: g((m₁ + m₂)/2) = 0, hence **w₀ = −½wᵀ(m₁ + m₂)**. On the projected line this places the threshold halfway between the projected means m̃₁ and m̃₂.
>
> **Assumption.** The midpoint is optimal only if the **projected classes have the same or similar scatter** and the **classes are equally likely (equal priors)**. Then the two projected distributions are mirror images, and they cross at the midpoint.
>
> **Is it reasonable here? No, both parts are violated.**
> - **Unequal spread:** the projected per-sample variances are wᵀΣ₁w = 3.44×10⁻³ and wᵀΣ₂w = 0.88×10⁻³, so on the projected line ω₁ is about **twice as wide** (std 0.059 vs 0.030). This alone would move the best threshold **towards the compact class ω₂**.
> - **Unequal priors:** estimated from the sample sizes, P(ω₁) = 0.2 and P(ω₂) = 0.8. This alone would move the threshold **towards the less probable class ω₁**.
> - The two effects push in **opposite directions**. If both projected classes are modelled as 1-D Gaussians, they happen to almost cancel here: the crossing point is at g ≈ 0.000. So the midpoint is acceptable for this data set, but only by coincidence, and this should be checked rather than assumed.
>
> **What to do.** (1) Treat the computed w₀ as a starting value, **try several nearby values**, and pick the one with the **best accuracy on training or validation data** (the lecturer's suggestion). (2) If the priors are known to differ, add the correction ln[P(ω₁)/P(ω₂)], which moves the boundary towards the less probable class. (3) If one class is much more spread out, move the threshold towards the more compact class, e.g. place it where the two projected densities cross.

> 【中文】
> **如何确定。** 选 w₀ 使**两类均值的中点落在决策边界上**：g((m₁ + m₂)/2) = 0，所以 **w₀ = −½wᵀ(m₁ + m₂)**。在投影直线上，阈值位于两个投影均值的正中间。
>
> **假设。** 只有当**两类投影后的散度相同或相近**、并且**两类出现的可能性相同（先验相等）**时，中点才是最优的。这时两个投影分布左右对称，正好在中点相交。
>
> **本题是否合理？不合理，两个假设都不成立。**
> - **散度不等**：两类投影后的单样本方差 wᵀΣ₁w = 3.44×10⁻³，wᵀΣ₂w = 0.88×10⁻³，所以投影后 ω₁ 的宽度大约是 ω₂ 的**两倍**（标准差 0.059 对 0.030）。单看这一点，最佳阈值应**往紧凑的 ω₂ 移**。
> - **先验不等**：按样本数估计，P(ω₁) = 0.2，P(ω₂) = 0.8。单看这一点，阈值应**往先验小的 ω₁ 移**。
> - 两个效应**方向相反**。把投影后的两类当作一维高斯，本题中两者恰好几乎抵消，交点在 g ≈ 0.000。所以中点阈值对这组数据还算可以，但这只是巧合，应当检验而不是默认成立。
>
> **怎么办。** (1) 以算出的 w₀ 为起点，**在附近试几个值**，选**训练集或验证集上准确率最高**的那个（老师给的办法）。(2) 已知先验不等时，加上修正项 ln[P(ω₁)/P(ω₂)]，边界往先验小的一类移。(3) 某一类分散得多时，阈值往紧凑的一类移，例如放在两个投影密度的交点处。

> 得分点：
> - 中点原理和公式 2 分。
> - 两个假设（散度相近、先验相等）2 分。
> - 结合本题数据，指出先验不等（20 : 80）和 ω₁ 更分散，并说明两者把阈值往相反方向推，2 分。只答出其中一点给 1 分。
> - 两种补救办法 2 分。

---

## Question 3

### 3(a) Multinomial Naive Bayes (10 marks)

**Counts.**
- Tech has 9 words; Food has 6 words.
- Vocabulary (unique words in all training data): new, phone, camera, battery, life, fast, chip, fresh, pasta, sauce, recipe, so **d = 11**.
- Smoothed denominators: Tech 9 + 11 = **20**; Food 6 + 11 = **17**.
- Priors: P(Tech) = 3/5, P(Food) = 2/5.

| Word | count in Tech | P(w \| Tech) | count in Food | P(w \| Food) |
|---|---|---|---|---|
| new | 2 | 3/20 | 1 | 2/17 |
| phone | 2 | 3/20 | 0 | 1/17 |
| recipe | 0 | 1/20 | 1 | 2/17 |

"recipe" appears **twice** in the test document, so its probability is multiplied twice.

- Tech: (3/5) · (3/20) · (3/20) · (1/20) · (1/20) = 27/800000 = **3.375 × 10⁻⁵**
- Food: (2/5) · (2/17) · (1/17) · (2/17) · (2/17) = 16/417605 = **3.831 × 10⁻⁵**

**Decision: Food.** Normalised posteriors: Food 0.532, Tech 0.468.

**(ii) Without smoothing.**
- P(recipe | Tech) = 0/9 = 0, so the Tech score is 0.
- P(phone | Food) = 0/6 = 0, so the Food score is also 0.
- Both products vanish because of a single unseen word, and **no decision can be made**. Laplace smoothing gives every word a small non-zero probability.

**Confidence.** The two posteriors are close (0.53 vs 0.47), so the classification is **low confidence**. "new" and "phone" point to Tech; only the repeated word "recipe" tips the decision to Food. If "recipe" appeared once, the result would reverse: Tech 6.75×10⁻⁴ vs Food 3.26×10⁻⁴.

> 得分点：
> - d = 11、两个分母、先验，2 分。
> - 列出三个词在两类下的平滑概率，3 分。
> - 两个乘积（recipe 乘两次）和结论，3 分。
> - 不平滑的后果和置信度说明，2 分。

### 3(b) Entropy root node (8 marks)

**Root impurity.** 6 Yes : 4 No.
i(root) = −0.6 log₂0.6 − 0.4 log₂0.4 = **0.9710**

**Temperature.**
- The sorted class sequence is Y Y Y Y N Y N Y N N.
- Candidate thresholds are the midpoints of adjacent values: 19, 21, 23, 25, 26.5, 28, 30, 32, 34.
- Only thresholds where the class changes can be optimal: 25, 26.5, 28, 30, 32.

| t | left (Y:N) | right (Y:N) | weighted entropy |
|---|---|---|---|
| 19 | 1:0 | 5:4 | 0.8920 |
| 21 | 2:0 | 4:4 | 0.8000 |
| 23 | 3:0 | 3:4 | 0.6897 |
| **25** | **4:0** | **2:4** | **0.6 × 0.9183 = 0.5510** |
| 26.5 | 4:1 | 2:3 | 0.8464 |
| 28 | 5:1 | 1:3 | 0.7145 |
| 30 | 5:2 | 1:2 | 0.8797 |
| 32 | 6:2 | 0:2 | 0.6490 |
| 34 | 6:3 | 0:1 | 0.8265 |

Best threshold **t = 25**: weighted entropy 0.5510, information gain 0.9710 − 0.5510 = **0.4200**.

**Windy.**
- Windy = Yes: 1 Yes : 4 No, entropy = −0.2 log₂0.2 − 0.8 log₂0.8 = 0.7219.
- Windy = No: 5 Yes : 0 No, entropy = 0 (pure).
- Weighted entropy = 0.5 × 0.7219 + 0.5 × 0 = **0.3610**; information gain = **0.6100**.

**Root: Windy**, because it has the smaller weighted entropy (0.361 < 0.551), i.e. the larger information gain. The "No" branch is already a pure leaf (Play = Yes). The "Yes" branch (1:4) could be split further, e.g. by Temperature.

> 得分点：
> - 根节点熵 1 分。
> - 写出候选切分点的做法（排序取中点）1 分。
> - 求出 Temperature 的最佳切分点和加权熵 3 分；考场上可以只算类别变化处的候选点，但要说明理由。
> - Windy 的加权熵 2 分。
> - 比较后选根节点 1 分。

### 3(c) Random forest (7 marks)

> A random forest is an **ensemble of classification trees** built by **bagging (bootstrap aggregating)**.
>
> **Bootstrapping.** From the training set of N samples with d features, draw n subsets **by random sampling with replacement**, each of size N. Some samples appear several times in a subset, and others not at all. For each subset, also **select m < d features at random**, and grow a classification tree on that subset using only those features.
>
> **Aggregating.** To classify a new sample, pass it through all n trees. Each tree votes for a class, and the forest outputs the class with the **most votes (majority voting)**.
>
> **Why it is better than a single tree.** A single tree has **high variance**: it overfits easily, and a small change in the data can change the tree a lot. The two sources of randomness, random samples and random features, make the trees **diverse and weakly correlated**, so their errors tend to cancel when the votes are combined. Averaging many low-bias, high-variance trees **reduces variance**. The forest is therefore **more accurate, less prone to overfitting and more robust to noise and outliers**. Each tree uses only a subset of features, which also helps with high-dimensional data. The cost is **more computation and loss of interpretability**.

> 【中文】随机森林是用 **bagging（自助聚合）** 构建的**分类树集成**。
>
> **自助采样。** 从含 N 个样本、d 个特征的训练集中，**有放回地随机抽样**，得到 n 个大小为 N 的子集；有的样本在一个子集里出现多次，有的根本不出现。对每个子集再**随机选 m < d 个特征**，只用这些特征在该子集上建一棵分类树。
>
> **聚合。** 分类新样本时，让它经过全部 n 棵树，每棵树投一票，森林输出**得票最多的类（多数投票）**。
>
> **为什么比单棵树好。** 单棵树**方差大**：容易过拟合，数据稍有变化，树就可能变化很大。两处随机性（随机样本、随机特征）使各棵树**多样、相关性弱**，投票合并时错误相互抵消。把许多低偏差、高方差的树平均起来**降低了方差**，所以森林**更准确、更不容易过拟合、对噪声和离群点更稳健**。每棵树只用一部分特征，对高维数据也有帮助。代价是**计算量更大、失去可解释性**。

> 得分点：bootstrap 有放回抽样 2 分；随机特征子集 1 分；多数投票 1 分；多样性 / 降低方差 2 分；代价 1 分。

---

## Question 4

### 4(a) DBSCAN (10 marks)

**ε-neighbourhoods** (d ≤ 1.5, including the point itself):

| Point | Neighbourhood | Count | Type |
|---|---|---|---|
| A | A, B, C | 3 | border (in the neighbourhood of core B and C) |
| B | A, B, C, D (BD = 1.500 counts) | 4 | **core** |
| C | A, B, C, D | 4 | **core** |
| D | B, C, D, E | 4 | **core** |
| E | D, E | 2 | border (neighbour of core D) |
| F | F, G, H, K, L (FK = 1.500 counts) | 5 | **core** |
| G | F, G, H, I (GI = 1.500 counts) | 4 | **core** |
| H | F, G, H, I, L | 5 | **core** |
| I | G, H, I, J | 4 | **core** |
| J | I, J | 2 | border (neighbour of core I) |
| K | F, K, L | 3 | border (neighbour of core F and L) |
| L | F, H, K, L | 4 | **core** |
| M | M, N | 2 | **noise** |
| N | M, N | 2 | **noise** |

**Clusters.**
- Start from core B: add A, C, D. From core C and D, add E. No new core points can be reached, so **Cluster 1 = {A, B, C, D, E}**. Core points: B, C, D; border points: A, E.
- Start from core F: add G, H, K, L. From G, H, I and L, reach I and J. **Cluster 2 = {F, G, H, I, J, K, L}**. Core points: F, G, H, I, L; border points: J, K.
- **Noise: {M, N}.** They are neighbours of each other, but neither is a core point and neither lies within ε of a core point. Two points together do not form a cluster.

**Watch the equality cases.** BD, FK and GI are exactly 1.5. With "d ≤ ε" they count; with "d < ε", B, D and G would no longer be core points and the result would change. Always state the convention you use.

> 得分点：
> - 14 个点的邻域个数 3 分。
> - 核心点 / 边界点 / 噪声点判对 3 分。
> - 两个簇的扩展过程和结果 3 分。
> - 说明 M、N 为什么是噪声 1 分。

### 4(b) K-means vs DBSCAN; DBSCAN limitations (6 marks)

> (i) K-means **assigns every point to a cluster**; it has no notion of noise. M and N would be forced into whichever cluster has the nearer centre, and they would **pull that centre towards them**, distorting the cluster mean and possibly the boundary. With K = 3 they might even form a cluster of their own. DBSCAN instead labels them as **noise**, because they lie in a low-density region, and the two real clusters remain unaffected.
>
> (ii) Limitations: (1) it works poorly when **clusters have very different densities**, since one ε cannot fit both sparse and dense clusters; (2) it is **sensitive to ε and MinPts**, and ε is hard to choose in high dimensions; (3) border points may be assigned differently depending on the visiting order. **Choosing ε:** for every point, compute the distance to its k-th nearest neighbour (k = MinPts − 1), sort these distances and plot them (**k-distance plot**). Take ε at the **elbow** of the curve. MinPts is usually at least D + 1 (D = dimension), e.g. 2D.

> 【中文】(i) K-means **会把每个点都分到某个簇**，没有"噪声"的概念。M 和 N 会被强行分给中心较近的那个簇，并**把那个中心往它们那边拉**，使簇均值甚至边界都被扭曲；K = 3 时它们甚至可能自成一簇。DBSCAN 则因为它们处在低密度区域而把它们标为**噪声**，两个真正的簇不受影响。
>
> (ii) 局限：(1) **各簇密度差别很大时效果差**，一个 ε 无法同时适应稀疏和密集的簇；(2) **对 ε 和 MinPts 敏感**，高维时 ε 很难选；(3) 边界点的归属可能取决于访问顺序。**怎么选 ε**：对每个点算它到第 k 个最近邻的距离（k = MinPts − 1），排序后画出来（**k-distance 图**），取曲线的**肘部**作为 ε。MinPts 通常至少取 D + 1（D 为维数），例如 2D。

> 得分点：(i) K-means 强制分配并拉偏中心 2 分；(ii) 两个局限 2 分，k-distance 图选 ε 2 分。

### 4(c) Davies-Bouldin index (9 marks)

**Definition.**
- Within-cluster scatter: δᵢ = √((1/nᵢ) Σ_{x∈Dᵢ} ‖x − mᵢ‖²).
- Between-cluster separation: d(i, j) = ‖mᵢ − mⱼ‖.
- Similarity of two clusters: Sᵢⱼ = (δᵢ + δⱼ)/d(i, j).
- DB = (1/K) Σᵢ maxⱼ≠ᵢ Sᵢⱼ. **Smaller is better**: compact clusters that are far apart give small Sᵢⱼ.

**Partition I.**
- Cluster {P1, P2, P3}: m = (4/3, 5/3). Squared distances 5/9, 8/9, 17/9, mean 10/9, so δ = √(10/9) = **1.0541**.
- Cluster {P4, P5, P6}: m = (17/3, 4). Squared distances 4/9, 10/9, 10/9, mean 8/9, so δ = √(8/9) = **0.9428**.
- d = √((13/3)² + (7/3)²) = √218 / 3 = **4.9216**.
- S₁₂ = (1.0541 + 0.9428)/4.9216 = 0.4057. With K = 2, both clusters have the same maximum, so **DB = 0.406**.

**Partition II.**
- Cluster {P1, P2}: m = (1.5, 1). Squared distances 0.25, 0.25, so δ = **0.5**.
- Cluster {P3, P4, P5, P6}: m = (4.5, 3.75). Squared distances 12.8125, 0.3125, 3.8125, 2.8125, mean 4.9375, so δ = **2.2220**.
- d = √(3² + 2.75²) = √16.5625 = **4.0697**.
- S₁₂ = (0.5 + 2.2220)/4.0697 = 0.6689, so **DB = 0.669**.

**Conclusion.** **Partition I is better** (0.406 < 0.669). Putting P3 with the right-hand cluster makes that cluster much more spread out (δ = 2.22), and it also brings the two centroids closer together.

> 得分点：
> - 写出定义并说明"越小越好"2 分。
> - 两个划分的质心和 δ 各 2 分。
> - 两个 DB 值 2 分。
> - 结论和解释 1 分。

---

## 自评对照表

| 题 | 考点 | 对应讲义 | 满分 | 我的得分 |
|---|---|---|---|---|
| 2(a) | LDA 设计；协方差换算成散度矩阵 Sᵢ = NᵢΣᵢ | 专题 01 §7.2、§7.4；复习讲义 §3.1 | 11 | |
| 2(b) | 分类、距离、置信度 | 专题 01 §1.4、§7 | 6 | |
| 2(c) | w₀ 的原理、假设、补救 | 专题 01 §5、E3 | 8 | |
| 3(a) | Multinomial NB + Laplace 平滑、重复词 | 专题 03 §5.5、§6.6 | 10 | |
| 3(b) | entropy 选根节点、连续特征切分点 | 专题 04 §4、§5、§8 | 8 | |
| 3(c) | 随机森林 | 专题 04 §11、E5 | 7 | |
| 4(a) | DBSCAN 手算 | 专题 06 §7.2–§7.4 | 10 | |
| 4(b) | K-means 没有噪声概念；DBSCAN 局限、选 ε | 专题 06 §7.5、§7.6 | 6 | |
| 4(c) | Davies-Bouldin 指数 | 专题 06 §8.3；复习讲义 §3.10 | 9 | |
| | | | **75** | |

哪一小问丢分最多，就回到对应讲义的那一节，再看一遍例题。

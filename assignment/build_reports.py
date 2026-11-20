"""Build the two explanatory Markdown documents and three PDF deliverables.

Run after both solutions and verify_results.py:
    python build_reports.py
Dependencies: reportlab, pypdf. The saved JSON results are the numerical source.
"""
from pathlib import Path
import json
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
R1 = json.loads((ROOT / "Assignment1/results/results.json").read_text(encoding="utf-8"))
R2 = json.loads((ROOT / "Assignment2/results/results.json").read_text(encoding="utf-8"))
VERIFY = json.loads((ROOT / "verification.json").read_text(encoding="utf-8"))
assert VERIFY["status"] == "passed"


def fmt(v, digits=6):
    return f"{v:.{digits}f}"


def mdtable(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def write_explanations():
    class_rows = [[c, n, fmt(p, 9)] for c, n, p in zip(R1["class_order"], R1["class_counts"], R1["priors"])]
    means = [[c, *map(fmt, row)] for c, row in zip(R1["class_order"], R1["means"])]
    variances = [[c, *map(fmt, row)] for c, row in zip(R1["class_order"], R1["variances_mle"])]
    predictions = [[i, *[f"{v:.1f}" for v in x], y]
                   for i, (x, y) in enumerate(zip(R1["test_features"], R1["predicted_labels"]), 1)]
    fences = [[f, *[fmt(R1[key][j], 2) for key in ["q1", "q3", "iqr", "lower", "upper"]]]
              for j, f in enumerate("ABCD")]
    outliers = [[d["row"], d["cell"], d["value"], fmt(d["upper"], 2), d["label"]]
                for d in R1["outlier_details"]]
    coeff = [[c, *[fmt(v, 9) for v in row]] for c, row in zip(R1["class_order"], R1["quadratic_coefficients"])]
    lin = [[c, *[fmt(v, 9) for v in row]] for c, row in zip(R1["class_order"], R1["linear_coefficients"])]
    doc1 = r"""# Assignment 1 说明文档：高斯朴素贝叶斯分类

## 1. 作业要求与交付物

原题 `Assignment 1.pdf` 要求：(1) 识别训练集缺失值、异常值，各说明三种处理方法；(2) 删除受影响样本，给出朴素贝叶斯的设计步骤与判别函数；(3) 预测全部 30 个测试样本。

`Assignment1_Report.pdf` 是英文作答报告，根目录 `EE6407_Assignments_1_and_2.pdf` 合并了两份作业。原题要求合并 PDF 提交，且明确代码无需提交。本说明文档和代码用于理解、核查和复现。

训练文件实际名称为 `TrainingData (2).xlsx`，有 120 行、5 列，无标题行。A–D 为特征，E 为标签。测试集 `TestData.xlsx` 有 30 行、4 列。下文行号均为从 1 开始的原始 Excel 行号，未删除任何重复样本，也未推断标签对应的物种。

## 2. 缺失值、异常值与清洗结果

发现 3 个以 `?` 标记的缺失单元格：**A39、A61、D83**。先删除这 3 行，剩余 117 个完整样本。

原题没有指定异常值规则。本解采用一次性、逐特征的 Tukey **1.5 IQR** 规则：只在这 117 行的 A–D 上计算 Q1 和 Q3，分位数采用 `linear` 插值；IQR = Q3 - Q1，下界 L = Q1 - 1.5 IQR，上界 U = Q3 + 1.5 IQR。任一特征严格落在区间 [L,U] 外，删除整行。边界内含端点；不使用标签列或测试集，不迭代重新计算阈值。

{{FENCES}}

该规则发现以下 5 个异常样本：

{{OUTLIERS}}

第 91、96、117 行的数值偏离显著。第 53、98 行只是所选规则判定的统计异常，**不能据此断言是录入错误**。不同异常值定义可能产生不同的清洗集合，因此作答明确公开规则。

最终删除行号为 **39、53、61、83、91、96、98、117**，共 8 行，保留 **112 行**。逐行处理理由见 `results/cleaning_audit.csv`；清洗后数据及对应原始行号见 `results/cleaned_training.csv`。

### 缺失值的三种处理方法

1. **删除样本**：移除任何存在缺失特征的整行。实现简单，适合缺失比例较小的情况；会损失数据，缺失并非随机时也可能改变类别分布。本题后续采用此法。
2. **简单插补**：用训练集特征均值或中位数填充。中位数对极端值更稳健；但固定填充值会压缩变异并削弱特征间关系。交叉验证时仅用该折训练部分估计填充值。
3. **模型插补**：利用其他特征，通过 kNN、回归或多重插补预测缺失值。可利用特征之间的信息，但增加模型假设与复杂度；预处理模型只能拟合训练数据，不能使用未知测试标签。

### 异常值的三种处理方法

1. **检测后删除**：用 IQR、稳健距离或领域规则标记后删除整行。可以避免极端值拉偏参数，但也可能删除真实的少数样本。本题采用明确的 IQR 删除规则。
2. **缩尾（Winsorization）**：将越界值限制为预先确定的训练集上下界，保留其余信息。能减小极端值影响，但会改变原始观测。
3. **稳健估计或降权**：例如以中位数/MAD估计位置和尺度，或在训练时降低极端样本权重。保留观测并降低其影响，但与本题指定的直接删除方案不同，判别器需使用相应稳健模型。

## 3. 朴素贝叶斯设计步骤

1. 按上述规则得到 112 个完整、未被标记为异常的训练样本。
2. 保留原始标签 1、2、3，按类分组。估计经验先验 $\pi_c=n_c/N$。
3. 假设同一类别内 4 个特征条件独立，且每个特征服从单变量高斯分布。
4. 估计 $\mu_{cj}=\frac{1}{n_c}\sum_{i:y_i=c}x_{ij}$，$\sigma_{cj}^2=\frac{1}{n_c}\sum_{i:y_i=c}(x_{ij}-\mu_{cj})^2$。这里使用**最大似然方差**，分母为 $n_c$，不是无偏方差的 $n_c-1$。
5. 对每个测试样本计算三类对数判别分数，取最大者。计算使用完整精度；所有方差均大于 0，无须方差平滑。

先验及样本数：

{{CLASS}}

均值（特征顺序为 A、B、C、D）：

{{MEANS}}

MLE 方差（不是标准差）：

{{VARIANCES}}

### 判别函数

类别条件似然为

$$p(\mathbf{x}\mid c)=\prod_{j=1}^{4}\frac{1}{\sqrt{2\pi\sigma_{cj}^2}}\exp\left[-\frac{(x_j-\mu_{cj})^2}{2\sigma_{cj}^2}\right].$$

根据 Bayes 公式，$p(\mathbf{x})$ 对所有类别相同，不影响最大值比较，因此采用

$$g_c(\mathbf{x})=\ln\pi_c-\frac12\sum_{j=1}^{4}\left[\ln(2\pi\sigma_{cj}^2)+\frac{(x_j-\mu_{cj})^2}{\sigma_{cj}^2}\right],\quad \hat y=\arg\max_{c\in\{1,2,3\}}g_c(\mathbf{x}).$$

若发生完全相同的最大分数，代码按类别顺序选较小标签；本测试集无此类并列。条件独立指的是**给定类别后**独立，而不是所有样本混在一起时独立。

还可写为 $g_c(\mathbf{x})=b_c+\sum_{j=1}^{4}(a_{cj}x_j^2+d_{cj}x_j)$，其中 $a=-1/(2\sigma^2)$，$d=\mu/\sigma^2$，$b_c=\ln\pi_c-\frac12\sum_j[\ln(2\pi\sigma_{cj}^2)+\mu_{cj}^2/\sigma_{cj}^2]$。

二次项系数 $a$：

{{QUADRATIC}}

一次项系数 $d$：

{{LINEAR}}

常数项：{{CONSTANTS}}

表格数值为展示用舍入值，精确参数保存于 `results/results.json` 和 `results/model.npz`。

## 4. 全部测试预测

严格按 `TestData.xlsx` 原始顺序排列：

{{PREDICTIONS}}

第 11 个样本预测为 **3**，不是依靠其所在的行段猜测类别。其类别 2 和类别 3 的对数分数分别为 {{SCORE2}} 和 {{SCORE3}}，类别 3 分数更高。完整的三类分数与模型后验概率见 `results/test_predictions.csv`。后验概率是模型输出，不代表经过验证的正确率。

### 异常值规则敏感性

辅助方案只删除缺失行和明显极端的第 91、96、117 行，保留第 53、98 行，训练样本数为 114。重新拟合后，**全部 30 个测试预测与主方案一致**。辅助方案只检查这两个边界样本的影响，主报告的参数、先验和判别函数始终使用 112 行方案；没有依据测试表现选择规则。

## 5. 如何复现与验证

推荐 Python 3.10 或更新版本。在根目录安装依赖后运行：

```powershell
python -m pip install -r requirements.txt
python Assignment1/solution.py
python Assignment2/solution.py
python verify_results.py
python build_reports.py
```

`solution.py` 从原始 Excel 文件重算清洗、参数、预测，覆盖本作业 `results` 中的派生结果；不会改动原始数据。`verify_results.py` 从原始输入独立复现清洗，与 `sklearn.naive_bayes.GaussianNB(var_smoothing=0.0)` 核对均值、方差、后验概率和预测，结果全部通过。也核对每行测试数据的原始顺序及输入文件 SHA-256。

两份作业的测试文件都没有提供真实标签，**无法计算测试准确率**。这里交付的是按公开假设计算出的类别预测，没有编造测试成绩。

### 文件索引

- `Assignment1_Report.pdf`：英文完整作答。
- `solution.py`：清洗及 Gaussian Naive Bayes 实现。
- `results/results.json`：全精度参数、行号、预测及敏感性结果。
- `results/test_predictions.csv`：30 个样本、3 类对数分数、模型后验及预测标签。
- `results/cleaning_audit.csv`：每个原始训练行的保留／删除原因。
- `results/cleaned_training.csv`：112 行清洗后训练数据。
- `results/model.npz`：可直接载入的模型参数。
"""
    replacements = {
        "FENCES": mdtable(["特征", "Q1", "Q3", "IQR", "下界 L", "上界 U"], fences),
        "OUTLIERS": mdtable(["原始行", "单元格", "数值", "对应上界", "类别"], outliers),
        "CLASS": mdtable(["类别", "样本数", "先验"], class_rows),
        "MEANS": mdtable(["类别", "A", "B", "C", "D"], means),
        "VARIANCES": mdtable(["类别", "A", "B", "C", "D"], variances),
        "QUADRATIC": mdtable(["类别", "A", "B", "C", "D"], coeff),
        "LINEAR": mdtable(["类别", "A", "B", "C", "D"], lin),
        "CONSTANTS": "；".join(f"$b_{c}={fmt(v,9)}$" for c,v in zip(R1["class_order"], R1["constants"])),
        "PREDICTIONS": mdtable(["测试序号", "A", "B", "C", "D", "预测类别"], predictions),
        "SCORE2": fmt(R1["log_scores"][10][1], 6), "SCORE3": fmt(R1["log_scores"][10][2], 6),
    }
    for key, value in replacements.items():
        doc1 = doc1.replace("{{" + key + "}}", value)
    (ROOT / "Assignment1/说明文档.md").write_text(doc1, encoding="utf-8")

    mean_table = [["+1", R2["class_counts"]["+1"], *map(fmt, R2["mean_positive"])],
                  ["-1", R2["class_counts"]["-1"], *map(fmt, R2["mean_negative"])]]
    weights = [[f"w{i}", fmt(w, 12)] for i, w in enumerate(R2["w"], 1)] + [["w0", fmt(R2["w0"], 12)]]
    scatter = [[i, *[fmt(v, 6) for v in row]] for i, row in enumerate(R2["within_scatter"], 1)]
    predicted = [[i, *[fmt(v, 6) for v in x], fmt(g, 12), "+1" if y == 1 else "-1"]
                 for i, (x, g, y) in enumerate(zip(R2["test_features"], R2["test_scores"], R2["predicted_labels"]), 1)]
    doc2 = r"""# Assignment 2 说明文档：Fisher 线性判别分类

## 1. 作业要求与数据

原题 `Assignment2.pdf` 要求：(1) 训练 Fisher 线性判别器，给出权重向量 $\mathbf w$、偏置 $w_0$、决策规则；(2) 给出全部 23 个测试样本的预测标签。

`Assignment2_Report.pdf` 是英文完整作答，根目录 `EE6407_Assignments_1_and_2.pdf` 是两份作业的合并报告。本说明文档解释计算约定、复现方法和验证证据。

`data_train.mat` 内的 `data_train` 为 577×5，每行是一个样本；`label_train.mat` 内的 `label_train` 为 577×1；`data_test.mat` 内的 `data_test` 为 23×5。所有特征均为有限数值。标签为 **+1、-1**，不会把原标签改成 0、1，也不会转置特征矩阵。

## 2. Fisher 判别方向的计算

### 类均值

分别按标签统计样本数、计算均值：

{{MEANS}}

### 类内散度

$$S_+=\sum_{i:y_i=+1}(\mathbf x_i-\mathbf m_+)(\mathbf x_i-\mathbf m_+)^T,\quad S_-=\sum_{i:y_i=-1}(\mathbf x_i-\mathbf m_-)(\mathbf x_i-\mathbf m_-)^T,$$

$$S_W=S_++S_-.$$

这里的 $S_W$ 是**未除以样本数的散度矩阵**，不是协方差矩阵。

{{SCATTER}}

$S_W$ 满秩，条件数约为 **{{COND}}**。不需要加正则项或使用伪逆。代码用 `np.linalg.solve` 解线性方程，避免显式求逆。

### 最大化 Fisher 准则

定义 $\boldsymbol\Delta=\mathbf m_+-\mathbf m_-$，类间散度 $S_B=\boldsymbol\Delta\boldsymbol\Delta^T$。投影后的类间差异与类内离散程度之比为

$$J(\mathbf w)=\frac{\mathbf w^TS_B\mathbf w}{\mathbf w^TS_W\mathbf w}=\frac{[\mathbf w^T(\mathbf m_+-\mathbf m_-)]^2}{\mathbf w^TS_W\mathbf w}.$$

其最大化方向满足

$$\mathbf w\propto S_W^{-1}(\mathbf m_+-\mathbf m_-).$$

本解固定符号，使类别 +1 的投影均值大于类别 -1 的投影均值，选择

$$\boxed{\mathbf w=S_W^{-1}(\mathbf m_+-\mathbf m_-)}.$$

## 3. 偏置与决策规则

Fisher 准则确定投影方向，但不唯一确定分类阈值。原题没有指定阈值选择法。本解明确采用**两类投影均值的中点**；在共享协方差的高斯模型下，该阈值与等先验、等误判代价的 LDA 决策一致：

$$t=\frac12\mathbf w^T(\mathbf m_++\mathbf m_-),\qquad w_0=-t.$$

采用同一尺度的权重与偏置：

{{WEIGHTS}}

$$g(\mathbf x)=\mathbf w^T\mathbf x+w_0,\qquad\hat y=\begin{cases}+1,&g(\mathbf x)\ge 0,\\-1,&g(\mathbf x)<0.\end{cases}$$

训练集投影均值：类别 +1 为 {{MPROJ}}，类别 -1 为 {{NPROJ}}，中点阈值 $t=$ {{THRESHOLD}}。代入各自均值，类别 +1 的分数为正、类别 -1 为负，说明方向与标签映射一致。

### 为什么不同实现的权重可能相差常数倍

Fisher 方向有尺度不唯一性。只要同时把 $\mathbf w$ 与 $w_0$ 乘同一个**正数**，边界和预测不变。若方向乘负数，还必须反转决策标签。不要只归一化权重而保留原偏置。

单位长度版本为：

$$\mathbf w_{\mathrm{unit}}=[{{UNITW}}]^T,\qquad w_{0,\mathrm{unit}}={{UNITB}}.$$

若使用合并无偏协方差 $\Sigma_p=S_W/(N-2)=S_W/575$，其权重和偏置是本报告数值的 **575 倍**，预测一致。本解没有使用样本比例 301/577 与 276/577 改变中点阈值。等先验下的 LDA 比较才与这里的规则对应；经验先验会带来额外的对数先验偏置。

## 4. 全部测试预测

按 `data_test` 原始行顺序排列。分数使用未归一化散度版本：

{{PREDICTIONS}}

预测序列为：第 **1–9** 个测试样本均为 **-1**；第 **10–23** 个均为 **+1**。这是逐行代入模型计算的结果，没有按行段预设答案。

表格中的特征和参数为展示用舍入值，计算使用完整精度。全精度结果见 `results/results.json`，CSV 见 `results/test_predictions.csv`。

## 5. 复现方法和验证

在根目录执行：

```powershell
python -m pip install -r requirements.txt
python Assignment1/solution.py
python Assignment2/solution.py
python verify_results.py
python build_reports.py
```

`Assignment2/solution.py` 仅需 numpy、scipy，直接从 MAT 文件读取数据，重算均值、散度、权重、偏置与 23 个预测；不会修改原始文件。

独立验证 `verify_results.py` 使用 `LinearDiscriminantAnalysis(solver="svd", priors=[0.5,0.5])`，在统一方向和 575 倍尺度后核对权重、偏置、判别分数及全部预测，检查全部通过。还检查输入文件 SHA-256 未变、CSV 顺序一致、报告保留 12 位小数后的权重与偏置不会改变任何预测。

线性方程相对残差 $\|S_W\mathbf w-\boldsymbol\Delta\|_2/\|\boldsymbol\Delta\|_2=$ **{{RESIDUAL}}**。

在训练集上回代的正确率为 **{{TRAINACC}}**，仅作为实现检查，属于训练重代入指标，不是泛化成绩。测试集没有真实标签，**不能计算测试准确率**。

### 文件索引

- `Assignment2_Report.pdf`：英文完整作答。
- `solution.py`：Fisher 分类器实现。
- `results/results.json`：全精度均值、散度、系数、分数和预测。
- `results/test_predictions.csv`：23 个样本、判别分数和标签。
- `results/model.npz`：可直接载入的权重、偏置及训练统计量。
"""
    replacements2 = {
        "MEANS": mdtable(["类别", "样本数", "x1", "x2", "x3", "x4", "x5"], mean_table),
        "SCATTER": mdtable(["行", "列1", "列2", "列3", "列4", "列5"], scatter),
        "COND": fmt(R2["within_scatter_condition_number"], 6),
        "WEIGHTS": mdtable(["系数", "数值"], weights),
        "MPROJ": fmt(R2["projected_positive_mean"], 12), "NPROJ": fmt(R2["projected_negative_mean"], 12),
        "THRESHOLD": fmt(R2["projected_midpoint"], 12),
        "UNITW": ", ".join(map(lambda v: fmt(v, 9), R2["unit_w"])), "UNITB": fmt(R2["unit_w0"], 9),
        "PREDICTIONS": mdtable(["测试序号", "x1", "x2", "x3", "x4", "x5", "g(x)", "预测类别"], predicted),
        "RESIDUAL": f"{R2['linear_solve_relative_residual']:.3e}",
        "TRAINACC": f"{100*R2['training_resubstitution_accuracy']:.4f}%",
    }
    for key, value in replacements2.items():
        doc2 = doc2.replace("{{"+key+"}}", value)
    assert "{{" not in doc1 and "{{" not in doc2
    (ROOT / "Assignment2/说明文档.md").write_text(doc2, encoding="utf-8")


def register_fonts():
    # Prefer a portable system or installed plotting-library font when available.
    candidates = [
        Path("D:/Anaconda3/Lib/site-packages/matplotlib/mpl-data/fonts/ttf"),
        Path("/usr/share/fonts/truetype/dejavu"),
    ]
    try:
        import matplotlib
        candidates.insert(0, Path(matplotlib.get_data_path()) / "fonts/ttf")
    except ImportError:
        pass
    for directory in candidates:
        if (directory / "DejaVuSans.ttf").exists():
            pdfmetrics.registerFont(TTFont("ReportFont", str(directory / "DejaVuSans.ttf")))
            pdfmetrics.registerFont(TTFont("ReportBold", str(directory / "DejaVuSans-Bold.ttf")))
            pdfmetrics.registerFontFamily("ReportFont", normal="ReportFont", bold="ReportBold", italic="ReportFont", boldItalic="ReportBold")
            return "ReportFont", "ReportBold"
    return "Helvetica", "Helvetica-Bold"


FONT, BOLD = register_fonts()
NAVY = colors.HexColor("#18344C")
MUTED = colors.HexColor("#52616D")
STYLES = {
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=20, leading=26, textColor=NAVY, spaceAfter=8),
    "subtitle": ParagraphStyle("subtitle", fontName=FONT, fontSize=10, leading=15, textColor=MUTED, spaceAfter=14),
    "h": ParagraphStyle("h", fontName=BOLD, fontSize=11.5, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=6),
    "p": ParagraphStyle("p", fontName=FONT, fontSize=9.2, leading=14, spaceAfter=7),
    "small": ParagraphStyle("small", fontName=FONT, fontSize=8.2, leading=12, textColor=MUTED, spaceAfter=6),
    "eq": ParagraphStyle("eq", fontName=FONT, fontSize=10, leading=17, spaceBefore=3, spaceAfter=8, leftIndent=8),
}


def P(txt, style="p"):
    return Paragraph(txt, STYLES[style])


def H(txt):
    return P(txt, "h")


def T(headers, rows, widths=None, size=8.3, pad=5):
    data = [list(map(str, headers))] + [list(map(str, row)) for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), size),
        ("FONTNAME", (0, 0), (-1, 0), BOLD), ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#18252F")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"), ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), pad), ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#CAD4DC")),
    ]))
    return t


def title(number, subtitle):
    return [P(f"Assignment {number}", "title"), P(subtitle, "subtitle")]


def assignment1_story():
    r = R1
    story = title(1, "EE6407 Genetic Algorithms &amp; Machine Learning | Gaussian Naive Bayes")
    story += [H("1. Missing values and outlier detection"),
              P('The workbook <b>TrainingData (2).xlsx</b> has 120 rows and no header: columns A-D are features, and column E contains the class labels. Row and cell references below use the original, one-based Excel numbering. <b>TestData.xlsx</b> contains 30 unlabeled samples.'),
              P('Three feature values are missing and marked with <b>?</b>. I remove those rows before calculating the outlier thresholds.'),
              T(["Excel cell", "Original row", "Feature", "Marker"], [["A39",39,"A","?"],["A61",61,"A","?"],["D83",83,"D","?"]], [125,125,125,125]),
              H("Outlier rule and thresholds"),
              P('Because the question does not specify an outlier rule, I use Tukey\'s <b>1.5 IQR rule</b> once on the 117 complete samples. Quartiles use linear interpolation. For each feature, IQR = Q3 - Q1, with lower and upper fences L = Q1 - 1.5 IQR and U = Q3 + 1.5 IQR. A row is flagged if any A-D value falls outside [L, U]. I do not recalculate the fences after deletion, and neither labels nor test data enter this calculation.'),
              T(["Feature","Q1","Q3","IQR","Lower L","Upper U"], [[f,*[fmt(r[k][j],2) for k in ["q1","q3","iqr","lower","upper"]]] for j,f in enumerate("ABCD")], [60,88,88,88,88,88]),
              H("Samples flagged by this rule"),
              T(["Original row","Cell","Value","Upper limit","Class"], [[d["row"],d["cell"],fmt(d["value"],1),fmt(d["upper"],2),d["label"]] for d in r["outlier_details"]], [100,100,100,100,100]),
              Spacer(1,8),
              P('Rows 91, 96 and 117 are the most extreme. Rows 53 and 98 are less extreme, but still fall outside the chosen fences. These are statistical flags; on their own, they do not show that the values are recording errors.'),
              P('After removing the three incomplete rows and five flagged rows, the analysis uses <b>112 samples</b> (original rows 39, 53, 61, 83, 91, 96, 98 and 117 were removed). No duplicate rows were removed.'),
              PageBreak()]

    story += title(1, "Question 1: handling methods | Question 2: classifier design")
    story += [H("Three methods for missing values"),
              P('<b>1. Complete-case deletion.</b> Remove any row with a missing feature. This is straightforward when few values are missing, but discards data and can bias the retained sample if missingness is systematic. I use this approach for the classifier below.'),
              P('<b>2. Mean or median imputation.</b> Fill a missing value with a feature statistic estimated from the training set. The median is less affected by extreme values, though either fixed replacement can reduce variability and distort feature relationships.'),
              P('<b>3. Model-based imputation.</b> Estimate the missing value from the other observed features, for example with k-nearest neighbors, regression, or multiple imputation. This can make use of relationships among features, but adds modeling assumptions. Fit the imputer on training data only.'),
              H("Three methods for outliers"),
              P('<b>1. Detect and delete.</b> Flag extreme rows using IQR, robust distances, or justified domain limits, then remove them. This can keep extreme values from dominating parameter estimates, but may also discard valid rare observations. The main analysis uses IQR deletion.'),
              P('<b>2. Winsorization.</b> Replace values beyond chosen limits with the limits themselves. This keeps the rows and reduces the effect of extreme values, although it changes the observed data. Set the limits using training data.'),
              P('<b>3. Robust estimation or downweighting.</b> Estimate location and scale with robust statistics such as the median and MAD, or give extreme observations less weight when fitting. The observations remain in the data, but the estimator differs from the ordinary MLE model used here.'),
              H("2. Gaussian Naive Bayes design steps"),
              P('<b>Step 1:</b> Apply the documented deletion rule. <b>Step 2:</b> Partition the 112 retained samples by their original labels 1, 2 and 3. <b>Step 3:</b> Estimate each class prior and each feature mean and variance. <b>Step 4:</b> Compute a log-posterior score for each class. <b>Step 5:</b> Predict the class with the largest score.'),
              P('For Gaussian Naive Bayes, I treat the four features as <b>conditionally independent given the class</b>; each class-feature pair has its own univariate Gaussian distribution. This does not assume that the features are independent in the pooled data.'),
              P('π<sub>c</sub> = n<sub>c</sub> / N; &nbsp; μ<sub>cj</sub> = (1/n<sub>c</sub>) Σ<sub>i:yᵢ=c</sub> x<sub>ij</sub><br/>σ<sub>cj</sub><super>2</super> = (1/n<sub>c</sub>) Σ<sub>i:yᵢ=c</sub> (x<sub>ij</sub> - μ<sub>cj</sub>)<super>2</super>', 'eq'),
              P('I use the maximum-likelihood variance, whose denominator is <b>n<sub>c</sub></b> rather than n<sub>c</sub> - 1. Every fitted variance is positive, so no smoothing is needed. Predictions use the unrounded parameters.'),
              PageBreak()]

    story += title(1, "Question 2: fitted parameters and discriminant functions")
    story += [T(["Class c","Retained n_c","Prior π_c"], [[c,n,fmt(p,9)] for c,n,p in zip(r["class_order"],r["class_counts"],r["priors"])], [140,180,180]),
              H("Estimated means μ"),
              T(["Class","A","B","C","D"], [[c,*map(fmt,row)] for c,row in zip(r["class_order"],r["means"])], [60,110,110,110,110]),
              H("Estimated variances σ² (MLE)"),
              T(["Class","A","B","C","D"], [[c,*map(fmt,row)] for c,row in zip(r["class_order"],r["variances_mle"])], [60,110,110,110,110]),
              H("Logarithmic discriminant"),
              P('p(x | c) = ∏<sub>j=1..4</sub> [(2πσ<sub>cj</sub><super>2</super>)<super>-1/2</super> exp{-(x<sub>j</sub> - μ<sub>cj</sub>)<super>2</super>/(2σ<sub>cj</sub><super>2</super>)}<br/>g<sub>c</sub>(x) = ln π<sub>c</sub> - (1/2) Σ<sub>j=1..4</sub> [ln(2πσ<sub>cj</sub><super>2</super>) + (x<sub>j</sub> - μ<sub>cj</sub>)<super>2</super>/σ<sub>cj</sub><super>2</super>]<br/><b>Decision: ŷ = arg max<sub>c∈{1,2,3}</sub> g<sub>c</sub>(x).</b>', 'eq'),
              P('The evidence p(x) is common to all classes, so it does not affect the argmax. Using log scores also avoids multiplying very small densities. If two scores tie exactly, the implementation selects the smaller label; no test sample has a tie.'),
              P('An explicit numerical form is g<sub>c</sub>(x) = b<sub>c</sub> + Σ<sub>j=1..4</sub> (a<sub>cj</sub>x<sub>j</sub><super>2</super> + d<sub>cj</sub>x<sub>j</sub>), where a = -1/(2σ²), d = μ/σ² and b<sub>c</sub> = ln π<sub>c</sub> - (1/2)Σ<sub>j</sub>[ln(2πσ<sub>cj</sub><super>2</super>) + μ<sub>cj</sub><super>2</super>/σ<sub>cj</sub><super>2</super>].'),
              P('; &nbsp; '.join(f'b<sub>{c}</sub> = {fmt(v,9)}' for c,v in zip(r["class_order"],r["constants"])), "small"),
              P('Together, the parameter tables and this expression define the three discriminant functions. Values shown here are rounded; predictions use full precision.'),
              H("Sensitivity and verification"),
              P('As a sensitivity check, I refit the model after keeping rows 53 and 98 and removing only the three incomplete rows and the more extreme rows 91, 96 and 117. The 114-row fit gives <b>the same predictions for all 30 test samples</b>; the reported parameters and scores come from the 112-row fit.'),
              P('I also checked the results against GaussianNB with var_smoothing = 0. The fitted means, MLE variances, posterior probabilities and predictions agree. The test set was not used to estimate parameters or choose the deletion rule.'),
              PageBreak()]

    story += title(1, "Question 3: predictions for all 30 test samples")
    story += [P('Rows follow the original order in <b>TestData.xlsx</b>. The scores g<sub>1</sub>, g<sub>2</sub> and g<sub>3</sub> use the 112-sample main model; the largest score determines the prediction.'),
              T(["Test","A","B","C","D","g1","g2","g3","Label"],
                [[i,*[fmt(v,1) for v in x],*[fmt(v,5) for v in scores],y]
                 for i,(x,scores,y) in enumerate(zip(r["test_features"],r["log_scores"],r["predicted_labels"]),1)],
                [34,32,32,32,32,94,86,86,72],size=7.6,pad=3.8),
              Spacer(1,8),
              P('For sample 11, class 3 has the highest log score (-3.648565, compared with -7.311077 for class 2), so the model assigns it to class 3. The test set has no true labels, so <b>test accuracy cannot be calculated</b>.', "small")]
    return story


def assignment2_story():
    r = R2
    story = title(2, "EE6407 Genetic Algorithms &amp; Machine Learning | Fisher linear discriminant")
    story += [H("1. Training data and class statistics"),
              P('The MATLAB files contain 577 training rows with five features, 577 labels, and 23 unlabeled test rows. Each row is one sample. I retain the original labels, <b>+1 and -1</b>; all feature values are finite.'),
              T(["Class","n","Mean x1","Mean x2","Mean x3","Mean x4","Mean x5"],
                [["+1",r["class_counts"]["+1"],*map(fmt,r["mean_positive"])],["-1",r["class_counts"]["-1"],*map(fmt,r["mean_negative"])]],
                [40,40,84,84,84,84,84],size=7.6,pad=4),
              H("Within-class scatter and Fisher direction"),
              P('m<sub>c</sub> = (1/n<sub>c</sub>)Σ<sub>i:yᵢ=c</sub> x<sub>i</sub><br/>S<sub>c</sub> = Σ<sub>i:yᵢ=c</sub> (x<sub>i</sub> - m<sub>c</sub>)(x<sub>i</sub> - m<sub>c</sub>)<super>T</super>; &nbsp; S<sub>W</sub> = S<sub>+</sub> + S<sub>-</sub><br/>Δ = m<sub>+</sub> - m<sub>-</sub>; &nbsp; S<sub>B</sub> = ΔΔ<super>T</super><br/>J(w) = (w<super>T</super>S<sub>B</sub>w)/(w<super>T</super>S<sub>W</sub>w) = (w<super>T</super>Δ)<super>2</super>/(w<super>T</super>S<sub>W</sub>w)<br/><b>Choose w = S<sub>W</sub><super>-1</super>(m<sub>+</sub> - m<sub>-</sub>).</b>', 'eq'),
              P('S<sub>W</sub> is the unnormalized within-class scatter matrix, formed by summing centered outer products. It has rank 5 and condition number '+fmt(r["within_scatter_condition_number"],6)+', so I solve S<sub>W</sub>w = Δ directly without regularization.'),
              T(["S_W","Column 1","Column 2","Column 3","Column 4","Column 5"],
                [[i,*[fmt(v,6) for v in row]] for i,row in enumerate(r["within_scatter"],1)], [40,92,92,92,92,92],size=7.3,pad=4),
              H("Bias and classification threshold"),
              P('Fisher\'s criterion determines the projection direction, but a threshold is also needed to classify samples. I use the <b>midpoint of the projected class means</b>. With shared class covariance, this is the LDA threshold for equal priors and equal misclassification costs:'),
              P('t = (1/2)w<super>T</super>(m<sub>+</sub> + m<sub>-</sub>); &nbsp; w<sub>0</sub> = -t<br/>g(x) = w<super>T</super>x + w<sub>0</sub><br/><b>Predict +1 if g(x) ≥ 0; predict -1 if g(x) &lt; 0.</b>', 'eq'),
              T(["w1","w2","w3","w4","w5","w0"], [[*[fmt(v,12) for v in r["w"]],fmt(r["w0"],12)]], [83.3]*6,size=7.6,pad=3.5),
              Spacer(1,6),
              P('The projected class means are '+fmt(r["projected_positive_mean"],12)+' (+1) and '+fmt(r["projected_negative_mean"],12)+' (-1), with midpoint '+fmt(r["projected_midpoint"],12)+'.', "small"),
              PageBreak()]

    story += title(2, "Question 2: predictions for all 23 test samples")
    story += [P('Apply the classifier to <b>data_test</b> in its original order, using the unnormalized-scatter coefficients above. True labels are not supplied, so <b>test accuracy is unavailable</b>.'),
              T(["Test","x1","x2","x3","x4","x5","g(x)","Label"],
                [[i,*[fmt(v,6) for v in x],fmt(score,9),"+1" if y==1 else "-1"]
                 for i,(x,score,y) in enumerate(zip(r["test_features"],r["test_scores"],r["predicted_labels"]),1)],
                [34,66,66,66,66,66,90,46],size=7.5,pad=3),
              Spacer(1,7),
              P('Applying g(x) gives label -1 for samples 1-9 and +1 for samples 10-23.'),
              H("Coefficient scale and verification"),
              P('Multiplying both w and w<sub>0</sub> by the same positive constant leaves the decision boundary unchanged.'),
              P('Using the pooled covariance S<sub>W</sub>/575 instead scales both coefficients by 575. An independent SVD LDA fit with equal priors [0.5, 0.5] gives the same boundary, scores and 23 labels after this rescaling. Unequal empirical priors would shift the threshold.'),
              P('The relative solve residual is ||S<sub>W</sub>w - Δ||<sub>2</sub>/||Δ||<sub>2</sub> = '+f'{r["linear_solve_relative_residual"]:.3e}'+'. The displayed coefficients are rounded to 12 decimals; the calculation uses full precision.', "small")]
    return story


def build_pdf(path, story, number):
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CAD4DC"))
        canvas.setLineWidth(.5)
        canvas.line(46, 39, A4[0]-46, 39)
        canvas.setFont(FONT, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(46, 26, f"EE6407 | Assignment {number}")
        canvas.drawRightString(A4[0]-46, 26, f"Page {doc.page}")
        canvas.restoreState()
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=46, leftMargin=46,
                            topMargin=40, bottomMargin=52,
                            title=f"EE6407 Assignment {number}", author="",
                            subject="Classification methods, fitted parameters, and complete test predictions")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    write_explanations()
    p1 = ROOT / "Assignment1/Assignment1_Report.pdf"
    p2 = ROOT / "Assignment2/Assignment2_Report.pdf"
    build_pdf(p1, assignment1_story(), 1)
    build_pdf(p2, assignment2_story(), 2)
    counts = [len(PdfReader(p).pages) for p in [p1, p2]]
    # Intentional pagination is a quality check, not merely an export check.
    if counts != [4, 2]:
        raise ValueError(f"Unexpected pagination: {counts}; adjust layout before delivery")
    merged = ROOT / "EE6407_Assignments_1_and_2.pdf"
    writer = PdfWriter()
    writer.append(p1, outline_item="Assignment 1 - Gaussian Naive Bayes")
    writer.append(p2, outline_item="Assignment 2 - Fisher linear discriminant")
    writer.add_metadata({"/Title": "EE6407 Assignments 1 and 2", "/Subject": "Complete classification assignment answers"})
    with merged.open("wb") as f:
        writer.write(f)
    assert len(PdfReader(merged).pages) == 6
    print(json.dumps({"individual_pages": counts, "merged_pages": 6,
                      "pdfs": [str(p1), str(p2), str(merged)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

# EE6407 Assignment 1 & 2

完整英文作答已合并为 [EE6407_Assignments_1_and_2.pdf](EE6407_Assignments_1_and_2.pdf)，共 6 页。内容包括异常值规则、缺失值与异常值各三种处理方法、Gaussian Naive Bayes 的设计与参数、Fisher 权重与偏置、全部 53 个测试预测。

原题要求两份作业合并为一个 PDF；Assignment 1 明示代码无须提交。以下文件用于学习、复现和核查。

| 作业 | 中文说明 | 英文报告 | 实现 | 预测明细 |
| --- | --- | --- | --- | --- |
| Assignment 1 | [说明文档](Assignment1/说明文档.md) | [报告](Assignment1/Assignment1_Report.pdf) | [solution.py](Assignment1/solution.py) | [30 个预测](Assignment1/results/test_predictions.csv) |
| Assignment 2 | [说明文档](Assignment2/说明文档.md) | [报告](Assignment2/Assignment2_Report.pdf) | [solution.py](Assignment2/solution.py) | [23 个预测](Assignment2/results/test_predictions.csv) |

Assignment 1 的主要清洗规则是完整样本上的一次性逐特征 1.5 IQR，保留 112 个训练样本。保留两个较轻异常样本的 114 行辅助方案给出相同的 30 个预测。Assignment 2 采用投影类均值中点阈值，方向为 `Sw^(-1)(m_positive - m_negative)`，分数非负预测 +1。具体假设和全精度参数已写入各作业说明及 `results/results.json`。

## 运行

在本目录执行（推荐 Python 3.10+）：

```powershell
python -m pip install -r requirements.txt
python Assignment1/solution.py
python Assignment2/solution.py
python verify_results.py
python build_reports.py
```

每份 `solution.py` 可独立运行；运行时会重建本作业 `results` 中的派生结果。原始 Excel、MAT、题目 PDF 不会被修改。报告生成脚本从保存的 JSON 读取数值。

`verification.json` 保存独立数值核对结果：GaussianNB (`var_smoothing=0`) 与等先验 SVD LDA 的参数、分数、预测均一致，清洗行号及测试顺序一致，原始数据 SHA-256 未变。测试集未提供真实标签，因此没有计算或声称测试准确率。

本次实际验证使用 numpy 1.26.4、pandas 1.5.3、scipy 1.10.0、scikit-learn 1.2.1。核心实现不依赖随机初始化。

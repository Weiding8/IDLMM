import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# 读取CSV文件
df = pd.read_csv('../data/updated_file1.csv')

# 获取真实值和预测概率
# y_true = df.iloc[:, 3].to_numpy()  # 第四列真实值
# y_scores = df.iloc[:, 8].to_numpy()  # 第七列预测概率
y_true = df['response'].to_numpy()  # 第四列真实值
y_scores = df['predicted'].to_numpy()  # 第七列预测概率best_id2_a

# 计算精确度和召回率
precision, recall, _ = precision_recall_curve(y_true, y_scores)
average_precision = average_precision_score(y_true, y_scores)

# 绘制图形
plt.figure(figsize=(8.5, 6), dpi=1200)
plt.step(recall, precision, where='post', color='red', label='Classifier', linewidth=2.5)
# plt.fill_between(recall, precision, alpha=0.2, color='red')
plt.xlim(-0.02, 1.02)
plt.ylim(0.68, 1.02)
plt.xlabel('Recall',fontsize=26)
plt.ylabel('Precision',fontsize=26)
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.title('TCGA Classifier Precision-Recall',fontsize=26)
plt.text(0.5, 0.85, f'AUCPR: {average_precision:.4f}', horizontalalignment='center', verticalalignment='center',fontsize=24)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
plt.tight_layout()
# plt.legend()
# plt.show()
plt.savefig('../fig/auc_id2.png', bbox_inches='tight')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件
from matplotlib.transforms import ScaledTranslation

file_path = '../data/drug_pearson_drugcell_transEncoder1408_JZ.csv'  # 替换为您的 CSV 文件路径

data = pd.read_csv(file_path)

# 2. 提取药物名称和 Pearson 相关性
drugs = data['drug'].tolist()  # 假设这一列的名称为 'drug'
pearson_corr = data['pearson_correlation'].tolist()  # 假设这一列的名称为 'pearson_correlation'
# 2. 假设 CSV 文件中包含 'drug' 和 'pearson_corr' 列
# drugs = data['drug'].values  # 药物名称
pearson_corr = data['pearson_correlation'].values  # 皮尔逊相关系数

# 3. 分隔高置信度和低置信度药物
high_confidence = pearson_corr > 0.5
low_confidence = pearson_corr <= 0.5
# 计算相关性 > 0.5 的药物数量
count_highly_relevant = sum(c > 0.5 for c in pearson_corr)
count_total = len(drugs)

# 3. 创建图形
fig, ax = plt.subplots(figsize=(12, 10),dpi=350)#
fig.patch.set_alpha(0.0)

bars = ax.bar(np.arange(count_total), pearson_corr, width=1,color='#CA2323')

# 根据相关性设定条形颜色
for bar, value in zip(bars, pearson_corr):
    if value <= 0.5:
        bar.set_color('#1A188C')  # 相关性较低的部分

# 5. 填充绘图区域
# ax.fill_betweenx(range(len(drugs)), 0, 1, where=high_confidence, color='red', alpha=0.6, label='High confidence drugs (rho > 0.5)')
# ax.fill_betweenx(range(len(drugs)), 0, 1, where=low_confidence, color='blue', alpha=0.4)

# 6. 设置坐标轴
ax.set_xlim(0, count_total)
ax.set_ylim(-0.21, 1.5)
# ax.set_xlim(0, 250)
ax.set_xticks([])
plt.yticks([0.0,0.4,0.8],[0.0,0.4,0.8],fontsize=30)
# ax.set_xlabel('Drugs',fontsize=30)
ax.set_ylabel('Pearson Correlation',fontsize=36,labelpad=2)
plt.plot([0.0, 684], [0.0, 0.0], color='black', linewidth=2)
plt.axvline(x=0.0, ymin=-0.21, ymax=0.75, color='black', linewidth=3)

plt.text(90, -0.08, r'Drugs',
         # family='Times New Roman',  # 标注文本字体
         fontsize=38,  # 文本大小
         color='black'  # 文本颜色
         )

plt.text(13, -0.19, 'High confidence drugs (ρ>0.5)',
         family='Times New Roman',  # 标注文本字体
         fontsize=36,  # 文本大小
         # fontweight='bold',  # 字体粗细
         color='black'  # 文本颜色
         )
plt.scatter(8, -0.17, color='#CA2323', s=500, edgecolor='#CA2323', zorder=5,marker='s')



ax_inset = fig.add_axes([0.52, 0.69, 0.5, 0.3])  # 创建小图的区域
ax_inset.bar(drugs[:10],pearson_corr[:10], color='#CA2323')
ax_inset.set_ylim(0.5, 0.95)
ax_inset.set_ylabel('Pearson Corr',fontsize=32)
plt.yticks([0.5,0.7,0.9],[0.5,0.7,0.9],fontsize=22)
ax_inset.set_xticklabels(drugs[:10], ha='right', fontsize=28, rotation=50)
dx = 12 / 72.
offset = ScaledTranslation(dx, 0, fig.dpi_scale_trans)
for label in ax_inset.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

# 隐藏顶部和右侧边框
for spine in ['top', 'right']:
    ax_inset.spines[spine].set_visible(False)

# 隐藏边框
for spine in ax.spines.values():
    spine.set_visible(False)


# 8. 显示图例
ax.legend(loc='upper right')


# 9. 显示主要图形
plt.tight_layout()
# plt.show()
plt.savefig('../fig/drugplot_bio.png', bbox_inches='tight',transparent=True)


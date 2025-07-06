


import numpy as np
import matplotlib.pyplot as plt

categories = ['IDLMM', 'DrugGene', 'DrugCell', 'Ridge', 'EN', 'Lasso']
My_Model = [0.82886, 0.80177, 0.79033, 0.763, 0.747, 0.683]
# 2. 设置条形的位置
bar_width = 0.5  # 柱子的宽度
x = np.arange(len(categories))  # 类别的位置

# 3. 创建图形
fig, ax = plt.subplots(figsize=(13, 9), dpi=350)

# 4. 绘制多系列条形图
bars0 = ax.bar(x, My_Model, width=bar_width, label='My_Model', color='#1A188C')

# 6. 设置坐标轴和图例
# ax.set_xlabel('Model',fontsize='40', labelpad=-28)
ax.set_ylabel('Pearson correlation', fontsize=40, labelpad=2)
# ax.set_title('图表标题')
ax.set_xticks(x)  # 设置 X 轴的刻度
ax.set_xticklabels(categories, fontsize=40) # 设置刻度标签
plt.xticks(rotation=30,ha='center')
ax.tick_params(axis='y', labelsize=34)
ax.set_ylim(0.65,0.85)  # 设置 Y 轴的范围
ax.set_yticks(np.arange(0.7, 0.8, 0.05))
# plt.yticks(fontsize=34)

ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
# 7. 显示图形
plt.tight_layout()
# plt.show()
plt.savefig('../fig/Pearson2.png', bbox_inches='tight')
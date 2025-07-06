
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 从CSV文件读取数据
# 请确保将'file_path.csv'替换为你的CSV文件路径
data = pd.read_csv('../data/mymodel.csv')
data = data[data['cell_line'] == 'NCIH23']

# 选择需要的列并找到pred_auc的最小10个值
top_10 = data.nsmallest(10, 'pred_auc')

# 获取细胞系和预测的AUC值
drugs = top_10['drug'].tolist()
predicted_auc = top_10['pred_auc'].tolist()

# 设置图形尺寸
plt.figure(figsize=(7, 9), dpi=350)

# 创建柱状图
bars = plt.bar(drugs, predicted_auc, color='#3274A1',width=0.6)

# 添加标签和标题
plt.ylabel('Predicted AUC', fontsize=32)
plt.xlabel('Drug', fontsize=32, labelpad=-14)
# plt.title('PD0325901', fontsize=35, loc='center', pad=20)

plt.ylim(0, 1.08)
plt.yticks(np.arange(0.1, 0.91, 0.2), fontsize=26)

# 设置坐标轴线宽度
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)

drugs = [drug.replace('Sepantronium-bromide', 'YM155') for drug in drugs]
# 调整x轴标签的旋转角度
plt.xticks(rotation=60, fontsize=28,ha='right')
# 移动标签位置（向右偏移)
ax.set_xticks(np.arange(len(drugs)) + 0.4)  # 0.2是一个示例偏移量，调整此值以获得所需的效果
ax.set_xticklabels(drugs, fontsize=28)
ax.tick_params(axis='x', pad=-1)

# # 添加外框
# rect_data = plt.Rectangle((-0.5, 0.3), len(cell_lines), 0.41, linewidth=2, edgecolor='black', facecolor='none')
# plt.gca().add_patch(rect_data)

# 添加标题框
rect_title = plt.Rectangle((-1, 0.93), len(drugs)+1, 2, linewidth=3, edgecolor='black', facecolor='#D0D7DB')
plt.gca().add_patch(rect_title)
plt.text(2.8, 0.97, r'NCIH23',
         # family='Times New Roman',  # 标注文本字体
         fontsize=34,  # 文本大小
         # fontweight='bold',  # 字体粗细
         color='black'  # 文本颜色
         )
# 显示图形
plt.tight_layout()

# plt.show()
plt.savefig('../fig/NCIH23_bio.png', bbox_inches='tight')
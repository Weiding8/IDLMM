import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# 读取CSV文件（替换为你的文件路径）
file_path = '../data/updated_file1.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 确保第8列和第9列正确
# 假设列名为'id2'（第8列）和'response'（第9列）
precision_column = 'predicted'  # 第8列的列名
type_column = 'response'

# 分组数据
group_reactive = data[data[type_column] == 1][precision_column]
group_non_reactive = data[data[type_column] == 0][precision_column]

# 计算p值
stat, p_value = ttest_ind(group_reactive, group_non_reactive, nan_policy='omit')

# 计算均值并根据均值排序
mean_values = data.groupby(type_column)[precision_column].mean().sort_values(ascending=False)
sorted_types = mean_values.index

# 设置绘图风格
plt.figure(figsize=(8.5, 6),dpi=350)
violin_plot = sns.violinplot(x=type_column, y=precision_column, data=data, order=sorted_types, palette="Blues", inner=None)

# 设置标题和标签
plt.xlabel('')
plt.ylabel('Probability', fontsize=26)
plt.ylim(0, 1)  # 设置y轴范围
plt.xticks(ticks=range(len(sorted_types)), labels=['With response', 'No response'], fontsize=26)  # 自定义x轴标签
plt.yticks(fontsize=24)  # 自定义x轴标签

# 在图中标注p值
plt.text(0.45, 0.25, f'p = {p_value:.3e}', ha='center', fontsize=24, transform=plt.gca().transAxes)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
plt.tight_layout()
# plt.show()
plt.savefig('../fig/response_id2.png', bbox_inches='tight')
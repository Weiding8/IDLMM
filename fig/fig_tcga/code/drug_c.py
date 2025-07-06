


import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件（替换为你的文件路径）
file_path = '../data/updated_file1.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 筛选出药物名称为 'Carboplatin' 和 'Cisplatin' 的行
filtered_data = data[data['drug'].isin(['Carboplatin', 'Cisplatin'])]

# # 统计每个项目的样本数量
# counts = filtered_data['project'].value_counts()
#
# # 筛选出数量大于1的项目
# valid_projects = counts[counts > 1].index
#
# # 筛选数据，只保留有效项目
# filtered_data = filtered_data[filtered_data['project'].isin(valid_projects)]

# 按 'project' 分组，并计算每个组的 'id2' 的均值
summary = filtered_data.groupby('project')['predicted'].mean().sort_values(ascending=False)

# 提取前10个项目名称和对应的概率
top_projects = summary.head(10)
projects = top_projects.index.str.replace('TCGA-', '', regex=False)  # 去掉'TCGA-'前缀
probabilities = top_projects.values
print(probabilities)

# 绘制柱状图
plt.figure(figsize=(8.5, 6), dpi=350)
plt.bar(projects, probabilities, color='#0D8AB1', width=0.5)
plt.title('Predicted Probability of Response for\n Carboplatin and Cisplatin', fontsize=24)
plt.ylabel('Probability', fontsize=26)
plt.ylim(0.3, 1)  # 设置y轴范围为0到1
plt.xticks(rotation=45, ha='center', fontsize=26)  # 旋转x轴标签
plt.yticks(fontsize=24)

# 设置坐标轴线宽度
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)

plt.tight_layout()
# plt.show()
plt.savefig('../fig/F_id2.png', bbox_inches='tight')
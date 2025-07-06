import matplotlib.pyplot as plt

# 假设你的第一个文件存储为 "data1.txt"
file_path_1 = "../data/Paclitaxel_rlipp_linear.txt"
# 假设你的第二个文件存储为 "data2.txt"
file_path_2 = "../data/descendants.txt"

# 读取第一个文件的GO编号和rlipp值
go_numbers = []
rlipp_values = []

with open(file_path_1, 'r') as f:
    for i in range(195):
        line = f.readline()
        parts = line.strip().split('\t')  # 按制表符分割
        if len(parts) == 2:
            go_numbers.append(parts[0])  # GO编号
            rlipp_values.append(float(parts[1]))  # rlipp值

# 读取第二个文件的GO编号
go_numbers_in_file_2 = set()  # 用集合去重
with open(file_path_2, 'r') as f:
    content = f.read()
    go_numbers_in_file_2.update(content.split(","))  # 按逗号分割

# 准备条形图数据，设置颜色
colors = ['red' if go in go_numbers_in_file_2 else '#1B248F' for go in go_numbers]

# 绘制条形图
plt.figure(figsize=(10, 12), dpi=350)
plt.bar(go_numbers, rlipp_values, color=colors,width=1.5)

plt.xlabel(u'Top 10% of subsystems ', fontsize=35, labelpad=15.5)
plt.ylabel(u'Importance for paclitaxel response\n(RLIPP score) ', fontsize=35, labelpad=15.5)


plt.xticks([])
plt.ylim((0.5, 4))
plt.yticks([1,2,3], [1,2,3], fontsize=35)
plt.tick_params(width=2)

ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)

plt.tight_layout()

# plt.show()

plt.savefig('../fig/barplot.png', bbox_inches='tight')



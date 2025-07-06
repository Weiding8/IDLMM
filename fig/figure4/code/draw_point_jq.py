import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


cell_mutation = np.genfromtxt("../data/mut.txt", delimiter=',')
cell_exp = np.genfromtxt("../data/exp.txt", delimiter=',')

min_max_scaler = MinMaxScaler()
exp_result = min_max_scaler.fit_transform(cell_exp)
cell_exp = np.around(exp_result, 5)
cell_features = cell_exp#+cell_mutation

cell_features = np.concatenate((cell_features[:, 287:288], cell_features[:, 801:802]), axis=1)

index_list = []
index = open("../data/jq1_index.txt")
for line in index:
    line = line.rstrip().split()
    index_list.append(int(line[0]))
index.close()

mu_count = 0
cell_list = []
for x in range(len(cell_features)):
    if (cell_features[x][0] > 0.45 or cell_features[x][1] > 0.45) and x in index_list:
        mu_count += 1
        cell_list.append(x)

print("cell_list: ", cell_list)
print("cell_list_len: ", len(cell_list))
print("mu_count: ", mu_count)

pc1 = []
pc2 = []
pc1_file = open("../data/GO_0008150_PC1.txt")
for line in pc1_file:
    line = line.rstrip().split()
    pc1.append(float(line[0]))
pc1_file.close()

pc2_file = open("../data/GO_0008150_PC2.txt")
for line in pc2_file:
    line = line.rstrip().split()
    pc2.append(float(line[0]))
pc2_file.close()

pc1_exist = []
pc1_noexist = []
for x in range(len(pc1)):
    if x in cell_list:
        pc1_exist.append(pc1[x])
    else:
        pc1_noexist.append(pc1[x])

pc2_exist = []
pc2_noexist = []
for x in range(len(pc2)):
    if x in cell_list:
        pc2_exist.append(pc2[x])
    else:
        pc2_noexist.append(pc2[x])



plt.figure(figsize=(15, 20), dpi=100)
# plt.plot([-0.8, 1.0], [-0.8, 1.0], color='black', linewidth=2, label="best line")

area1 = [400] * len(pc1_exist)
area2 = [100] * len(pc1_noexist)

# label = ['red'] * 58 + ['lime'] * 626

plt.scatter(pc1_noexist, pc2_noexist, label="train data", s=area2, c="#808080")
plt.scatter(pc1_exist, pc2_exist, label="train data", s=area1, c="red", cmap='viridis')

plt.xlabel(u'Genotype embedding(PC1) ', fontsize=55, labelpad=18.5)
plt.ylabel(u'Genotype embedding(PC2) ', fontsize=55, labelpad=28.5)

plt.xticks([-1.5, 3], [-1.5, 3], fontsize=45)
plt.yticks([-1.5, 2], [-1.5, 2], fontsize=45)

plt.xticks([])  # 去x坐标刻度
plt.yticks([])  # 去y坐标刻度


ax=plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)

plt.text(-1.3, 1.7, r'n = {}'.format(len(pc1_exist)),
         family='Times New Roman',  # 标注文本字体
         fontsize=63,  # 文本大小
         # fontweight='bold',  # 字体粗细
         color='black'  # 文本颜色
         )
plt.text(0.5, 1.5, 'BRAF or EGFR\n     expression',
         family='Times New Roman',  # 标注文本字体
         fontsize=63,  # 文本大小
         # fontweight='bold',  # 字体粗细
         color='black'  # 文本颜色
         )



plt.savefig('../img/jq_point.png', bbox_inches='tight')
# plt.show()



import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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


pre_list = []
pre = open("../data/JQ1_JZ.predict")
for line in pre:
    line = line.rstrip().split()
    pre_list.append(float(line[0]))
pre.close()

pre_np = []
for i in range(len(pre_list)):
    temp = []
    temp.append(pre_list[i])
    pre_np.append(temp)

min_max_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
pre_result = min_max_scaler.fit_transform(np.array(pre_np))
pre_result = np.around(pre_result, 5)
# print(pre_result)
pre_list = list(pre_result)


index_list = []
index = open("../data/JQ1_index.txt")
for line in index:
    line = line.rstrip().split()
    index_list.append(int(line[0]))
index.close()

print("pre_list: ", pre_list)
print("index_list: ", index_list)

pc1_exist = []
pc1_noexist = []
for x in range(len(pc1)):
    for y in index_list:
        index_list.index(y)
    if x in index_list:
        pc1_exist.append(pc1[x])
    else:
        pc1_noexist.append(pc1[x])

pc2_exist = []
pc2_noexist = []
for x in range(len(pc2)):
    if x in index_list:
        pc2_exist.append(pc2[x])
    else:
        pc2_noexist.append(pc2[x])

plt.figure(figsize=(15, 20), dpi=100)
area1 = [400] * len(pc1_exist)
area2 = [100] * len(pc1_noexist)

# label = ['red'] * 58 + ['lime'] * 626

plt.scatter(pc1_noexist, pc2_noexist, label="train data", s=area2, c="#808080")
scatter = plt.scatter(pc1_exist, pc2_exist, label="train data", s=area1, c=pre_list,cmap='coolwarm')#,

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

plt.text(-1.3, 1.7, r'n={}'.format(len(pc1_exist)),
         family='Times New Roman',  # 标注文本字体
         fontsize=63,  # 文本大小
         # fontweight='bold',  # 字体粗细
         color='black'  # 文本颜色
         )

# plt.savefig('../img/jq_pre.png', bbox_inches='tight')
plt.show()


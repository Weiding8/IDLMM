import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


pc1 = []
pc2 = []
pc1_file = open("../data/Pacli_JZ_pc1.txt")
for line in pc1_file:
    line = line.rstrip().split()
    pc1.append(float(line[0]))
pc1_file.close()

pc2_file = open("../data/Pacli_JZ_pc2.txt")
for line in pc2_file:
    line = line.rstrip().split()
    pc2.append(float(line[0]))
pc2_file.close()

print("pc1: ", pc1)
print("pc2: ", pc2)
print(len(pc1))
print(len(pc2))


pre_list = []
pre = open("../data/jjh_transEn1408_JZ_Paclitaxel.predict")
for line in pre:
    line = line.rstrip().split()
    pre_list.append(float(line[0]))
pre.close()

pre_np = []
for i in range(len(pre_list)):
    temp = []
    temp.append(pre_list[i])
    pre_np.append(temp)

print(pre_np)
# min_max_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
# pre_result = min_max_scaler.fit_transform(np.array(pre_np))
# pre_result = np.around(pre_result, 5)
# # print(pre_result)
# pre_list = list(pre_result)

# for i in range(len(pre_list)):
#     if 0.75 < pre_list[i] < 0.8:
#         pre_list[i] *= 0.5
#     if pre_list[i] < 0.67:
#         pre_list[i] *= 0.6
#     elif pre_list[i] < 0.9:
#         pre_list[i] *= 1.1

plt.figure(figsize=(10, 12), dpi=350)
area1 = [250] * len(pc1)
plt.scatter(pc1, pc2, label="train data", s=area1, c=pre_list,cmap='coolwarm')#,cmap='gist_heat'

# cb1 = plt.colorbar(fraction=0.05, pad=0.03, orientation="horizontal")
# cb1 = plt.colorbar(fraction=0.03, pad=0.03)
plt.xlabel(u'Regulation of RNA metabolic\nembedding (PC1) ', fontsize=35, labelpad=10.5)
plt.ylabel(u'Regulation of RNA metabolic\nembedding (PC2) ', fontsize=35, labelpad=10.5)

plt.xticks([-1.5, 2], [-1.5, 2], fontsize=30)
plt.yticks([-0.5, 1], [-0.5, 1], fontsize=30)

plt.xticks([])
plt.yticks([])

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)

plt.text(-1.4, 0.9, r'n={}'.format(len(pc1)),
         family='Times New Roman',  # 标注文本字体
         fontsize=42,  # 文本大小
         # fontweight='bold',  # 字体粗细
         color='black'  # 文本颜色
         )

# plt.show()

plt.savefig('../fig/drug_Paclitaxel_linear.png', bbox_inches='tight')
# plt.close()



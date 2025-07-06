
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


pc1 = []
pc2 = []
pc1_file = open("../data/drugJZ_PC1.txt")
for line in pc1_file:
    line = line.rstrip().split()
    pc1.append(float(line[0]))
pc1_file.close()

pc2_file = open("../data/drugJZ_PC2.txt")
for line in pc2_file:
    line = line.rstrip().split()
    pc2.append(float(line[0]))
pc2_file.close()

print("pc1: ", pc1)
print("pc2: ", pc2)
print(len(pc1))
print(len(pc2))


target_BRAF = [52, 147, 165]#all
target_EGFR = [19, 70, 75, 131]
target_BRD4 = [21, 142,159]#159
target_PARP = [124, 130, 157, 177,195]#195
target_MEK = [138, 168, 186,153]#



target_class = target_BRAF + target_EGFR + target_BRD4 + target_PARP + target_MEK
target_class = sorted(target_class)
print(target_class)

pc1_exist_BRAF = []
pc1_exist_EGFR = []
pc1_exist_BRD4 = []
pc1_exist_PARP = []

pc1_exist_MEK = []


pc1_noexist = []
for x in range(len(pc1)):
    if x in target_BRAF:
        pc1_exist_BRAF.append(pc1[x])
    elif x in target_EGFR:
        pc1_exist_EGFR.append(pc1[x])
    elif x in target_BRD4:
        pc1_exist_BRD4.append(pc1[x])
    elif x in target_PARP:
        pc1_exist_PARP.append(pc1[x])
    elif x in target_MEK:
        pc1_exist_MEK.append(pc1[x])
    else:
        pc1_noexist.append(pc1[x])

pc2_exist_BRAF = []
pc2_exist_EGFR = []
pc2_exist_BRD4 = []
pc2_exist_PARP = []

pc2_exist_MEK = []

pc2_noexist = []
for x in range(len(pc2)):
    if x in target_BRAF:
        pc2_exist_BRAF.append(pc2[x])
    elif x in target_EGFR:
        pc2_exist_EGFR.append(pc2[x])
    elif x in target_BRD4:
        pc2_exist_BRD4.append(pc2[x])
    elif x in target_PARP:
        pc2_exist_PARP.append(pc2[x])
    elif x in target_MEK:
        pc2_exist_MEK.append(pc2[x])
    else:
        pc2_noexist.append(pc2[x])

plt.figure(figsize=(15, 20), dpi=100)

area1 = [150] * (211 - len(target_class))
area2_BRAF = [400] * len(target_BRAF)
area2_EGFR = [400] * len(target_EGFR)
area2_BRD4 = [400] * len(target_BRD4)
area2_PARP = [400] * len(target_PARP)
area2_MEK = [400] * len(target_MEK)

target_color1 = ["#CC4D4D"] * len(target_BRAF)
target_color2 = ["#4349A4"] * len(target_EGFR)
target_color3 = ["#40959B"] * len(target_BRD4)
target_color4 = ["green"] * len(target_PARP)
target_color5 = ["#DE8A0A"] * len(target_MEK)


plt.scatter(pc1_noexist, pc2_noexist, s=area1, c="silver")

plt.scatter(pc1_exist_EGFR, pc2_exist_EGFR, label="EGFR", s=area2_EGFR, c=target_color2)
plt.scatter(pc1_exist_BRD4, pc2_exist_BRD4, label="BRD4", s=area2_BRD4, c=target_color3)
plt.scatter(pc1_exist_PARP, pc2_exist_PARP, label="PARP", s=area2_PARP, c=target_color4)
plt.scatter(pc1_exist_MEK, pc2_exist_MEK, label="MEK", s=area2_MEK, c=target_color5)
plt.scatter(pc1_exist_BRAF, pc2_exist_BRAF, label="BRAF", s=area2_BRAF, c=target_color1)


plt.xlabel(u'Drug structure\n embedding(PC1) ', fontsize=55, labelpad=18.5)
plt.ylabel(u'Drug structure embedding(PC2) ', fontsize=55, labelpad=28.5)

plt.xlim(-3, 4)
plt.ylim(-2, 4)
plt.xticks([])
plt.yticks([])

legend = plt.legend(title='Target class',title_fontsize='40', fontsize='38', loc='upper right', markerscale=1.3,
                    borderpad=0.5, labelspacing=0.5,handletextpad = 0.3, frameon=False)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)


# plt.show()

plt.savefig('../img/drug_pc_JZ.png', bbox_inches='tight')
# plt.close()


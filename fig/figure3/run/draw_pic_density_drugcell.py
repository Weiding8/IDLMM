import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


con_list = []
mu_list = []
con_data = open("../data/drug_pearson_drugcell_transEncoder1408_JZ.txt")
for line in con_data:
    line = line.rstrip().split()
    con_list.append(float(line[0]))

con_data.close()
# *******
mu_data = open("../data/drug_pearson_drugcell.txt")
# mu_data = open("../drug_res/pearson_mu.txt")
for line in mu_data:
    line = line.rstrip().split()
    mu_list.append(float(line[0]))

mu_data.close()

# print(con_list)
# print(mu_list)

con_than_mu = 0
for i in range(len(mu_list)):
    if con_list[i] > mu_list[i]:
        con_than_mu += 1
# print(con_than_mu)

from scipy.stats import gaussian_kde

mu_list = np.array(mu_list)
con_list = np.array(con_list)

xy = np.vstack([mu_list, con_list])  #  将两个维度的数据叠加
z = gaussian_kde(xy)(xy)

idx = z.argsort()
mu_list, con_list, z = mu_list[idx], con_list[idx], z[idx]
import matplotlib.ticker as ticker

plt.figure(figsize=(10.1, 9), dpi=350)
# fig, ax=plt.subplots(figsize=(10, 10), dpi=100)
plt.plot([-0.3, 1], [-0.3,1], color='black', linewidth=2, label="best line",linestyle='--')
# plt.scatter(mu_list, con_list, c=z, label="train data", s=100, cmap='RdYlBu_r')#RdYlGn_r\Spectral_r
plt.scatter(mu_list, con_list, c='black', label="train data", s=100)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=30, width=2)
#
# tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
# cb.locator = tick_locator
# cb.update_ticks()

plt.xlabel(u'Pearson correlation (DrugCell) ', fontsize=40)
plt.ylabel(u'Pearson correlation (IDLMM) ', fontsize=40)

# plt.xticks([0.0,0.5,1.0], [0.0,0.5,1.0], fontsize=35)
# plt.yticks([0.0,0.5,1.0], [0.0,0.5,1.0], fontsize=35)
plt.xticks([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], fontsize=34)
plt.yticks([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], fontsize=34)
plt.tick_params(width=2)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
plt.tight_layout()

# plt.show()

plt.savefig('../fig/desity_drugcell.png', bbox_inches='tight')


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 读取数据
# cell_lines = pd.read_csv('../data/cell_tissue.txt', delim_whitespace=True, header=None)
# cell_lines.columns = ['cell_line', 'cell_name', 'tissue_type']
#
# pearson_data = pd.read_csv('../data/cell_pearson_drugcell_transEncoder1408_JZ.csv')
#
# # 合并数据
# merged_data = pd.merge(pearson_data, cell_lines, how='left', left_on='cell_line', right_on='cell_name')
#
# merged_data['tissue_type'] = merged_data['tissue_type'].replace({
#     'central_nervous_system': 'CNS',
#     'soft_tissue': 'ST',
#     'haematopoietic_and_lymphoid_tissue': 'HLT',
#     'autonomic_ganglia': 'AG',
#     'kidney':'Kid',
#     'pancreas':'Panc',
#     'large_intestine': 'LI',
#     'urinary_tract': 'UT',
#     'oesophagus':'OE',
#     'stomach':'Sto',
#     'upper_aerodigestive_tract': 'UAT'
# })
#
# # 计算每个组织类型的皮尔逊相关系数的中值
# median_values = merged_data.groupby('tissue_type')['pearson_correlation'].median().reset_index()
#
# # 计算每个组织类型的细胞系数量
# tissue_counts = merged_data['tissue_type'].value_counts().reset_index()
# tissue_counts.columns = ['tissue_type', 'cell_line_count']
#
# # 合并中值和细胞系数量
# tissue_stats = median_values.merge(tissue_counts, on='tissue_type')
#
# # 选择细胞系数量最多的前十个组织类型
#
# top_tissues = tissue_stats.nlargest(15, 'cell_line_count')
# ordered_tissues = top_tissues.sort_values('pearson_correlation', ascending=False)['tissue_type'].tolist()
#
# # 画小提琴图，使用有序的组织类型
# plt.figure(figsize=(12, 9), dpi=350)
# sns.violinplot(x='tissue_type', y='pearson_correlation', data=merged_data,
#                 order=ordered_tissues)
# # sns.violinplot(x='tissue_type', y='pearson_correlation', data=merged_data,
# #                 order=top_tissues)
# plt.axhline(y=0.8, color='#446CA0', linestyle='--', linewidth=2)
#
# plt.xlabel('Tissue Type',fontsize=36,labelpad=6)
# plt.ylabel('Pearson Correlation',fontsize=36,labelpad=6)
# plt.xticks(rotation=70,fontsize=34,ha='center')#,
#
# plt.yticks(np.arange(0, 1.3, 0.4), fontsize=30)
#
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(3)
# ax.spines['left'].set_linewidth(3)
# ax.spines['right'].set_linewidth(0)
# ax.spines['top'].set_linewidth(0)
#
#
# plt.tight_layout()
# # plt.savefig('../fig/xiaotiqin_big.png', bbox_inches='tight')
# plt.show()
#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
cell_lines = pd.read_csv('../data/cell_tissue.txt', delim_whitespace=True, header=None)
cell_lines.columns = ['cell_line', 'cell_name', 'tissue_type']

pearson_data = pd.read_csv('../data/cell_pearson_drugcell_transEncoder1408_JZ.csv')

# 合并数据
merged_data = pd.merge(pearson_data, cell_lines, how='left', left_on='cell_line', right_on='cell_name')

merged_data['tissue_type'] = merged_data['tissue_type'].replace({
    'central_nervous_system': 'CNS',
    'soft_tissue': 'ST',
    'haematopoietic_and_lymphoid_tissue': 'HLT',
    'autonomic_ganglia': 'AG',
    'kidney':'Kid',
    'pancreas':'Panc',
    'large_intestine': 'LI',
    'urinary_tract': 'UT',
    'oesophagus':'OE',
    'stomach':'Sto',
    'upper_aerodigestive_tract': 'UAT'
})

# 计算每个组织类型的皮尔逊相关系数的中值
median_values = merged_data.groupby('tissue_type')['pearson_correlation'].median().reset_index()

# 计算每个组织类型的细胞系数量
tissue_counts = merged_data['tissue_type'].value_counts().reset_index()
tissue_counts.columns = ['tissue_type', 'cell_line_count']

# 合并中值和细胞系数量
tissue_stats = median_values.merge(tissue_counts, on='tissue_type')

# 选择细胞系数量最多的前十个组织类型

top_tissues = tissue_stats.nlargest(15, 'cell_line_count')
ordered_tissues = top_tissues.sort_values('pearson_correlation', ascending=False)['tissue_type'].tolist()

# 画小提琴图，使用有序的组织类型
plt.figure(figsize=(16, 9), dpi=350)#
sns.violinplot(x='tissue_type', y='pearson_correlation', data=merged_data,
                order=ordered_tissues)
# sns.violinplot(x='tissue_type', y='pearson_correlation', data=merged_data,
#                 order=top_tissues)
plt.axhline(y=0.8, color='#446CA0', linestyle='--', linewidth=2)

plt.xlabel('')
# plt.xlabel('Tissue Type',fontsize=36,labelpad=6)
plt.ylabel('Pearson Correlation',fontsize=40,labelpad=6)
plt.xticks(rotation=45,fontsize=40,ha='right')#,

plt.yticks(np.arange(0, 1.3, 0.4), fontsize=34)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)


plt.tight_layout()
plt.savefig('../fig/xiaotiqin_big.png', bbox_inches='tight')
# plt.show()


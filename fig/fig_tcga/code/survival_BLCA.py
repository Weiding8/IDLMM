import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# Loading predictions
predictions = pd.read_csv("../data/updated_file1.csv")
predictions = predictions[predictions['project'] == 'TCGA-BLCA']
predictions.rename(columns={predictions.columns[1]: "bcr_patient_barcode"}, inplace=True)

# Loading TCGA clinical data
clin = pd.read_csv("../data/TCGA_clinical_metadata.csv", index_col=0)

# Data preparation
df = pd.merge(predictions, clin, on="bcr_patient_barcode")

# Clean and convert vital_status and OS
df['OS'] = pd.to_numeric(df['OS'], errors='coerce')
df['vital_status'] = df['vital_status'].replace({'Alive': 0, 'Dead': 1})  # 0为生存, 1为死亡
df = df.dropna(subset=['OS', 'vital_status'])

# Stratifying patients based on median value of probability of response
median_pred = df['predicted'].median()#best_id2_a   id2_300_a
group1 = df[df['predicted'] > median_pred]
group2 = df[df['predicted'] < median_pred]

# Create survival data
group1.loc[:, 'group'] = 'High Probability'
group2.loc[:, 'group'] = 'Low Probability'
surv = pd.concat([group1, group2])

# Create Kaplan-Meier fit
kmf = KaplanMeierFitter()

# Fit and plot for both groups

color_map = {'High Probability': '#000CB8', 'Low Probability': 'red'}
# Fit and plot for both groups
plt.figure(figsize=(8.5, 6), dpi=1200)
for name, grouped_df in surv.groupby('group'):
    kmf.fit(grouped_df['OS'], event_observed=grouped_df['vital_status'], label=name)
    color = color_map[name]
    kmf.plot_survival_function(ci_show=False, color=color, linewidth=2.5)

# Log-rank test
results = logrank_test(group1['OS'], group2['OS'],
                       event_observed_A=group1['vital_status'], event_observed_B=group2['vital_status'])
p_value = results.p_value
print(f'Log-rank test p-value: {p_value}')

ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Add titles and labels
plt.title('Survival Analysis for BLCA in TCGA', fontsize=26)
plt.xlabel('Days',fontsize=26)
plt.ylabel('Survival Probability',fontsize=26)
plt.xlim(0, 1500)
plt.axhline(y=0.5, color='gray', linestyle='--')
plt.text(1000, 0.3, f'p =  {p_value:.4f}', horizontalalignment='center', verticalalignment='center',fontsize=24)
# plt.legend()
# plt.grid()
plt.legend(loc='best', fontsize=20, frameon=True)
plt.tight_layout()

# Show the plot
# plt.show()
plt.savefig('../fig/BLCA_id2.png', bbox_inches='tight')
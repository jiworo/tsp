import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from colors import Color

plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = 'Linux Libertine O'
plt.rcParams['figure.dpi'] = 600

# get data
with open('groups.csv', 'r') as file:
    IAT_results = pd.read_csv(file)
IAT_results['id'] = IAT_results['id'].str.lower()

with open('main-questionnaire/values.csv', 'r') as file:
    q = pd.read_csv(file)

# preprocess questionnaire data
q = q.drop([
    'StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress',
    'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId',
    'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
    'ExternalReference', 'LocationLatitude', 'LocationLongitude',
    'DistributionChannel', 'UserLanguage'
], axis=1).drop(range(6)).reset_index(drop=True)
for column in q.columns[1:]:
    q[column] = pd.to_numeric(q[column], downcast='integer', errors='coerce')
    q[column] = q[column] - 3
q['Q44'] = q['Q44'].str.lower()

# for questions that have low response score associated with high awareness or responsibility we invert the score:
for col in ['Q7', 'Q11', 'Q18', 'Q41', 'Q43', 'Q37', 'Q38']:
    q[col] = q[col].apply(lambda x: -x)

# for questions that have extreme response score associated with high awareness or responsibility
for col in ['Q13', 'Q15', 'Q16']:
    q[col] = q[col].apply(lambda x: 2 * abs(x) - 2)

# increase weight of very important questions
for col in ['Q19', 'Q20']:
    q[col] = q[col].apply(lambda x: 2 * x)

# rescale: score between -1 and 1
for column in q.columns[1:]:
    q[column] = q[column] / 2

# create awareness and responsibility score
q['Awareness'] = np.mean([q[f'Q{i}'] for i in [5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 18, 19, 20]], axis=0)
q['Responsibility'] = np.mean([q[f'Q{i}'] for i in [22, 23, 25, 27, 33, 34, 35, 36, 41, 43, 37, 38]], axis=0)


def extract_group_ids(group_name):
    return IAT_results.loc[np.where(IAT_results['group'] == group_name)]['id'].reset_index(drop=True)


def extract_group_scores(score_name, ids):
    return [q.loc[np.where(q['Q44'] == id)][score_name].item() for id in ids]


# get awareness and responsibility scores of each group (at/above median and below median IAT d-score)
group_above_median = extract_group_ids('A')
group_below_median = extract_group_ids('B')

q['Group'] = 'At/Above Median'
for p in group_below_median:
    q.loc[q['Q44'] == p, 'Group'] = 'Below Median'

print(q.columns)

above_median_awareness = extract_group_scores('Awareness', group_above_median)
below_median_awareness = extract_group_scores('Awareness', group_below_median)

above_median_responsibility = extract_group_scores('Responsibility', group_above_median)
below_median_responsibility = extract_group_scores('Responsibility', group_below_median)

# t tests
print("awareness+IAT t-test:", stats.ttest_ind(above_median_awareness, below_median_awareness), sep='\n')
print("responsibility+IAT t-test:", stats.ttest_ind(above_median_responsibility, below_median_responsibility), sep='\n')
print("awareness t-test:", stats.ttest_1samp(q['Awareness'], 0), sep='\n')
print("responsibility t-test:", stats.ttest_1samp(q['Responsibility'], 0), sep='\n')

q_long = pd.melt(q, id_vars='Group', value_vars=['Awareness', 'Responsibility'],
                 var_name='Variable', value_name='Value')

plt.figure(figsize=(10, 7))
ax = sns.boxplot(data=q_long, x='Variable', y='Value', palette=[Color.BERRY.value])
plt.ylim([-1, 1])
plt.xlabel("Questionnaire Score Distributions")
plt.ylabel('Score')
plt.axhline(0, color='k', linestyle='dashed', label='Zero Point')
plt.legend()
plt.savefig('Questionnaire Score Distribution.png')
# plt.show()

plt.figure(figsize=(10, 7))
ax = sns.boxplot(data=q_long, x='Variable', y='Value', hue='Group', palette=[Color.LADY_BUG.value, Color.MAROON.value])
plt.xlabel('Questionnaire Scores Distributions by IAT Group')
plt.ylabel('Score')
plt.axhline(0, color='k', linestyle='dashed', linewidth=1, label='Zero Point')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('Questionnaire scores by Median Split Group.png')
# plt.show()

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re

plt.rcParams.update({'font.size': 16})
# plt.rcParams['font.family'] = 'Linux Libertine O'
plt.rcParams['figure.dpi'] = 600


def calculate_iat_d6_score_from_custom_blocks(
        df_participant: pd.DataFrame,
        block3_4_condition_name: str,
        block6_7_condition_name: str,
        n_trials_b3: int,
        n_trials_b4: int,
        n_trials_b6: int,
        n_trials_b7: int
) -> float:
    """
    calculates the IAT D6-score for a single participant based on study ITA
    """
    # .iloc[0] gets the first element's value without taking into account  index labels.
    func_internal_pid_value = df_participant['participant_id'].iloc[0]
    log_pid_label = f"P{func_internal_pid_value}"

    rt_too_fast_threshold = 300  # study tresholds given
    rt_too_slow_threshold = 10000
    error_penalty = 600
    min_valid_trials_for_block_mean = 3

    block_dfs_raw = {}
    processed_trials_for_sd = {}
    block_means_penalized = {}

    if not df_participant['rowNo'].is_monotonic_increasing:
        df_participant = df_participant.sort_values(by='rowNo').reset_index(drop=True)

    b3_b4_trials_df = df_participant[df_participant['condition1'] == block3_4_condition_name]
    if len(b3_b4_trials_df) < (n_trials_b3 + n_trials_b4):
        print(f"Warning ({log_pid_label}): Insufficient trials for condition '{block3_4_condition_name}'. "
              f"Found {len(b3_b4_trials_df)}, expected {n_trials_b3 + n_trials_b4}. skip.")
        return np.nan
    block_dfs_raw['B3'] = b3_b4_trials_df.iloc[:n_trials_b3].copy()
    block_dfs_raw['B4'] = b3_b4_trials_df.iloc[n_trials_b3: n_trials_b3 + n_trials_b4].copy()

    b6_b7_trials_df = df_participant[df_participant['condition1'] == block6_7_condition_name]
    if len(b6_b7_trials_df) < (n_trials_b6 + n_trials_b7):
        print(f"Warning ({log_pid_label}): Insufficient trials for condition '{block6_7_condition_name}'. "
              f"Found {len(b6_b7_trials_df)}, expected {n_trials_b6 + n_trials_b7}. skip.")
        return np.nan
    block_dfs_raw['B6'] = b6_b7_trials_df.iloc[:n_trials_b6].copy()
    block_dfs_raw['B7'] = b6_b7_trials_df.iloc[n_trials_b6: n_trials_b6 + n_trials_b7].copy()

    for block_label, block_df_orig in block_dfs_raw.items():
        if block_df_orig.empty:
            print(f"Warning ({log_pid_label}): Block {block_label} is empty. skip.")
            return np.nan

        block_df_filtered = block_df_orig[
            (block_df_orig['RT'] >= rt_too_fast_threshold) &
            (block_df_orig['RT'] <= rt_too_slow_threshold)
            ].copy()

        if len(block_df_filtered) < min_valid_trials_for_block_mean:
            print(
                f"Warning ({log_pid_label}): < {min_valid_trials_for_block_mean} valid trials in {block_label}. skip.")
            return np.nan

        correct_trials_in_block = block_df_filtered[block_df_filtered['correct'] == 1]
        if len(correct_trials_in_block) == 0:
            print(f"Warning ({log_pid_label}): No correct trials in {block_label} for penalty. skip.")
            return np.nan
        mean_correct_rt_for_block = correct_trials_in_block['RT'].mean()

        block_df_filtered['RT_penalized'] = block_df_filtered['RT']
        error_indices = block_df_filtered[block_df_filtered['correct'] == 0].index
        block_df_filtered.loc[error_indices, 'RT_penalized'] = mean_correct_rt_for_block + error_penalty

        processed_trials_for_sd[block_label] = block_df_filtered['RT_penalized']
        block_means_penalized[block_label] = block_df_filtered['RT_penalized'].mean()

    all_trials_B3_B4_penalized = pd.concat([
        processed_trials_for_sd.get('B3', pd.Series(dtype=float)),
        processed_trials_for_sd.get('B4', pd.Series(dtype=float))
    ])
    if len(all_trials_B3_B4_penalized) < 2:
        print(f"Warning ({log_pid_label}): Not enough data for SD_3_4. skip.")
        return np.nan
    sd_3_4 = all_trials_B3_B4_penalized.std(ddof=0)

    all_trials_B6_B7_penalized = pd.concat([
        processed_trials_for_sd.get('B6', pd.Series(dtype=float)),
        processed_trials_for_sd.get('B7', pd.Series(dtype=float))
    ])
    if len(all_trials_B6_B7_penalized) < 2:
        print(f"Warning ({log_pid_label}): Not enough data for SD_6_7. skip.")
        return np.nan
    sd_6_7 = all_trials_B6_B7_penalized.std(ddof=0)

    if sd_3_4 == 0 or pd.isna(sd_3_4) or sd_6_7 == 0 or pd.isna(sd_6_7):
        print(f"Warning ({log_pid_label}): Zero or NaN pooled SD. skip.")
        return np.nan

    required_means = ['B3', 'B4', 'B6', 'B7']
    if not all(b_label in block_means_penalized for b_label in required_means):
        print(f"Warning ({log_pid_label}): Not all block means calc'd. skip.")
        return np.nan

    mean_B3 = block_means_penalized['B3']
    mean_B4 = block_means_penalized['B4']
    mean_B6 = block_means_penalized['B6']
    mean_B7 = block_means_penalized['B7']

    score1 = (mean_B6 - mean_B3) / sd_3_4
    score2 = (mean_B7 - mean_B4) / sd_6_7
    d6_score = (score1 + score2) / 2

    for block_label, block_df_orig in block_dfs_raw.items():
        original_trials_count = len(block_df_orig)
        if original_trials_count > 0:
            fast_responses_count = len(block_df_orig[block_df_orig['RT'] < rt_too_fast_threshold])
            if (fast_responses_count / original_trials_count) > 0.10:
                print(f"FLAG ({log_pid_label}): >10% trials < {rt_too_fast_threshold}ms in raw {block_label} "
                      f"({fast_responses_count}/{original_trials_count}). D-score ({d6_score:.4f}) calculated, but consider participant exclusion.")
    return d6_score


# getting data
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "implicit-association-test")

list_of_dataframes = []
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if file_name.endswith('.csv'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            header_idx = -1
            for idx, line in enumerate(lines):
                if line.strip().startswith('rowNo'):
                    header_idx = idx
                    break
            if header_idx == -1:
                print(f"No data header ('rowNo') found in {file_name}, skip.")
                continue
            df = pd.read_csv(file_path, skiprows=header_idx)
            df['participant_id'] = file_name.split('.')[0][-3:]
            list_of_dataframes.append(df)
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

CONDITION1_FOR_BLOCK_3_4 = "incongruent"
CONDITION1_FOR_BLOCK_6_7 = "congruent"

N_TRIALS_B3 = 10
N_TRIALS_B4 = 10
N_TRIALS_B6 = 10
N_TRIALS_B7 = 10

# loop
if not list_of_dataframes:
    print("Error: `list_of_dataframes` is empty. Please check data loading.")
else:
    print(f"Found {len(list_of_dataframes)} participant dataframes.")
    print(
        f"Using Condition1 for B3/B4 (e.g., Incongruent): '{CONDITION1_FOR_BLOCK_3_4}' ({N_TRIALS_B3} pract, {N_TRIALS_B4} test)")
    print(
        f"Using Condition1 for B6/B7 (e.g., Congruent): '{CONDITION1_FOR_BLOCK_6_7}' ({N_TRIALS_B6} pract, {N_TRIALS_B7} test)")

    all_d_scores_calculated = []
    participant_ids_for_scores = []

    for i, df_participant_orig in enumerate(list_of_dataframes):
        participant_num_actual = i + 1  # simple counter for participant number if ID not in in the file

        df_participant = df_participant_orig.copy()
        if 'participant_id' not in df_participant.columns:
            df_participant['participant_id'] = participant_num_actual

        outer_loop_pid_value = df_participant['participant_id'].iloc[0]
        outer_loop_participant_label = f"{outer_loop_pid_value}"
        print(f"\nProcessing {outer_loop_participant_label}...")

        required_cols = ['rowNo', 'condition1', 'RT', 'correct']
        if not all(col in df_participant.columns for col in required_cols):
            print(f"  {outer_loop_participant_label}: Missing required columns:skip.")
            continue

        try:
            for col in ['RT', 'correct', 'rowNo']:
                df_participant[col] = pd.to_numeric(df_participant[col], errors='coerce')
        except Exception as e:
            print(f"  {outer_loop_participant_label}: Error converting essential columns to numeric: {e}. skip.")
            continue

        if df_participant[['RT', 'correct', 'rowNo']].isnull().values.any():
            print(f"  {outer_loop_participant_label}: NaN in essential columns after conversion. skip.")
            continue

        unique_conditions = df_participant['condition1'].unique()
        if CONDITION1_FOR_BLOCK_3_4 not in unique_conditions:
            print(f"  {outer_loop_participant_label}: Missing B3/B4 condition '{CONDITION1_FOR_BLOCK_3_4}'. skip.")
            continue
        if CONDITION1_FOR_BLOCK_6_7 not in unique_conditions:
            print(f"  {outer_loop_participant_label}: Missing B6/B7 condition '{CONDITION1_FOR_BLOCK_6_7}'. skip.")
            continue

        d_score = calculate_iat_d6_score_from_custom_blocks(
            df_participant,
            CONDITION1_FOR_BLOCK_3_4,
            CONDITION1_FOR_BLOCK_6_7,
            N_TRIALS_B3,
            N_TRIALS_B4,
            N_TRIALS_B6,
            N_TRIALS_B7
        )

        if pd.notna(d_score):
            all_d_scores_calculated.append(d_score)
            participant_ids_for_scores.append(outer_loop_pid_value)
            print(f"  {outer_loop_participant_label} D6-score: {d_score:.4f}")
        else:
            print(f"  {outer_loop_participant_label}: Couldn't calculate D-score.")

    # d-scores
    if all_d_scores_calculated:
        d_scores_array = np.array(all_d_scores_calculated)
        print(f"\n\n--- Overall IAT Analysis ({len(d_scores_array)} D-scores calculated) ---")

        mean_d = d_scores_array.mean()
        std_d = d_scores_array.std(ddof=1)
        median_d = np.median(d_scores_array)
        min_d = d_scores_array.min()
        max_d = d_scores_array.max()

        print(f"Mean D-score: {mean_d:.3f}")
        print(f"Standard Deviation of D-scores: {std_d:.3f}")
        print(f"Median D-score: {median_d:.3f}")
        print(f"Min D-score: {min_d:.3f}")
        print(f"Max D-score: {max_d:.3f}")

        if len(d_scores_array) > 1:
            t_statistic, p_value = stats.ttest_1samp(d_scores_array, 0)
            print(f"\nOne-sample t-test (against 0):")
            print(f"  t-statistic = {t_statistic:.3f}")
            print(f"  p-value = {p_value:.4f}")
            print(f"  Degrees of freedom = {len(d_scores_array) - 1}")

            if p_value < 0.05:
                print("  The mean D-score is significantly different from 0.")
            else:
                print("  The mean D-score is NOT significantly different from 0.")
        else:
            print("\nNot enough valid D-scores for a t-test.")

        print("\nRsults IAT:")
        print(f"  - B3/B4 ('{CONDITION1_FOR_BLOCK_3_4}'): Male+Assistant / Female+Expert")
        print(f"  - B6/B7 ('{CONDITION1_FOR_BLOCK_6_7}'): Male+Expert / Female+Assistant")
        print(f"  - If pairing in B6/B7 is FASTER (lower RTs) than B3/B4, D-score will be NEGATIVE.")
        print(f"    A NEGATIVE D-score means a stronger implicit association for the B6/B7 pairing.")
        print(f"    A POSITIVE D-score means a stronger implicit association for the B3/B4 pairing.")
        print(f"  - The overall  Mean D-score is: {mean_d:.3f}")

        # median split
        if len(d_scores_array) >= 2:
            print(f"\n--- Median Split Analysis (Median D-score = {median_d:.3f}) ---")

            scores_df = pd.DataFrame(
                {'id': [f"P{pid}" for pid in participant_ids_for_scores], 'd_score': d_scores_array})

            group_below_median = scores_df[scores_df['d_score'] < median_d]
            group_at_or_above_median = scores_df[scores_df['d_score'] >= median_d]

            print(f"\nGroup 1: D-scores BELOW median ({len(group_below_median)} participants)")
            if not group_below_median.empty:
                print(f"  Mean D-score: {group_below_median['d_score'].mean():.3f}")

                if len(group_below_median) > 1:
                    print(f"  SD D-score:   {group_below_median['d_score'].std(ddof=1):.3f}")
                else:
                    print(f"  SD D-score:   N/A")
                print(f"  Min D-score:  {group_below_median['d_score'].min():.3f}")
                print(f"  Max D-score:  {group_below_median['d_score'].max():.3f}")
            else:
                print("  No participants in this group (all scores could be equal to median).")

            print(f"\nGroup 2: D-scores AT or ABOVE median ({len(group_at_or_above_median)} participants)")
            if not group_at_or_above_median.empty:
                print(f"  Mean D-score: {group_at_or_above_median['d_score'].mean():.3f}")

                if len(group_at_or_above_median) > 1:
                    print(f"  SD D-score:   {group_at_or_above_median['d_score'].std(ddof=1):.3f}")
                else:
                    print(f"  SD D-score:   N/A")
                print(f"  Min D-score:  {group_at_or_above_median['d_score'].min():.3f}")
                print(f"  Max D-score:  {group_at_or_above_median['d_score'].max():.3f}")
            else:
                print("  No participants in this group")
        else:
            print("\nNot enough D-scores for median split ")

    # plots
    if all_d_scores_calculated:

        # histogram of d-scores
        plt.figure(figsize=(8, 6))
        sns.histplot(d_scores_array, kde=True, bins=10)
        # plt.title('Distribution of IAT D-scores')
        plt.xlabel('D-score')
        plt.ylabel('Frequency')
        plt.axvline(mean_d, color='r', linestyle='dashed', linewidth=1, label=f'Mean D: {mean_d:.3f}')
        plt.axvline(median_d, color='g', linestyle='dashed', linewidth=1, label=f'Overall Median D: {median_d:.3f}')
        plt.axvline(0, color='k', linestyle='solid', linewidth=1, label='Zero Point')
        plt.legend()
        plt.tight_layout()
        plt.savefig("Distribution of IAT D-scores.png")

        # boxplot of d-scores by median split groups
        if 'scores_df' not in locals() or 'iat_group' not in scores_df.columns:
            temp_scores_df = pd.DataFrame({
                'id': [f"P{pid}" for pid in participant_ids_for_scores],
                'd_score': d_scores_array
            })
            temp_scores_df['iat_group'] = np.where(
                temp_scores_df['d_score'] < median_d,
                'Below Median',
                'At/Above Median'
            )
        else:
            temp_scores_df = scores_df.copy()

        if not temp_scores_df.empty:
            plt.figure(figsize=(10, 7))
            ax = sns.boxplot(x='iat_group', y='d_score', data=temp_scores_df, order=['Below Median', 'At/Above Median'],
                             palette={'Below Median': 'skyblue', 'At/Above Median': 'lightcoral'})
            sns.stripplot(x='iat_group', y='d_score', data=temp_scores_df, order=['Below Median', 'At/Above Median'],
                          color='black', alpha=0.4, jitter=True, ax=ax)

            ax.axhline(median_d, color='purple', linestyle='dotted', linewidth=2,
                       label=f'Overall Median ({median_d:.3f})')

            group_medians = temp_scores_df.groupby('iat_group')['d_score'].median()

            for i, group_name in enumerate(['Below Median', 'At/Above Median']):
                if group_name in group_medians:
                    group_median_val = group_medians[group_name]
                    text_y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    text_y = group_median_val + text_y_offset if group_median_val >= 0 else group_median_val - text_y_offset * 2
                    if group_name == 'Below Median' and group_median_val < -1:
                        text_y = group_median_val + text_y_offset * 2

                    ax.text(i,
                            text_y,  # y-position for the text
                            f'Group Median: {group_median_val:.3f}',
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            color='black',
                            fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.2))

            # plt.title('D-scores by Median Split Group')
            plt.xlabel('IAT Group (Split by Overall Median D-score)')
            plt.ylabel('D-score')
            plt.axhline(0, color='k', linestyle='dashed', linewidth=1, label='Zero Point')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig("D-scores by Median Split Group.png")

# linor
"""
the group_below_median is the sexist group and group_at_or_above_median is the nice group
they are both panda dataframes
they have wo columns id, d_score

the id is the number of the participant, for example P12 is participant 12, i recommend 
using this id's for your analysis as well in case the order of the csv files changes or something
"""

print(group_below_median[:5])
print(len(group_at_or_above_median))
print(participant_ids_for_scores)

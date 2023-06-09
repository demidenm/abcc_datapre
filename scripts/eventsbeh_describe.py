import argparse
import json
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


# test sub,ses,task 
#in_dir = '/Users/michaeldemidenko/Downloads'
#out_dir = '/Users/michaeldemidenko/Downloads'
#sub = '###'
#ses = '2YearFollowUpYArm1'
#task= 'MID'


def convert_to_numeric(df):
    """
    Convert all object columns in the pd dataframe to numeric values.

    """
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
    return df


# Create ArgumentParser object
parser = argparse.ArgumentParser(description="This script runs and converts the .txt events data to events.tsv for each MID run.")

# Add required positional arguments
parser.add_argument("-i", "--in_dir", type=str, required=True,
                    help="Input directory path where the events.tsv files  are")
parser.add_argument("-o", "--out_dir", type=str, required=True,
                    help="Output directory path where resulting json/png images show go")
parser.add_argument("-s", "--sub", type=str, required=True,
                    help="subject argument (e.g. NVDAFL737P) without sub prefix")
parser.add_argument("-e", "--ses", type=str, required=True,
                    help="session argument (e.g. baselinearm1, 2YearFollowUpYArm1) without ses prefix")
parser.add_argument("-t", "--task", type=str, required=True,
                    help="task argument (e.g. MID, SST, nback)")

# Parse command-line arguments
args = parser.parse_args()

# Assign values to variables
in_dir = args.in_dir
out_dir = args.out_dir
sub = args.sub
ses = args.ses
task = args.task

# fine all files
files = glob(f"{in_dir}/sub-{sub}_ses-{ses}_task-{task}_run-0*_events.tsv", recursive=True)


if task == "MID":
    # column names, specify
    trial_n = 'SubTrial'
    probe_mrt = ['prbrt',"OverallRT"]
    cue_type = 'Condition'
    feedback_type = 'Result'
    trial_accuracy = 'prbacc'
    run_accuracy = 'percentacc'

    # create empty variables to save values to
    mid_descr = {}
    mid_fig_combined, axes = plt.subplots(nrows=2, ncols=7, figsize=(30, 14))
    
    def map_condition_to_reward_category(condition):
        if condition in ['LgReward', 'SmallReward']:
            return 'gain'
        elif condition in ['LgPun', 'SmallPun']:
            return 'loss'
        elif condition == 'Triangle':
            return 'neutral'
        else:
            return np.nan
        
    
    for i, file_name in enumerate(files):
        # Load the file into a DataFrame
        df = pd.read_csv(file_name, sep ='\t')
        
        
        if probe_mrt[1] in df.columns:
            mrt = probe_mrt[1]
        else:
            mrt = probe_mrt[0]
            
        # create neew gain, loss, neutral general column
        df['RewCat'] = df[cue_type].apply(map_condition_to_reward_category)
        df['TrialwiseCat'] = df['RewCat'].shift() + '-' + df['RewCat']
            
        # Replace to distinguish hit/miss for neutral
        # e.g. 'No money at stake!' with 'Neutral Hit' for trial_accuracy == 1
        df.loc[(df['Result'] == 'No money at stake!') & (df[trial_accuracy] == 1), 'Result'] = 'Neutral Hit'

        # e.g. 'No money at stake!' with 'Neutral Miss' for trial_accuracy == 0
        df.loc[(df['Result'] == 'No money at stake!') & (df[trial_accuracy] == 0), 'Result'] = 'Neutral Miss'
        
        # running accuracy not extracted into .tsv for MID from AHRB, calculate here.
        #cumulative_hits = df[trial_accuracy].dropna().cumsum()
        #cumulative_trials = pd.Series(range(1, len(df) + 1))
        
        
        try:
            cumulative_hits = df[trial_accuracy].astype(int).cumsum()
            cumulative_trials = pd.Series(range(1, len(df) + 1))
            running_accuracy = cumulative_hits / cumulative_trials
        except (TypeError, ValueError):
            # scenrio adjusting for when trials are missing values and have '?' or 'nan' values
            df = df.dropna(subset=['Cue.OnsetTime', 'Anticipation.OnsetTime', 'Anticipation.Duration', 'Feedback.OnsetTime'])
            df = convert_to_numeric(df)
            cumulative_hits = df[trial_accuracy].astype(int).cumsum()
            cumulative_trials = pd.Series(range(1, len(df) + 1))
            cumulative_hits = cumulative_hits.astype(float)
            cumulative_trials = cumulative_trials.astype(float)
            running_accuracy = cumulative_hits / cumulative_trials
            
        df[run_accuracy] = running_accuracy*100
        
        # create subtrial column (e.g. len rows)
        df[trial_n] = range(1, len(df) + 1)
        
        # calculating values for jsons
        # Calculate the counts per Condition/Result variable
        counts_per_condition = df[cue_type].value_counts().to_dict()
        counts_per_result = df[feedback_type].value_counts().to_dict()
        
        # trial sequences
        trialwise_seq_counts = df['TrialwiseCat'].value_counts().to_dict()
        
        try:
            trial_order_n = df['TrialOrder'].unique().astype(int)
        except KeyError:
            trial_order_n = np.array([999])
            
        # Calculate the sum of the counts for total trial count
        total_trial_count = sum(counts_per_condition.values())
        
        # calculate 
        mrt_no_nazero = df.loc[(df[mrt].notna()) & (df[mrt] != 0), mrt].mean()
    
        # Start creating the different plots, FOUR figures for each run
        # Group by Condition and calculate the mean of prbacc
        grouped_hit = df.groupby(cue_type)[trial_accuracy].mean()
        # Calculate the overall average of prbacc
        overall_avg = df[trial_accuracy].mean()
    
        ## plot1: Create a bar plot of the grouped means and add a line for the overall average
        grouped_plot = axes[i, 0].bar(grouped_hit.index, grouped_hit.values)
        axes[i, 0].axhline(overall_avg, color='r', linestyle='dashed')
        axes[i, 0].axhline(0.6, color='k', linestyle='solid')
        axes[i, 0].set_ylim([0, 1])
        axes[i, 0].set_ylabel('Accuracy (%)')
        axes[i, 0].set_title(f'Run 0{i+1} \n Accuracy by Condition \n (red: avg across all; black: target)')
        axes[i, 0].tick_params(axis='x', rotation=45, labelsize=10)
    
        # plot2: reate a line plot of PECENT ACCURACY by SubTrial
        accuracy_plot = df.plot(x=trial_n, y=run_accuracy, ax=axes[i, 1])
        axes[i, 1].set_ylim([0, 100])
        axes[i, 1].set_ylabel('Accuracy (%)')
        axes[i, 1].set_title(f'Run 0{i+1} \n Accuracy Plot \n (Percent Accuracy: {df[run_accuracy].iloc[-1]:.2f}%)')
        
        # plot3: create a plot of RT across hit/miss
        grouped_df = df.groupby(trial_accuracy)[mrt].agg(['mean', 'std'])
        grouped_df.plot(kind='bar', y='mean', yerr='std', ax=axes[i, 2], legend=False)
        # set x-axis tick labels
        axes[i, 2].set_xticks([0, 1])
        axes[i, 2].set_xticklabels(['Miss', 'Hit'])
        axes[i, 2].set_ylim([50, 600])
        axes[i, 2].set_title(f'Run 0{i+1} \n Mean and SD for Probe \n across Hit/Miss')
        axes[i, 2].set_xlabel('Probe Hit/Miss')
        axes[i, 2].set_ylabel('RT (ms)')
        
        # Create RT plot across trials    
        # some EPRIME configs have 0 or missing. so remove
        # will ger error if no 
        try:
            RT_zero_na = df[mrt].value_counts(dropna=False).loc[0] + df[mrt].isna().sum()
        except KeyError:
            RT_zero_na = df[mrt].value_counts(dropna=False).get(0, 0) + df[mrt].isna().sum()
        
        non_zero_df = df[df[mrt].notna() & (df[mrt] != 0)]
        
        
        # Create a line plot of MRT by SubTrial
        rt_plot = non_zero_df.plot(x=trial_n, y=mrt, ax=axes[i, 3])
        axes[i, 3].set_ylim([50, 650])
        axes[i, 3].set_ylabel('RT (ms)')
        axes[i, 3].set_title(f'Run 0{i+1} \n RT Plot \n (Removed NAs/Zeros: {RT_zero_na})')
        
        # create a bar plot of MRT by condition
        mrt_by_acc = non_zero_df.groupby(trial_accuracy)[mrt].mean()
        grouped_mrt = non_zero_df.groupby(cue_type)[mrt].mean()
        mrt_cond_plot = axes[i, 4].bar(grouped_mrt.index, grouped_mrt.values)
        axes[i, 4].axhline(mrt_no_nazero, color='r', linestyle='dashed')
        axes[i, 4].set_ylim([50, 650])
        axes[i, 4].set_ylabel('RT (ms)')
        axes[i, 4].set_title(f'Run 0{i+1} \n RT by Condition \n (Removed NAs/zeros: {RT_zero_na})')
        axes[i, 4].tick_params(axis='x', rotation=45, labelsize=10)
        
        # Create feedback counts
        # create a list of conditions in the desired order
        feedback_order = ['You earn $5!','You did not earn $5!','You earn $0.20!','You did not earn $0.20!',
                          'Neutral Hit', 'Neutral Miss',
                          'You keep $5!', 'You lose $5!','You lose $0.20!','You keep $0.20!'
                         ]
        feedback_ord_abv = ['Earn $5','Didnt earn $5','Earn $0.20','Didnt earn $0.20',
                          'Neutral Hit', 'Neutral Miss',
                          'Keep $5', 'Lose $5','Lose $0.20','Keep $0.20']
                
        # create a list of values corresponding to the condition order
        feedback_values = [counts_per_result.get(c,0) for c in feedback_order]
        
        # create bar plot of counts_per_result with ordered conditions
        result_group = axes[i, 5].bar(feedback_order, feedback_values)
        
        # set x-axis tick labels
        axes[i, 5].set_xticklabels(feedback_ord_abv)
        
        # set plot limits, title, and labels
        axes[i, 5].set_ylim([0, 10])
        axes[i, 5].set_title(f'Run 0{i+1} \n Feedback Hit/Miss')
        axes[i, 5].set_xlabel('Condition Type')
        axes[i, 5].set_ylabel('Count (n)')
        axes[i, 5].tick_params(axis='x', rotation=45, labelsize=8)
        
        # plot 6 sequence
        order_plot = axes[i, 6].bar(trialwise_seq_counts.keys(), trialwise_seq_counts.values())

        # Add labels and title
        axes[i, 6].set_title(f'Run 0{i+1} \n T + T+1 Reward Category  \n Task Order: {trial_order_n}')
        axes[i, 6].set_ylim([0, 15])
        axes[i, 6].set_xlabel('T to T+1 Category')
        axes[i, 6].set_ylabel('Count (n)')
        axes[i, 6].tick_params(axis='x', rotation=45, labelsize=8)
        
        # Create a dictionary to store the calculated values for this run
        data = {
            'Total Trials': total_trial_count,
            'Trial Order': trial_order_n.tolist(),
            'Trialwise Sequence': trialwise_seq_counts,
            'Trials Per Cue Condition': counts_per_condition,
            'Trials Per Feedback Condition': counts_per_result,
            'Accuracy by Cue Condition': grouped_hit.to_dict(),
            'Overall Accuracy': overall_avg,
            'Mean RT': mrt_no_nazero,
            'Mean RT by Hit/Miss': mrt_by_acc.to_dict(),
            'Mean RT by Cue Condition': grouped_mrt.to_dict()
        }
    
        # Add the data for this run to the dictionary of all data
        mid_descr[f'Run {i+1}'] = data
    
    # Create combined figures. First, generate subtile. Then add the subplots
    # making tight layout and then save out png
    mid_fig_combined.tight_layout()
    mid_fig_combined.savefig(f"{out_dir}/sub-{sub}_ses-{ses}_task-{task}_plot-combined.png")
    
    # Writeout the .json file for each mid description
    with open(f"{out_dir}/sub-{sub}_ses-{ses}_task-{task}_beh-descr.json", 'w') as f:
        json.dump(mid_descr, f, indent=4)
        
elif task == "SST":
    # column names, specify
    lab_trial_n = 'Trial'
    lab_trial_type = 'TrialCode'
    lab_go_rt = 'Go.RT'
    lab_go_accuray = 'Go.ACC'
    lab_ssd_rt = 'SSD.RT'
    lab_stopsig_rt = 'StopSignal.RT'
    lab_stopsig_accuacy = 'StopSignal.ACC'
    lab_ssd_duration = 'SSDDur'
    
    # function for SST
    # formula provided by Jaime Rios: integration method w/ replacement of go omissions 
    def compute_ssrt(df, max_go_rt = 2000):
        # pull all go trials
        go_trials = df[df[lab_trial_type].isin(['CorrectGo', 'IncorrectGo'])]
        # pull all stop trials
        stop_trials = df[df[lab_trial_type].isin(['CorrectStop', 'IncorrectStop'])]
        
        # omission replacement
        go_replacement_df = go_trials.where(go_trials[lab_go_rt] != 0, max_go_rt)
        sorted_go = go_replacement_df[lab_go_rt].sort_values(ascending = True, ignore_index=True)
        stop_failure = stop_trials.loc[stop_trials[lab_stopsig_rt] != 0]
        
        p_respond = len(stop_failure)/len(stop_trials)
        avg_SSD = stop_trials.SSDDur.mean()
        
        nth_index = int(np.rint(p_respond*len(sorted_go))) - 1
        
        if nth_index < 0:
            nth_RT = sorted_go[0]
        elif nth_index >= len(sorted_go):
            nth_RT = sorted_go[-1]
        else:
            nth_RT = sorted_go[nth_index]
        
        ssrt = nth_RT - avg_SSD
        
        return ssrt

    # create empty variables to save values to
    sst_descr = {}
    sst_fig_combined, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 12))
    
    for i, file_name in enumerate(files):
        # Load the file into a DataFrame
        df = pd.read_csv(file_name, sep ='\t')
        
        # calculating values for jsons
        # n trial types 
        trialtype_count = {k: v for k, v in df[lab_trial_type].value_counts().items() 
                           if k not in ['BeginFix', 'EndFix']}

        
        # accuracies for Go, StopSignal
        go_acc = df[lab_go_accuray].mean()
        stopsig_acc = df[lab_stopsig_accuacy].mean()
        
        # mrt for Go, SSD and StopSignal
        go_mrt = df[lab_go_rt].mean()
        ssd_mrt = df.loc[df[lab_trial_type] == 'IncorrectStop', lab_ssd_rt].mean()
        stopsig_mrt = df.loc[df[lab_trial_type] == 'IncorrectStop', lab_stopsig_rt].mean()
        
        # durations, SSD
        min_value = df[lab_ssd_duration].min(skipna=True)
        max_value = df[lab_ssd_duration].max(skipna=True)
        mean_value = df[lab_ssd_duration].mean(skipna=True)
        
        ssd_dict = {'min': min_value, 'max': max_value, 'mean': mean_value}
        
        # calculate ssrt
        ssrt_est = compute_ssrt(df)
        
        
        # Create a dictionary to store the calculated values for this run
        data = {
            'Total Trial Types': trialtype_count,        
            'Go Accuracy': go_acc,
            'Go MRT': go_mrt,
            'Stop Signal Accuracy': stopsig_acc,
            'Stop Signal MRT': stopsig_mrt,
            'Stop Signal Delay MRT': ssd_mrt,
            'Stop Signal Reaction Time': ssrt_est,
            'Stop Signal Delay Durations': ssd_dict
        }
    
        # Add the data for this run to the dictionary of all data
        sst_descr[f'Run {i+1}'] = data
        
        ## Plotting figures
        
        # Plot1: trial types bar
        x_axis = list(trialtype_count.keys())
        y_axis = list(trialtype_count.values())
        bar_trialtypes = axes[i, 0].bar(x_axis, y_axis, align='center')
        axes[i, 0].set_ylim([0, 150])
        axes[i, 0].set_ylabel('Count (N)')
        axes[i, 0].set_xlabel('')
        axes[i, 0].set_title(f'Run 0{i+1} \n Count by Trial Type')
        axes[i, 0].tick_params(axis='x', rotation=45, labelsize=10)
        
        # Plot2: RTs across condition Go, SSD, StopSig
        go_rt = df[lab_go_rt].where(lambda x: x != 0).dropna()
        ssd_rt = df[lab_ssd_rt].where(lambda x: x != 0).dropna()
        stop_rt = df[lab_stopsig_rt].where(lambda x: x != 0).dropna()
        go_n = go_rt.count()
        ssd_n = ssd_rt.count()
        stop_n = stop_rt.count()
        
        # create a list of the three variables
        rt_non_nanzero_data = [go_rt, ssd_rt, stop_rt]
        # plot the data in one boxplot with a custom color
        rt_non_nanzero = axes[i, 1].boxplot(rt_non_nanzero_data, patch_artist=True, 
                                            boxprops=dict(facecolor='#1f77b4'),
                                            medianprops=dict(color='white'))
        for box in rt_non_nanzero['boxes']:
            box.set(edgecolor='#1f77b4', linewidth=1.5)
        
        axes[i, 1].set_ylim([0, 1250])
        axes[i, 1].axhline(y=ssrt_est, color='red', linestyle='--', label='SSRT')
        axes[i, 1].set_title(f'Run 0{i+1} \n RTs Across Trial Types \n Go (n:{go_n}), SSD (n:{ssd_n}), StopSig (n:{stop_n}), SSRT: red')
        axes[i, 1].set_xlabel('')
        axes[i, 1].set_xticklabels(['Go RT', 'Stop Sig Delay RT', 'Stop Signal Fail RT'])
        axes[i, 1].set_ylabel('RT (ms)')
        axes[i, 1].tick_params(axis='x', rotation=30, labelsize=10)
        
        # Plot3: RTs across time:
        axes[i, 2].set_xlabel('Trial')
        axes[i, 2].set_ylabel('RT (ms)')
        axes[i, 2].set_title(f'Run 0{i+1} \n RTs Across Trials \n Go (n:{go_n}), SSD (n:{ssd_n}), StopSig (n:{stop_n})')

        # plot the data for each variable as a line plot
        axes[i, 2].plot(go_rt.index, go_rt.values, label='Go RT')
        axes[i, 2].plot(ssd_rt.index, ssd_rt.values, label='SSD RT')
        axes[i, 2].plot(stop_rt.index, stop_rt.values, label='Stop Signal Fail RT')
        axes[i, 2].legend(loc='upper left', fontsize='small')
        
        # Plot3: RT distributions across Go/Stop trials
        df['Go_or_stop'] = df['TrialCode'].map({'CorrectGo': 'GoTrial', 'IncorrectGo': 'GoTrial',
                                        'CorrectStop': 'StopTrial', 'IncorrectStop': 'StopTrial'})
        
        go_data = df.loc[df['Go_or_stop'] == 'GoTrial', 'Go.Duration']
        ssd_data = df.loc[df['Go_or_stop'] == 'StopTrial', 'SSDDur']
        stop_signal_data = df.loc[df['Go_or_stop'] == 'StopTrial', 'StopSignal.Duration']
        
        # create a histogram for SSDDur
        n2, bins2, patches2 = axes[i,3].hist(x=ssd_data, bins=50, color='#1f77b4', alpha=0.7, rwidth=0.85, label='SSDDur')
        axes[i, 3].bar(bins2[:-1], n2, width=10, align='edge', color='#1f77b4')
        axes[i, 3].set_xlabel('Stop Signal Delay Duration (ms)')
        axes[i, 3].set_ylabel('Frequency')
        axes[i, 3].set_ylim([0,30])
        axes[i, 3].set_title(f'Run 0{i+1} \n Stop Signal Delay Durations \n (SSD n:{ssd_data.count()})')
        
        # create a histogram for StopSignal.Duration
        x_labels = ['Go Accuracy', 'Stop Signal Accuracy']
        means = [go_acc, stopsig_acc]

        axes[i, 4].bar(x_labels, means, color='#1f77b4')
        axes[i, 4].set_xlabel('Condition')
        axes[i, 4].set_ylabel('% Accuracy')
        axes[i, 4].set_ylim([0, 1])  # Assuming accuracy is a value between 0 and 1
        axes[i, 4].set_title(f'Run 0{i+1} \n Mean Accuracies \n (Go: {go_acc:.2f}, Stop: {stopsig_acc:.2f})')

        
        # making tight layout and then save out png
        sst_fig_combined.tight_layout()
        sst_fig_combined.savefig(f"{out_dir}/sub-{sub}_ses-{ses}_task-{task}_plot-combined.png")
        
        # Writeout the .json file for each mid description
        with open(f"{out_dir}/sub-{sub}_ses-{ses}_task-{task}_beh-descr.json", 'w') as f:
            json.dump(sst_descr, f, indent=4)        
        
elif task == "nback":
    # column names, specify
    lab_trial_n = 'SubTrial'
    lab_block = 'BlockType'
    lab_stimulus = 'StimType'
    lab_target = 'TargetType'
    lab_stim_acc = 'Stim.ACC'
    lab_stim_rt = 'Stim.RT'
    
    # define function for d-prime
    def calc_d_prime(df, block_col, acc_col):
        """
        n-back d-prime (D') for the specified conditions in a dataframe = df.
    
                        
        df (pandas.DataFrame): pd.DF containing the data.
        block_type_column (str): The column name for the block type e.g., 'BlockType'.
        accuracy_column (str): The column name for the accuracy (e.g. Stim.ACC).
        conditions (list): The list of conditions to calculate d-prime for (e.g. [0-Back, 2-Back]).
    
        Returns: A dictionary containing the calculated d-prime values for each condition.
        
        """
        
        block_type = df[block_col]
        accuracy = df[acc_col]
        
        # Step 2: Filter the data for 2-back and 0-back conditions
        is_2back = block_type == '2-Back'
        is_0back = block_type == '0-Back'
        accuracy_2back = accuracy[is_2back]
        accuracy_0back = accuracy[is_0back]
    
    
        hit_2back = accuracy_2back.mean()
        miss_2back = 1 - hit_2back
        hit_0back = accuracy_0back.mean()
        miss_0back = 1 - hit_0back
        
        hit_2back_z = norm.ppf(hit_2back)
        miss_2back_z = norm.ppf(miss_2back)
        hit_0back_z = norm.ppf(hit_0back)
        miss_0back_z = norm.ppf(miss_0back)
        
        dprime_2back = hit_2back_z - miss_2back_z
        dprime_0back = hit_0back_z - miss_0back_z
        
        dprime_dict = {
        '2back': dprime_2back,
        '0back': dprime_0back
        }
        
        return dprime_dict
    
    
    # create empty variables to save values to
    nback_descr = {}
    nback_fig_combined, axes = plt.subplots(nrows=2, ncols=4, figsize=(33, 20))
    
    
    for i, file_name in enumerate(files):
        # Load the file into a DataFrame
        df = pd.read_csv(file_name, sep ='\t')
        
        # counts of types
        blocktype_count = df.groupby(lab_block).size().to_dict()
        stimtype_count = df.groupby(lab_stimulus).size().to_dict()
        targettype_count = df.groupby(lab_target).size().to_dict()
        
        # acc by types
        blocktype_acc = df.groupby(lab_block)[lab_stim_acc].mean().to_dict()
        stimtype_acc = df.groupby(lab_stimulus)[lab_stim_acc].mean().to_dict()
        targettype_acc = df.groupby(lab_target)[lab_stim_acc].mean().to_dict()
        avg_acc = round(float(df[lab_stim_acc].mean()), 2)
        
        # RT by types
        blocktype_rt = df.groupby(lab_block)[lab_stim_rt].mean().to_dict()
        stimtype_rt = df.groupby(lab_stimulus)[lab_stim_rt].mean().to_dict()
        targettype_rt = df.groupby(lab_target)[lab_stim_rt].mean().to_dict()
        avg_rt = round(float(df[lab_stim_rt].mean()), 2)
        
        #d-prime calc
        dprime = calc_d_prime(df = df, block_col=lab_block, acc_col=lab_stim_acc)

        # Create a dictionary to store the calculated values for this run
        data = {
            'Block N': blocktype_count, 
            'Block Accuracy': blocktype_acc,
            'Block Mean RT': blocktype_rt,
            'Stimulus N': stimtype_count, 
            'Stimulus Accuracy': stimtype_acc,
            'Stimulus Mean RT': stimtype_rt,
            'Target N': targettype_count, 
            'Target Accuracy': targettype_acc,
            'Target Mean RT': targettype_rt,
            'Overall Accuracy': avg_acc,
            'Overall Mean RT': avg_rt,
            'D-prime':dprime
        }
    
        # Add the data for this run to the dictionary of all data
        nback_descr[f'Run {i+1}'] = data
        
        
        # Plot data for nback
        # plot counts across types
        #x_axis_counts = list(blocktype_count.keys()) + list(stimtype_count.keys()) + list(targettype_count.keys())
        #y_axis_counts = list(blocktype_count.values()) + list(stimtype_count.values()) + list(targettype_count.values())
        
        # Plot the bar chart on the appropriate subplot
        #bar_counttypes = axes[i, 0].bar(x_axis_counts, y_axis_counts, align='center')
        
        # Customize the subplot
        #axes[i, 0].set_ylim([0, max(y_axis_counts) + 5])
        #axes[i, 0].set_ylabel('Count (N)')
        #axes[i, 0].set_xlabel('')
        #axes[i, 0].set_title(f'Run 0{i+1} \n Counts for Stimulus Types')
        #axes[i, 0].tick_params(axis='x', rotation=45, labelsize=10)

        # plot accuracy across types
        x_axis_acc = list(blocktype_acc.keys()) + list(stimtype_acc.keys()) + list(targettype_acc.keys())
        y_axis_acc = list(blocktype_acc.values()) + list(stimtype_acc.values()) + list(targettype_acc.values())
        
        # Plot the bar chart on the appropriate subplot
        avg_acc_whole = avg_acc*100
        bar_acctypes = axes[i, 0].bar(x_axis_acc, y_axis_acc, align='center')
        
        # Customize the subplot
        axes[i, 0].set_ylim([0, 1.1])
        axes[i, 0].set_ylabel('Accuracy (%)')
        axes[i, 0].set_xlabel('')
        axes[i, 0].axhline(y=avg_acc, color='red', linestyle='--')
        axes[i, 0].set_title(f'Run 0{i+1} \n Accuracy across stimulus types \n Red: Avg Accuracy ({avg_acc_whole}%)')
        axes[i, 0].tick_params(axis='x', rotation=45, labelsize=10)
        
        # plot mRT across types
        x_axis_rt = list(blocktype_rt.keys()) + list(stimtype_rt.keys()) + list(targettype_rt.keys())
        y_axis_rt = list(blocktype_rt.values()) + list(stimtype_rt.values()) + list(targettype_rt.values())
        
        # Plot the bar chart on the appropriate subplot
        bar_rttypes = axes[i, 1].bar(x_axis_rt, y_axis_rt, align='center')
        
        # Customize the subplot
        axes[i, 1].set_ylim([0, 2000])
        axes[i, 1].set_ylabel('Mean RT (ms)')
        axes[i, 1].set_xlabel('')
        axes[i, 1].axhline(y=avg_rt, color='red', linestyle='--')
        axes[i, 1].set_title(f'Run 0{i+1} \n Mean RT Across Stimulus Types \n Red: mean RT ({avg_rt}ms)')
        axes[i, 1].tick_params(axis='x', rotation=45, labelsize=10)
        
        # plot trialwise data
        # Group the data by condition and trial and calculate the mean RT
        rt_by_condition = df.groupby([lab_block, lab_trial_n])[lab_stim_rt].mean().reset_index()
        
        # Pivot the data to have condition types as columns and trials as rows
        rt_pivot = rt_by_condition.pivot(index=lab_trial_n, columns=lab_block, values=lab_stim_rt)
        
        by_type_rt_plt = rt_pivot.plot(ax=axes[i,2], linestyle='-')
        axes[i, 2].set_ylabel('RT (ms)')
        axes[i, 2].set_xlabel('Trial')
        axes[i, 2].set_title(f'Run 0{i+1} \n RT Across Trials for Block Type')
        axes[i, 2].tick_params(axis='x', rotation=45, labelsize=10)
        axes[i, 2].legend(loc='upper right', facecolor='white', edgecolor='black')
        
        # plot dprime
        dprime_x_axis = list(dprime.keys())
        dprime_y_axis = list(dprime.values())
        
        axes[i, 3].bar(dprime_x_axis, dprime_y_axis, align='center')

        # Customize the subplot
        axes[i, 3].set_ylabel('D`')
        axes[i, 3].set_ylim([-2, 8])
        axes[i, 3].set_xlabel('Condition')
        axes[i, 3].set_title(f'Run 0{i+1} \n D` for Block Type')
        axes[i, 3].tick_params(axis='x', rotation=45, labelsize=10)
        
        # making tight layout and then save out png
        nback_fig_combined.tight_layout()
        nback_fig_combined.savefig(f"{out_dir}/sub-{sub}_ses-{ses}_task-{task}_plot-combined.png")
        
        # Writeout the .json file for each mid description
        with open(f"{out_dir}/sub-{sub}_ses-{ses}_task-{task}_beh-descr.json", 'w') as f:
            json.dump(nback_descr, f, indent=4)  
        
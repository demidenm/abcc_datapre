import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

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

# test
#in_dir = '/home/faird/mdemiden/slurm_ABCD_s3/Beh_Data/yr2_run1'
#out_dir = '/Users/michaeldemidenko/sherlock/data/AHRB/derivatives/analyses/proj_midinvar/mid_qc/files'
#in_dir = '/Users/michaeldemidenko/Downloads'
#out_dir = '/Users/michaeldemidenko/Downloads'
#sub = 'NDARINV33AC40WZ'
#ses = '2YearFollowUpYArm1'

# fine all files
files = glob(f"{in_dir}/sub-{sub}_ses-{ses}_task-${task}_run-0*_events.tsv", recursive=True)


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
    mid_fig_combined, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 10))
    
    for i, file_name in enumerate(files):
        # Load the file into a DataFrame
        df = pd.read_csv(file_name, sep ='\t')
        
        if probe_mrt[1] in df.columns:
            mrt = probe_mrt[1]
        else:
            mrt = probe_mrt[0]
        
        # running accuracy not extracted into .tsf for MID from AHRB, calculate here.
        cumulative_hits = df[trial_accuracy].cumsum()
        cumulative_trials = pd.Series(range(1, len(df) + 1))
        running_accuracy = cumulative_hits / cumulative_trials
        df[run_accuracy] = running_accuracy*100
        
        # create subtrial column (e.g. len rows)
        df[trial_n] = range(1, len(df) + 1)
        
        # calculating values for jsons
        # Calculate the counts per Condition/Result variable
        counts_per_condition = df[cue_type].value_counts().to_dict()
        counts_per_result = df[feedback_type].value_counts().to_dict()
        
        # Calculate the sum of the counts for total trial count
        total_trial_count = sum(counts_per_condition.values())
        
        # calculate 
        mrt_no_nazero = df.loc[(df[mrt].notna()) & (df[mrt] != 0), mrt].mean()
    
        # Start creating the different plots, FOUR figures for each run
        # Group by Condition and calculate the mean of prbacc
        grouped_hit = df.groupby(cue_type)[trial_accuracy].mean()
        # Calculate the overall average of prbacc
        overall_avg = df[trial_accuracy].mean()
    
        ## Create a bar plot of the grouped means and add a line for the overall average
        grouped_plot = axes[i, 0].bar(grouped_hit.index, grouped_hit.values)
        axes[i, 0].axhline(overall_avg, color='r', linestyle='dashed')
        axes[i, 0].axhline(0.6, color='k', linestyle='solid')
        axes[i, 0].set_ylim([0, 1])
        axes[i, 0].set_ylabel('Accuracy (%)')
        axes[i, 0].set_title(f'Run 0{i+1} \n Accuracy by Condition \n (red: avg across all; black: target)')
        axes[i, 0].tick_params(axis='x', rotation=45, labelsize=10)
    
        # Create a line plot of PECENT ACCURACY by SubTrial
        accuracy_plot = df.plot(x=trial_n, y=run_accuracy, ax=axes[i, 1])
        axes[i, 1].set_ylim([0, 100])
        axes[i, 1].set_ylabel('Accuracy (%)')
        axes[i, 1].set_title(f'Run 0{i+1} \n Accuracy Plot \n (Percent Accuracy: {df[run_accuracy].iloc[-1]:.2f}%)')
        
        # Create a plot of RT across hit/miss
        grouped_df = df.groupby(trial_accuracy)[mrt].agg(['mean', 'std'])
        grouped_df.plot(kind='bar', y='mean', yerr='std', ax=axes[i, 2], legend=False)
        # set x-axis tick labels
        axes[i, 2].set_xticklabels(['Miss', 'Hit'])
        axes[i, 2].set_ylim([50, 500])
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
        axes[i, 3].set_ylim([50, 500])
        axes[i, 3].set_ylabel('RT (ms)')
        axes[i, 3].set_title(f'Run 0{i+1} \n RT Plot \n (Removed NAs/Zeros: {RT_zero_na})')
        
        # create a bar plot of MRT by condition
        mrt_by_acc = non_zero_df.groupby(trial_accuracy)[mrt].mean()
        grouped_mrt = non_zero_df.groupby(cue_type)[mrt].mean()
        mrt_cond_plot = axes[i, 4].bar(grouped_mrt.index, grouped_mrt.values)
        axes[i, 4].axhline(mrt_no_nazero, color='r', linestyle='dashed')
        axes[i, 4].set_ylim([50, 500])
        axes[i, 4].set_ylabel('RT (ms)')
        axes[i, 4].set_title(f'Run 0{i+1} \n RT by Condition \n (Removed NAs/zeros: {RT_zero_na})')
        axes[i, 4].tick_params(axis='x', rotation=45, labelsize=10)
    
        # Create a dictionary to store the calculated values for this run
        data = {
            'Total Trials': total_trial_count,        
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
    mid_fig_combined.savefig(f"{out_dir}/sub-{sub}_ses-{ses}_task-${task}_plot-combined.png")
    
    # Writeout the .json file for each mid description
    with open(f"{out_dir}/sub-{sub}_ses-{ses}_task-${task}_beh-descr.json", 'w') as f:
        json.dump(mid_descr, f, indent=4)


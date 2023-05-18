import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="This script converts .png descriptions for MID into html file.")

# Add required positional arguments
parser.add_argument("-i", "--in_dir", type=str, required=True,
                    help="Input directory path where the output .png files exists for all subjects/sessions")
parser.add_argument("-o", "--out_dir", type=str, required=True,
                    help="Output directory where html file should be written")
parser.add_argument("-s", "--ses", type=str, required=True,
                    help="session argument (e.g. baselinearm1 or 2YearFollowUpYArm1) without ses prefix")
parser.add_argument("-t", "--task", type=str, required=True,
                    help="task argument (e.g. MID, SST, nback")


# Parse command-line arguments
args = parser.parse_args()

# Assign values to variables
in_dir = args.in_dir
out_dir = args.out_dir
ses = args.ses
task = args.task

# Define input directory
#in_dir = '/Users/michaeldemidenko/Downloads'
#out_dir = '/Users/michaeldemidenko/Downloads'
#ses = "2YearFollowUpYArm1"
#task = 'MID'


# Get a list of all files in the directory
files = os.listdir(in_dir)
files = [f for f in files if not f.startswith('._')]

# Filter the files to only include those that match the pattern
jsons = [f for f in files if 'sub-' in f and f'-{ses}_task-{task}_beh-descr' in f]
jsons = sorted(jsons, key=lambda f: (f.split('sub-')[1], f))

plots = [f for f in files if 'sub-' in f and f'-{ses}_task-{task}_plot' in f]
plots = sorted(plots, key=lambda f: (f.split('sub-')[1], f))

# Create empty df to calculate some estimates ACROSS ALL subjects
df = pd.DataFrame(columns=['Run','MeanRT', 'Accuracy',
                           'LgReward (%)','LgPun (%)','Triangle (%)',
                           'SmallReward (%)', 'SmallPun (%)'])

r1=0
r2=0
r1_miss=0
r2_miss=0

for j in jsons:
    # Open the file and load the JSON data
    with open(os.path.join(in_dir, j), 'r') as f:
        data = json.load(f)

    if 'Run 1' in data:
        r1=r1+1
        # Extract the Mean RT and Overall Accuracy for Run 1
        mean_rt_run1 = data['Run 1']['Mean RT']
        accuracy_run1 = data['Run 1']['Overall Accuracy']

        # extract accuracy by cond
        acc_cond_run1 = data['Run 1']['Accuracy by Cue Condition']

        # Add the extracted data to the pandas DataFrame
        df.loc[len(df)] = ['Run 1', mean_rt_run1, accuracy_run1,acc_cond_run1['LgReward'], acc_cond_run1['LgPun'], 
                           acc_cond_run1['Triangle'], acc_cond_run1['SmallReward'], acc_cond_run1['SmallPun']]
    else:
        r1_miss=r1_miss+1
        
    if 'Run 2' in data:
        r2=r2+1
        # Extract the Mean RT and Overall Accuracy for Run 2
        mean_rt_run2 = data['Run 2']['Mean RT']
        accuracy_run2 = data['Run 2']['Overall Accuracy']

        # extract accuracy by cond
        acc_cond_run2 = data['Run 2']['Accuracy by Cue Condition']
        
        # Add the extracted data to the pandas DataFrame
        df.loc[len(df)] = ['Run 2', mean_rt_run2, accuracy_run2, acc_cond_run2['LgReward'], acc_cond_run2['LgPun'], 
                           acc_cond_run2['Triangle'], acc_cond_run2['SmallReward'], acc_cond_run2['SmallPun']]
    else:
        r2_miss=r2_miss+1
    
# summarize the data and generate a figure
# convert to long
n = len(jsons)

df_long = pd.melt(df, id_vars=['Run'], 
                  value_vars=['MeanRT', 'Accuracy', 'LgReward (%)','LgPun (%)','Triangle (%)',
                  'SmallReward (%)', 'SmallPun (%)'], 
                  var_name='Measure', value_name='Value')



# Set style and background color
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create a figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 6))

# Plot the first subplot
sns.boxplot(x=df_long[df_long["Measure"].isin(['Accuracy', 'LgReward (%)','LgPun (%)','Triangle (%)',
                                               'SmallReward (%)', 'SmallPun (%)'])]['Run'], y=df_long["Value"], ax=ax1)
sns.stripplot(x=df_long[df_long["Measure"].isin(["Accuracy"])]['Run'], y=df_long["Value"], 
              color="orange", jitter=0.2, size=4, ax=ax1)
ax1.set_xlabel("Run")
ax1.set_ylabel("Accuracy %")
ax1.set_title(f'Accuracy for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')

# Plot the second subplot
sns.boxplot(x=df_long[df_long["Measure"].isin(["MeanRT"])]['Run'], y=df_long["Value"], ax=ax2)
sns.stripplot(x=df_long[df_long["Measure"].isin(["MeanRT"])]['Run'], y=df_long["Value"], color="orange", 
              jitter=0.2, size=4, ax=ax2)
ax2.set_xlabel("Run")
ax2.set_ylabel("RT (ms)")
ax2.set_title(f'Mean RT for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')

# plot accuracy by conditions
measures = ['LgReward (%)','LgPun (%)','Triangle (%)', 'SmallReward (%)', 'SmallPun (%)']
df_mean_sd = df_long[df_long["Measure"].isin(measures)].groupby(['Run', 'Measure']).agg(['mean', 'std'])
df_mean_sd.columns = ['_'.join(col).strip() for col in df_mean_sd.columns.values]

# Set color-blind friendly palette
colors = sns.color_palette('colorblind')
# Create bar plot with error bars
df_mean_sd['Value_mean'].unstack().plot(kind='bar', yerr=df_mean_sd['Value_std'].unstack(), ax=ax3, color=colors)

# Customize plot
ax3.set_xlabel("Run")
ax3.set_ylabel("Accuracy %")
ax3.set_title(f'Accuracy for Subjects across Conditions for \n Run 1 (N: {r1}) & Run 2 (N: {r2})')

# Move legend to top right and make it smaller
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False, fontsize=8)

# Make legend and y-axis label transparent
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False, fontsize=8)
ax3.yaxis.label.set_visible(False)


# save plot
fig.savefig(f"{out_dir}/subs-{n}_ses-{ses}_task-{task}_plot-averages.png")


# Create the beginning of the HTML output
html_output = '<html><head><title>Data Summary</title><style>td, th {font-family: Times New Roman;}</style></head><body>'
html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:50px;font-weight:bold;color:orange;">ABCD Study® \n MID Behavior</h2>'
html_output += '<h3 style="text-align:center;font-family:Times New Roman;font-size:40px;font-weight:bold;color:orange;">{0}</h3>'.format(ses)
# adding group average plot
avg_fig_name=f"subs-{n}_ses-{ses}_task-{task}_plot-averages.png"
text1=f"Summary of Subjects for Run 01 ({r1}) and Run 02 ({r2})."
text2=f"Missing Run 01: {r1_miss} & Run 02: {r2_miss}"
html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:26px;font-weight:bold;">{0}</h2>'.format(text1)
html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:20px;">{0}</h2>'.format(text2)
html_output += '<div style="text-align:center;"><img src="{0}" width="80%"/></div>'.format(os.path.join(out_dir, avg_fig_name))
# description 
html_output += '<div style="height: 20px;"></div>'
html_output += '<p style="font-family:Times New Roman;font-size:16px;">This report is a summary of the e-prime behavioral data for the Monetary Incentive Delay (MID) task from the ABCD study.</p>'
html_output += '<p style="font-family:Times New Roman;font-size:16px;">The Monetary Incentive Delay task is designed to measure cognitive and motivational processes related to reward anticipation and receipt.</p>'
html_output += '<p style="font-family:Times New Roman;font-size:16px;">The group-level summary includes extracted data from curated JSON files, displaying the accuracy and mean reaction time (RT) across runs, as well as the accuracy by cue conditions for each run.</p>'
html_output += '<p style="font-family:Times New Roman;font-size:16px;">Below the group-level summaries, you will find subject-specific run summaries. These include accuracies by condition, trial-wise accuracy, mean RT per hit/miss condition (when available), trial-wise RT, the number of feedback conditions, and the reward category alternation from the current trial (t) to the next trial (t+1).</p>'
html_output += '<div style="height: 20px;"></div>'
html_output += '<p style="font-family:Times New Roman;font-size:10px;">The data curation scripts were utilized to transform the eprime data into `_events.tsv` files, generate behavioral description JSONs/PNGs, and create the HTML report. You can find the scripts and additional information at the following GitHub repository: <a href="https://github.com/demidenm/abcc_datapre">Git: demidenm/abcc_datapre</a>.</p>'

html_output += '<hr/>' 
html_output += '<hr/>'
html_output += '<hr/>'

# Loop through each file in the input directory
for f in plots:

    # Extract the sub-[] and ses-[] information from the file names
    sub = f.split('sub-')[1].split('_')[0]
    ses = f.split('ses-')[1].split('_')[0]

    # adding sub + ses to labels
    html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:30px;font-weight:bold;">Subject: {0}</h2><h3 style="text-align:center;font-family:Times New Roman;font-size:16px;">Session: {1}</h3>'.format(sub.upper(), ses)

    # adding image
    html_output += '<div style="text-align:center;"><img src="../{0}_{1}/{2}" width="80%"/></div>'.format(ses,task,f)
    #html_output += '<div style="text-align:center;"><img src="./{0}" width="80%"/></div>'.format(f)

    # Add three lines to separate the subjects
    html_output += '<hr/>'
    html_output += '<hr/>'
    html_output += '<hr/>'

# Add the end of the HTML output
html_output += '</body></html>'

# Write the HTML output to a file
with open(f'{out_dir}/describe_ses-{ses}_task-{task}.html', 'w') as f:
    f.write(html_output)
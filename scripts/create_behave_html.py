import os
import argparse
import json
import numpy as np
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
#ses = "baselineYear1Arm1"
#task = 'nback'


# Get a list of all files in the directory
files = os.listdir(in_dir)
files = [f for f in files if not f.startswith('._')]

# Filter the files to only include those that match the pattern
jsons = [f for f in files if 'sub-' in f and f'-{ses}_task-{task}_beh-descr' in f]
jsons = sorted(jsons, key=lambda f: (f.split('sub-')[1], f))

plots = [f for f in files if 'sub-' in f and f'-{ses}_task-{task}_plot' in f]
plots = sorted(plots, key=lambda f: (f.split('sub-')[1], f))

if task == "MID":
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
            df.loc[len(df)] = ['Run 1',
                       mean_rt_run1,
                       accuracy_run1,
                       acc_cond_run1.get('LgReward', np.nan),
                       acc_cond_run1.get('LgPun', np.nan),
                       acc_cond_run1.get('Triangle', np.nan),
                       acc_cond_run1.get('SmallReward', np.nan),
                       acc_cond_run1.get('SmallPun', np.nan)]
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
            df.loc[len(df)] = ['Run 2',
                       mean_rt_run2,
                       accuracy_run2,
                       acc_cond_run2.get('LgReward', np.nan),
                       acc_cond_run2.get('LgPun', np.nan),
                       acc_cond_run2.get('Triangle', np.nan),
                       acc_cond_run2.get('SmallReward', np.nan),
                       acc_cond_run2.get('SmallPun', np.nan)]
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
    filtered_df = df_long[df_long["Measure"].isin(['Accuracy', 'LgReward (%)','LgPun (%)','Triangle (%)',
                                                   'SmallReward (%)', 'SmallPun (%)'])].dropna(subset=["Run", "Value"])

    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x=filtered_df['Run'], y=filtered_df["Value"], ax=ax1)
    sns.stripplot(x=filtered_df[filtered_df["Measure"] == "Accuracy"]['Run'], y=filtered_df[filtered_df["Measure"] == "Accuracy"]["Value"],
                  color="orange", jitter=0.2, size=4, ax=ax1)

    # Set the labels and title for the plot
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
    mean_accuracy = df_long[df_long['Measure'] == 'Accuracy'].groupby('Run')['Value'].mean()
    run1_mean_acc = round(mean_accuracy[0]*100, 1)
    run2_mean_acc = round(mean_accuracy[1]*100, 1)
    # Create bar plot with error bars
    df_mean_sd['Value_mean'].unstack().plot(kind='bar', yerr=df_mean_sd['Value_std'].unstack(), ax=ax3, color=colors)

    # Customize plot
    ax3.axhline(.60, linestyle='solid', color='black')
    ax3.set_xlabel("Run")
    ax3.set_ylabel("Accuracy %")
    ax3.set_title(f'Accuracy for Subjects across Conditions for \n Run 1 (N: {r1}, {run1_mean_acc}%) & Run 2 (N: {r2}; {run1_mean_acc}%) \n Black line = Target Accuracy')

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
    html_output += '<div style="text-align:center;"><img src="./{0}" width="80%"/></div>'.format(avg_fig_name)
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
        
elif task == "SST":

    # Create empty df to calculate some estimates ACROSS ALL subjects
    df = pd.DataFrame(columns=['Run','Go Acc (%)', 'Go Mean RT',
                               'Stop Sig Acc (%)','Stop Sig Fail MRT','SSRT',
                               'SSD Delay Min', 'SSD Delay Max'])
    
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
            # Extract variables for Run 1
            go_acc_run1 = data['Run 1']['Go Accuracy']
            go_rt_run1 = data['Run 1']['Go MRT']
            stopsig_acc_run1 = data['Run 1']['Stop Signal Accuracy']
            stopsigfail_mrt_run1 = data['Run 1']['Stop Signal MRT']
            ssrt_run1 = data['Run 1']['Stop Signal Reaction Time']
            ssddur_mix_run1 = data['Run 1']['Stop Signal Delay Durations']["min"]
            ssddur_max_run1 = data['Run 1']['Stop Signal Delay Durations']["max"]
    
            # Add the extracted data to the pandas DataFrame
            df.loc[len(df)] = ['Run 1',
                       go_acc_run1,
                       go_rt_run1,
                       stopsig_acc_run1,
                       stopsigfail_mrt_run1,
                       ssrt_run1,
                       ssddur_mix_run1,
                       ssddur_max_run1]
        else:
            r1_miss=r1_miss+1
            
        if 'Run 2' in data:
            r2=r2+1
            # Extract variables for Run 2
            go_acc_run2 = data['Run 2']['Go Accuracy']
            go_rt_run2 = data['Run 2']['Go MRT']
            stopsig_acc_run2 = data['Run 2']['Stop Signal Accuracy']
            stopsigfail_mrt_run2 = data['Run 2']['Stop Signal MRT']
            ssrt_run2 = data['Run 2']['Stop Signal Reaction Time']
            ssddur_mix_run2 = data['Run 2']['Stop Signal Delay Durations']["min"]
            ssddur_max_run2 = data['Run 2']['Stop Signal Delay Durations']["max"]
    
            # Add the extracted data to the pandas DataFrame
            df.loc[len(df)] = ['Run 2',
                       go_acc_run2,
                       go_rt_run2,
                       stopsig_acc_run2,
                       stopsigfail_mrt_run2,
                       ssrt_run2,
                       ssddur_mix_run2,
                       ssddur_max_run2]
        else:
            r2_miss=r2_miss+1
        
    # summarize the data and generate a figure
    # convert to long
    n = len(jsons)
    
    df_long = pd.melt(df, id_vars=['Run'], 
                      value_vars=['Go Acc (%)', 'Go Mean RT',
                                  'Stop Sig Acc (%)','Stop Sig Fail MRT','SSRT',
                                  'SSD Delay Min', 'SSD Delay Max'], 
                      var_name='Measure', value_name='Value')
    
    
    # Plotting
    # Set style and background color
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 10))
    
    # Plot 1
    # filter data
    filtered_acc_df = df_long[df_long["Measure"].isin(['Go Acc (%)','Stop Sig Acc (%)'])].dropna(subset=["Run", "Value"])
    
    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x="Run", y="Value", hue="Measure", data=filtered_acc_df, ax=ax1)
    # Set the labels and title for the plot
    ax1.set_ylim([0, 1.2])
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Accuracy %")
    ax1.set_title(f'Accuracy for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Type", framealpha=0.7)
    legend = ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), title=None, 
                        framealpha=1, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # Plot 2
    # filter data
    filtered_rt_df = df_long[df_long["Measure"].isin(['Go Mean RT','Stop Sig Fail MRT', 'SSRT'])].dropna(subset=["Run", "Value"])
    
    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x="Run", y="Value", hue="Measure", data=filtered_rt_df, ax=ax2)
    # Set the labels and title for the plot
    ax2.set_ylim([-500, 1000])
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Mean RT (ms)")
    ax2.set_title(f'Mean RTs for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Type", framealpha=0.7)
    legend = ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), title=None, 
                        framealpha=1, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # Plot 3
    # filter data
    filtered_ssd_df = df_long[df_long["Measure"].isin(['SSD Delay Min','SSD Delay Max'])].dropna(subset=["Run", "Value"])
    
    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x="Run", y="Value", hue="Measure", data=filtered_ssd_df, ax=ax3)
    # Set the labels and title for the plot
    ax3.set_ylim([-250, 1000])
    ax3.set_xlabel("Run")
    ax3.set_ylabel("SSD Delay (ms)")
    ax3.set_title(f'SSD Delay (Min/Max) for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title=None, framealpha=0.7)
    legend = ax3.legend(loc='upper right', bbox_to_anchor=(1, 1), title=None, 
                        framealpha=1, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # save plot
    fig.savefig(f"{out_dir}/subs-{n}_ses-{ses}_task-{task}_plot-averages.png")
    
    
    # Create the beginning of the HTML output
    html_output = '<html><head><title>Data Summary</title><style>td, th {font-family: Times New Roman;}</style></head><body>'
    html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:50px;font-weight:bold;color:orange;">ABCD Study® \n Stop Signal Task Behavior</h2>'
    html_output += '<h3 style="text-align:center;font-family:Times New Roman;font-size:40px;font-weight:bold;color:orange;">{0}</h3>'.format(ses)
    # adding group average plot
    avg_fig_name=f"subs-{n}_ses-{ses}_task-{task}_plot-averages.png"
    text1=f"Summary of Subjects for Run 01 ({r1}) and Run 02 ({r2})."
    text2=f"Missing Run 01: {r1_miss} & Run 02: {r2_miss}"
    html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:26px;font-weight:bold;">{0}</h2>'.format(text1)
    html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:20px;">{0}</h2>'.format(text2)
    html_output += '<div style="text-align:center;"><img src="./{0}" width="80%"/></div>'.format(avg_fig_name)
    # description 
    html_output += '<div style="height: 20px;"></div>'
    html_output += '<p style="font-family:Times New Roman;font-size:16px;">This report is a summary of the e-prime behavioral data for the Stop Signal Task (SST) task from the ABCD study.</p>'
    html_output += '<p style="font-family:Times New Roman;font-size:16px;">The SST is design to measure cognitive control, specifically, inhibitory control. The variable of interest is the Stop Signal Response Time (SSRT), which is the inhibition of the motor response following a Go Signal when the Stop Signal Appears.</p>'
    html_output += '<p style="font-family:Times New Roman;font-size:16px;">The group-level summary includes extracted data from curated JSON files, displaying the accuracy, mean reaction time (RT) and Stop Signal Delay Durations (SSDDur) min/max values across runs.</p>'
    html_output += '<p style="font-family:Times New Roman;font-size:16px;">Below the group-level summaries, you will find subject-specific run summaries.</p>'
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
        
        
elif task == "nback":
    # Create empty df to calculate some estimates ACROSS ALL subjects
    df = pd.DataFrame(columns=['Run',
                               'Oveall Acc','0-Back Acc', '2-Back Acc',
                               'Neg Face Acc','Neut Face Acc','Pos Face Acc', 'Place Acc',
                               'Oveall MRT','0-Back MRT', '2-Back MRT',
                               'Neg Face MRT','Neut Face MRT','Pos Face MRT', 'Place MRT',
                               '0-Back Dprime', '2-Back Dprime'])
    
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
            # Extract variables for Run 1
            n0back_acc_run1 = data['Run 1']['Block Accuracy']['0-Back'] 
            n2back_acc_run1 = data['Run 1']['Block Accuracy']['2-Back']
            negface_acc_run1 = data['Run 1']['Stimulus Accuracy']['NegFace'] 
            neutface_acc_run1 = data['Run 1']['Stimulus Accuracy']['NeutFace'] 
            posface_acc_run1 = data['Run 1']['Stimulus Accuracy']['PosFace']
            place_acc_run1 = data['Run 1']['Stimulus Accuracy']['Place']
            overall_acc_run1 = data['Run 1']['Overall Accuracy']
            
            # extract MRTs for Run 1
            n0back_mrt_run1 = data['Run 1']['Block Mean RT']['0-Back'] 
            n2back_mrt_run1 = data['Run 1']['Block Mean RT']['2-Back']
            negface_mrt_run1 = data['Run 1']['Stimulus Mean RT']['NegFace'] 
            neutface_mrt_run1 = data['Run 1']['Stimulus Mean RT']['NeutFace'] 
            posface_mrt_run1 = data['Run 1']['Stimulus Mean RT']['PosFace']
            place_mrt_run1 = data['Run 1']['Stimulus Mean RT']['Place'] 
            overall_mrt_run1 = data['Run 1']['Overall Mean RT']
            
            #dprime vars for run
            n0back_dprime_run1 = data['Run 1']['D-prime']['0back']
            n2back_dprime_run1 = data['Run 1']['D-prime']['2back']
                                                         
    
            # Add the extracted data to the pandas DataFrame
            df.loc[len(df)] = ['Run 1',
                       overall_acc_run1,
                       n0back_acc_run1, n2back_acc_run1,
                       negface_acc_run1, neutface_acc_run1, posface_acc_run1, place_acc_run1,
                       overall_mrt_run1,
                       n0back_mrt_run1,n2back_mrt_run1,
                       negface_mrt_run1,neutface_mrt_run1,posface_mrt_run1,place_mrt_run1,
                       n0back_dprime_run1, n2back_dprime_run1
                       ]
        else:
            r1_miss=r1_miss+1
            
        if 'Run 2' in data:
            r2=r2+1
            # Extract variables for Run 2
            n0back_acc_run2 = data['Run 2']['Block Accuracy']['0-Back'] 
            n2back_acc_run2 = data['Run 2']['Block Accuracy']['2-Back']
            negface_acc_run2 = data['Run 2']['Stimulus Accuracy']['NegFace'] 
            neutface_acc_run2 = data['Run 2']['Stimulus Accuracy']['NeutFace'] 
            posface_acc_run2 = data['Run 2']['Stimulus Accuracy']['PosFace']
            place_acc_run2 = data['Run 2']['Stimulus Accuracy']['Place']
            overall_acc_run2 = data['Run 2']['Overall Accuracy']
            
            # extract MRTs for Run 2
            n0back_mrt_run2 = data['Run 2']['Block Mean RT']['0-Back'] 
            n2back_mrt_run2 = data['Run 2']['Block Mean RT']['2-Back']
            negface_mrt_run2 = data['Run 2']['Stimulus Mean RT']['NegFace'] 
            neutface_mrt_run2 = data['Run 2']['Stimulus Mean RT']['NeutFace'] 
            posface_mrt_run2 = data['Run 2']['Stimulus Mean RT']['PosFace']
            place_mrt_run2 = data['Run 2']['Stimulus Mean RT']['Place'] 
            overall_mrt_run2 = data['Run 2']['Overall Mean RT']
            
            #dprime vars for run
            n0back_dprime_run2 = data['Run 2']['D-prime']['0back']
            n2back_dprime_run2 = data['Run 2']['D-prime']['2back']
                                                         
    
            # Add the extracted data to the pandas DataFrame
            df.loc[len(df)] = ['Run 2',
                       overall_acc_run2,
                       n0back_acc_run2, n2back_acc_run2,
                       negface_acc_run2, neutface_acc_run2, posface_acc_run2, place_acc_run2,
                       overall_mrt_run2,
                       n0back_mrt_run2,n2back_mrt_run2,
                       negface_mrt_run2,neutface_mrt_run2,posface_mrt_run2,place_mrt_run2,
                       n0back_dprime_run2, n2back_dprime_run2
                       ]
                       
        else:
            r2_miss=r2_miss+1
        
    # summarize the data and generate a figure
    # convert to long
    n = len(jsons)
    
    df_long = pd.melt(df, id_vars=['Run'], 
                      value_vars=['Oveall Acc','0-Back Acc', '2-Back Acc',
                                  'Neg Face Acc','Neut Face Acc','Pos Face Acc', 'Place Acc',
                                  'Oveall MRT','0-Back MRT', '2-Back MRT',
                                  'Neg Face MRT','Neut Face MRT','Pos Face MRT', 'Place MRT',
                                  '0-Back Dprime', '2-Back Dprime'], 
                      var_name='Measure', value_name='Value')
    
    
    # Plotting
    # Set style and background color
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # Create a figure with two subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, figsize=(30, 15)) 
    
    # Plot 1
    # filter data
    filtered_nback_acc_df = df_long[df_long["Measure"].isin(['Overall Acc',
                                                             '0-Back Acc','2-Back Acc'])].dropna(subset=["Run", "Value"])
    
    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x="Run", y="Value", hue="Measure", data=filtered_nback_acc_df, ax=ax1)
    # Set the labels and title for the plot
    ax1.set_ylim([0, 1.2])
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Accuracy %")
    ax1.set_title(f'Block Type Accuracy for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Type", framealpha=0.7)
    legend = ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), title=None, 
                        framealpha=1, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # Plot 2
    # filter data
    filtered_stim_acc_df = df_long[df_long["Measure"].isin(['Overall Acc', 'Neg Face Acc',
                                                             'Neut Face Acc','Pos Face Acc', 
                                                             'Place Acc'])].dropna(subset=["Run", "Value"])
    
    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x="Run", y="Value", hue="Measure", data=filtered_stim_acc_df, ax=ax2)
    # Set the labels and title for the plot
    ax2.set_ylim([0, 1.2])
    ax2.set_xlabel("Run")
    ax2.set_ylabel("Accuracy %")
    ax2.set_title(f'Stimulus Type Accuracy for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Type", framealpha=0.7)
    legend = ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), title=None, 
                        framealpha=1, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # Plot 3
    # filter data
    filtered_nback_mrt_df = df_long[df_long["Measure"].isin(['Overall MRT',
                                                             '0-Back MRT','2-Back MRT'])].dropna(subset=["Run", "Value"])
    
    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x="Run", y="Value", hue="Measure", data=filtered_nback_mrt_df, ax=ax3)
    # Set the labels and title for the plot
    ax3.set_ylim([0, 2000])
    ax3.set_xlabel("Run")
    ax3.set_ylabel("mean RT (ms)")
    ax3.set_title(f'Block Mean RT for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Type", framealpha=0.7)
    legend = ax3.legend(loc='upper right', bbox_to_anchor=(1, 1), title=None, 
                        framealpha=1, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # Plot 4
    # filter data
    filtered_stim_mrt_df = df_long[df_long["Measure"].isin(['Overall MRT', 'Neg Face MRT',
                                                             'Neut Face MRT','Pos Face MRT', 
                                                             'Place MRT'])].dropna(subset=["Run", "Value"])
    
    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x="Run", y="Value", hue="Measure", data=filtered_stim_mrt_df, ax=ax4)
    # Set the labels and title for the plot
    ax4.set_ylim([0, 2000])
    ax4.set_xlabel("Run")
    ax4.set_ylabel("Accuracy %")
    ax4.set_title(f'Stimulus Type Mean RT for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Type", framealpha=0.7)
    legend = ax4.legend(loc='upper right', bbox_to_anchor=(1, 1), title=None, 
                        framealpha=1, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # Plot 5
    # filter data
    filtered_dprime_df = df_long[df_long["Measure"].isin(['0-Back Dprime', '2-Back Dprime'])].dropna(subset=["Run", "Value"])
    
    # Create the boxplot and stripplot using the filtered data
    sns.boxplot(x="Run", y="Value", hue="Measure", data=filtered_dprime_df, ax=ax5)
    # Set the labels and title for the plot
    ax5.set_ylim([-10, 10])
    ax5.set_xlabel("Run")
    ax5.set_ylabel("Accuracy %")
    ax5.set_title(f'D-prime for Subjects across \n Run 1 (N: {r1}) & Run 2 (N: {r2})')
    ax5.legend(loc='upper right', bbox_to_anchor=(1.25, 1), title="Type", framealpha=0.7)
    legend = ax5.legend(loc='upper right', bbox_to_anchor=(1, 1), title=None, 
                        framealpha=1, edgecolor='black')
    legend.get_frame().set_facecolor('white')
    
    # save plot
    fig.savefig(f"{out_dir}/subs-{n}_ses-{ses}_task-{task}_plot-averages.png")
    
    
    # Create the beginning of the HTML output
    html_output = '<html><head><title>Data Summary</title><style>td, th {font-family: Times New Roman;}</style></head><body>'
    html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:50px;font-weight:bold;color:orange;">ABCD Study® \n Emotional N-Back Task</h2>'
    html_output += '<h3 style="text-align:center;font-family:Times New Roman;font-size:40px;font-weight:bold;color:orange;">{0}</h3>'.format(ses)
    # adding group average plot
    avg_fig_name=f"subs-{n}_ses-{ses}_task-{task}_plot-averages.png"
    text1=f"Summary of Subjects for Run 01 ({r1}) and Run 02 ({r2})."
    text2=f"Missing Run 01: {r1_miss} & Run 02: {r2_miss}"
    html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:26px;font-weight:bold;">{0}</h2>'.format(text1)
    html_output += '<h2 style="text-align:center;font-family:Times New Roman;font-size:20px;">{0}</h2>'.format(text2)
    html_output += '<div style="text-align:center;"><img src="./{0}" width="80%"/></div>'.format(avg_fig_name)
    # description 
    html_output += '<div style="height: 20px;"></div>'
    html_output += '<p style="font-family:Times New Roman;font-size:16px;">This report is a summary of the e-prime behavioral data for the Emotional Nback (NBack) task from the ABCD study.</p>'
    html_output += '<p style="font-family:Times New Roman;font-size:16px;">The Emotional Nback is design to measure affective and working memory load. The variable of interest is the D-prime.</p>'
    html_output += '<p style="font-family:Times New Roman;font-size:16px;">The group-level summary includes extracted data from curated JSON files, displaying the accuracy, mean reaction time (RT) and D-prime per run.</p>'
    html_output += '<p style="font-family:Times New Roman;font-size:16px;">Below the group-level summaries, you will find subject-specific run summaries.</p>'
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
     

     
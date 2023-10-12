import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="Checks initial onset values and DiffTriggerTimes in events.tsvs")

# Add required positional arguments
parser.add_argument("-i", "--in_dir", type=str, required=True,
                    help="Input directory path where the events files are exported")
parser.add_argument("-o", "--out_dir", type=str, required=True,
                    help="Output directory path where the results should go")
parser.add_argument("-t", "--task", type=str, required=True,
                    help="task argument (e.g. MID, SST, nback)")
# Parse command-line arguments
args = parser.parse_args()

# Assign values to variables
in_dir = args.in_dir
out_dir = args.out_dir
task = args.task



# Assign col name based on task label & create an empty list to store the data
if task == 'MID':
    loc_initial_onset = 'Cue.OnsetTime'
elif task == 'SST':
    loc_initial_onset = 'BeginFix.OnsetTime'
elif task == 'nback':
    loc_initial_onset = 'CueFix.OnsetTime'

diff_trigg = 'DiffTriggerTimes'
data = []


# Loop through all the files in the directory
print(f'Step 1: Extracting onset times from files for {task}')

for filename in os.listdir(in_dir):
    
    # Check if the filename matches the pattern
    if 'sub-' in filename and '_ses-' in filename and f'_task-{task}_run-' in filename and '_events.tsv' in filename:
        
        # Extract the sub, ses, and run numbers from the filename
        sub = filename.split('_')[0][4:]
        ses = filename.split('_')[1][4:]
        run = filename.split('_')[3][3:]
        
        # Create the new filename
        new_filename = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv"
        
        # Read the file into a dataframe
        df = pd.read_csv(os.path.join(in_dir, filename), sep='\t')
        
        # Extract the first value of Cue.OnsetTime and DiffTriggerTimes
        try:
            cue_onset_time = df.loc[0, loc_initial_onset]
        except KeyError:
            cue_onset_time = 'NA'
        
        diff_trigger_times = df.loc[0, diff_trigg]
        
        # Append the data to the list
        data.append((new_filename, cue_onset_time, diff_trigger_times))

# Convert the list to a dataframe & save
print(f'Step 2: Creating DF w/ values and saving in output folder as: \n \t \t ses-{ses}_task-{task}_events-onset_distribution.png')

df = pd.DataFrame(data, columns=['filename', 'TaskOnset', 'DiffTriggerTimes'])
df.to_csv(f'{out_dir}/ses-{ses}_task-{task}_events-timing-info.csv', index=False)


# plot and save distribution as .png for non-na values
print(f'Step 3: Creating distribution plot from DF and saving to: \n \t \t {out_dir}')

na_count = df['TaskOnset'].isna().sum()
print(f'\t \t {na_count} NA values in data which are excluded from plot')

non_na_values = pd.to_numeric(df['TaskOnset'], errors='coerce').dropna()
non_na_values.plot(kind='hist', bins=100)  # Plotting a histogram with 20 bins
plt.xlabel('Cue Onset Time')  # Adding x-axis label
plt.ylabel('Frequency')  # Adding y-axis label
plt.title(f'Session {ses} Distribution of {task} Onset Times')
plt.savefig(f'{out_dir}/ses-{ses}_task-{task}_events-onset_distribution.png')
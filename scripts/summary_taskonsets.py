import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Set the directory where the files are located
#in_dir = '/Users/michaeldemidenko/Downloads'
#out_dir = '/Users/michaeldemidenko/Downloads'
#task = 'MID'


# Create ArgumentParser object
parser = argparse.ArgumentParser(description="Checks initial onset values and DiffTriggerTimes in events.tsvs")

# Add required positional arguments
parser.add_argument("-i", "--in_dir", type=str, required=True,
                    help="Input directory path where the events file is within */func/")
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
for filename in os.listdir(in_dir):
    
    # Check if the filename matches the pattern
    if 'sub-' in filename and '_ses-' in filename and f'_task-{task}_run' in filename and '_events.tsv' in filename:
        
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

# Convert the list to a dataframe
df = pd.DataFrame(data, columns=['filename', 'Cue.OnsetTime', 'DiffTriggerTimes'])


# plot and save distribution
df['Cue.OnsetTime'].plot(kind='hist', bins=150)  # Plotting a histogram with 20 bins
plt.xlabel('Cue Onset Time')  # Adding x-axis label
plt.ylabel('Frequency')  # Adding y-axis label
plt.title(f'Distribution of {task} Onset Times')

# Save the dataframe to a CSV file ^ figure as .png
df.to_csv(f'{out_dir}/ses-{ses}_task-{task}_events-timing-info.csv', index=False)
plt.savefig(f'{out_dir}/ses-{ses}_task-{task}_events-onset_distribution.png')
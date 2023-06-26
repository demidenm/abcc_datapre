import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict

"""
Script created by Michael Demidenko to convert e-prime data for ABCD study SST, MID and nback tasks. 
Edge cases are considered given differences across eprime outputs and scanners - some errors may still exist.
The script uses a parser to differentiate tasks and runs labels for the file path. 
The file path is constructed using the format:
filepath = f"{in_dir}/sub-{sub}_ses-{ses}_task-{task}_run-{run}_bold_EventRelatedInformation.txt"

Each scanner has an onset time for the scanner and the task itself. A differences is calculated and
saved to each *events.tsv file "DiffTriggerTimes". The onsets and durations for each task are 
from the initiation of the task (e.g., after calibration volumes). Thus, to adjust to onset of scanner add
to Onsets/Durations the DiffTriggerTimes column. 

Note: For the SST task and the MID task, RTs/meanrts/SSDDur/StopSignal/Duration 
are left in miliseconds and note converted, as in other onset/durations to seconds (e.g. divided by 1000)
Most recent edit: June  25, 2023.

"""

# testing
#in_dir = '/Users/michaeldemidenko/Downloads'
#out_dir = '/Users/michaeldemidenko/Downloads'
#sub = '###'
#ses = 'baselineYear1Arm1'
#run = '01'
#task = 'nback'


# Using Taylor Taslo's convert txt to df code from github: tsalo/convert-eprime/ as could not use from package. 
# Credit for lines #35 to #99 should be all given to Taylor.
def remove_unicode(string):
    """
    Removes unicode characters in string.
    """
    return ''.join([val for val in string if 31 < ord(val) < 127])

def text_to_df(text_file):
    """
    Convert a raw E-Prime output text file into a pandas DataFrame.
    """
    # Load the text file as a list.
    with open(text_file, 'rb') as fo:
        text_data = list(fo)

    # Remove unicode characters.
    filtered_data = [remove_unicode(row.decode('utf-8', 'ignore')) for row in text_data]

    # Determine where rows begin and end.
    start_index = [i for i, row in enumerate(filtered_data) if row == '*** LogFrame Start ***']
    end_index = [i for i, row in enumerate(filtered_data) if row == '*** LogFrame End ***']
    if len(start_index) != len(end_index) or start_index[0] >= end_index[0]:
        print('Warning: LogFrame Starts and Ends do not match up.',
              'Including header metadata just in case.')
        # In cases of an experiment crash, the final LogFrame is never written, and the experiment metadata
        # (Subject, VersionNumber, etc.) isn't collected by the indices above. We can manually include the
        # metadata-containing Header Frame to collect these data from a partial-run crash dump.
        start_index = [i for i,row in enumerate(filtered_data) if row == '*** Header Start ***'] + start_index
        end_index = [i for i,row in enumerate(filtered_data) if row == '*** Header End ***'] + end_index
    n_rows = min(len(start_index), len(end_index))

    # Find column headers and remove duplicates.
    headers = []
    data_by_rows = []
    for i in range(n_rows):
        one_row = filtered_data[start_index[i]+1:end_index[i]]
        data_by_rows.append(one_row)
        for col_val in one_row:
            split_header_idx = col_val.index(':')
            headers.append(col_val[:split_header_idx])

    headers = list(OrderedDict.fromkeys(headers))

    # Preallocate list of lists composed of NULLs.
    data_matrix = np.empty((n_rows, len(headers)), dtype=object)
    data_matrix[:] = np.nan

    # Fill list of lists with relevant data from data_by_rows and headers.
    for i in range(n_rows):
        for cell_data in data_by_rows[i]:
            split_header_idx = cell_data.index(':')
            for k_header, header in enumerate(headers):
                if cell_data[:split_header_idx] == header:
                    data_matrix[i, k_header] = cell_data[split_header_idx+1:].lstrip()

    df = pd.DataFrame(columns=headers, data=data_matrix)

    # Columns with one value at the beginning, the end, or end - 1 should be
    # filled with that value.
    for col in df.columns:
        non_nan_idx = np.where(df[col].values == df[col].values)[0]
        if len(non_nan_idx) == 1 and non_nan_idx[0] in [0, df.shape[0]-1,
                                                        df.shape[0]-2]:
            df.loc[:, col] = df.loc[non_nan_idx[0], col]
    return df

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


parser.add_argument("-i", "--in_dir", type=str, required=True,
                    help="Input directory path where the events file is within */func/")
parser.add_argument("-o", "--out_dir", type=str, required=True,
                    help="Output directory path where the results should go")
parser.add_argument("-s", "--sub", type=str, required=True,
                    help="subject argument (e.g. NVDAFL737P) without sub prefix")
parser.add_argument("-e", "--ses", type=str, required=True,
                    help="session argument (e.g. baselinearm1) without ses prefix")
parser.add_argument("-r", "--run", type=str, required=True,
                    help="run argument (e.g. 01, 02) with run prefix")
parser.add_argument("-t", "--task", type=str, required=True,
                    help="task argument (e.g. MID, SST, nback)")

args = parser.parse_args()

# Assign values to variables
in_dir = args.in_dir
out_dir = args.out_dir
sub = args.sub
ses = args.ses
run = args.run
task = args.task


# Setting up the file path
filepath = f"{in_dir}/sub-{sub}_ses-{ses}_task-{task}_run-{run}_bold_EventRelatedInformation.txt"

# some files are written in utf-16 as opposed to most using uts-8. Using try and except rule to circumvent errors
# this, too, applies to checking the type of edge case errors when reading in files.
try:
    try:
        dat = pd.read_csv(filepath, nrows=3)
    except pd.errors.ParserError:
        dat = pd.read_csv(filepath, nrows=1)
        
    # check for edat2 in row 1
    if ".edat2" in dat.columns[0]:
        # check for 2nd row with edited data.. remove if label exist need to remove redudant rows
        if "edited data" in dat.iloc[0,0]: 
            dat = pd.read_csv(filepath, skiprows=2, sep="\t")
        else:
            dat = pd.read_csv(filepath, skiprows=1, sep="\t")
    elif "Header Start" in dat.columns[0]:
        dat = text_to_df(filepath)
    else:
        dat = pd.read_csv(filepath, sep = "\t")

# If UTF-8 encoding fails, try reading the file with UTF-16 encoding
except UnicodeDecodeError:
    try:
        dat = pd.read_csv(filepath, encoding='utf-16', nrows = 3)
    except pd.errors.ParserError:
        dat = pd.read_csv(filepath, encoding='utf-16', nrows = 1)
    # Chech for .edat2 in row1
    if ".edat2" in dat.columns[0]:
        if "edited data" in dat.iloc[0,0]:            
            dat = pd.read_csv(filepath, encoding='utf-16', skiprows=2, sep="\t")
        else:
            dat = pd.read_csv(filepath, encoding='utf-16', skiprows=1, sep="\t")
    elif "Header Start" in dat.columns[0]:
        dat = text_to_df(filepath)
    else:
        dat = pd.read_csv(filepath, encoding='utf-16', sep = "\t")
    
# assigned subdject ID column from NARGUID
dat['Subject'] = dat['NARGUID']


# Proceed to determine which task preprocessing to do: MID, SST, nback

if task == "MID":
    # creating run and trial labels using try/except
    try:
        dat['Run'] = dat['Block']
    except KeyError:
        dat['Run'] = dat['RunList.Cycle']
        dat['Run'] = dat['Run'].fillna(dat['Waiting4Scanner.Cycle'].combine_first(dat['PeriodList.Cycle']))
        dat['SubTrial'] = dat['RunList.Sample']

    # specify columns that have the start of scanner time (including initial volumes: GetReady; not calib volumes: PrepTime)
    prep_var = "PrepTime.OffsetTime"
    ready_var = "GetReady.RTTime"

    # Save the values for preptime and getready time (includes volume prep alt)
    prep_per_run = [lst[-1] for lst in (dat.groupby("Run")[prep_var]
                                .apply(lambda lst: [value for value in lst if pd.notna(value)]).tolist()) if lst]
    ready_per_run = [lst[0] for lst in (dat.groupby("Run")[ready_var]
                                .apply(lambda lst: [value for value in lst if pd.notna(value)]).tolist()) if lst]

    # remove NA subtrial columns
    dat = dat[~dat['SubTrial'].isna()]
    
    # get unique runs for loop
    uniq_runs = dat['Run'].unique().astype('int')
    
    for r in uniq_runs:
        # selection the first instance over GetReady and PrepTime for each run (see grouping above)
        try:
            dat['TriggerTime'] = np.tile(ready_per_run[r-1], len(dat))
            dat['TriggerTimeAlt'] = np.tile(prep_per_run[r-1], len(dat))
        except (IndexError, TypeError):
            dat['TriggerTime'] = np.tile(ready_per_run[0][r-1], len(dat))
            dat['TriggerTimeAlt'] = np.tile(prep_per_run[0][r-1], len(dat))
            
        # before conversion, convert values to numeric to ensure not objects 
        df = dat
        df_subset = convert_to_numeric(df)
        df_subset = df[df['Run'] == r]
        
        
        # keep column names for the output behavioral files, modify as needed
        keep_cols = ["Subject","Handedness","Run","SubTrial","Condition",
                     "Cue.OnsetTime","Cue.Duration",
                     "Anticipation.Duration","Anticipation.OnsetTime",
                     "Probe.Duration","Probe.OnsetTime","Probe.RESP",
                     "Result","prbacc","prbrt","OverallRT","meanrt",
                     "moneyamt","ResponseCheck",
                     "Feedback.OnsetTime","FeedbackDuration",
                     "SessionDate","TriggerTime","TriggerTimeAlt"]
        
        cols_to_keep = [col for col in keep_cols if col in df_subset.columns]
        df_subset = df_subset[cols_to_keep]


        # Converting ms to seconds; ONLY [onset times] subtract trigger time
        time_subtract = ["Cue.OnsetTime","Anticipation.OnsetTime","Probe.OnsetTime","Feedback.OnsetTime"]
        
        duration_subtract = ["Cue.Duration","Anticipation.Duration","Probe.Duration",
                            "FeedbackDuration"] # leaving RT in ms
        
        for col_time in time_subtract:
            df_subset[col_time] = (df_subset[col_time] - df_subset['TriggerTimeAlt'])/1000
            
        for dur_time in duration_subtract:
            df_subset[dur_time] = df_subset[dur_time]/1000
        
        
        # create diff between scanner start and task start
        df_subset["DiffTriggerTimes"] = (df_subset["TriggerTimeAlt"]-df_subset["TriggerTime"])/1000

        
        # writeout .tsv per run
        df_subset.to_csv(f"{out_dir}/sub-{sub}_ses-{ses}_task-MID_run-0{r}_events.tsv", sep = '\t', index = False)
    
elif task == "SST":
    # creating run and trial labels using try/except. Some eprime files are written oddly, so this info isn't always avail
    # need to create these columns to use in the group for select for readyruns + running by run
    try:
        unique_runs = dat['Trial'].unique().astype('int')
        if 1 < max(unique_runs) < 5:
            dat['Run'] = dat['Trial']
        else:
            dat['Run'] = dat['Block']
    except KeyError:
        run_cols = [col for col in dat.columns 
                    if ("TestList" in col) and (col.endswith("A.Cycle") or col.endswith("B.Cycle"))]
        dat['Run'] = np.where(dat[run_cols[0]] == "1", 1, 
                            np.where(dat[run_cols[1]] == "1", 2, 0)).astype(int)
        dat['Run'] = np.where((dat['Run'] == 0) & (dat.index <= 16), 1,
                      np.where((dat['Run'] == 0) & (dat.index > 16), 2, dat['Run']))

    # specify columns that have the start of scanner time (including initial volumes: GetReady; not calib volumes: PrepTime)
    if 'SiemensPad.OnsetTime' in dat.columns:
        #Siemens eprime cols
        ready_var = "SiemensPad.OnsetTime"
        prep_var = "SiemensPad.OffsetTime"
        
        # Save the values for preptime and getready time (includes volume prep alt)
        prep_per_run = dat.groupby("Run")[prep_var].apply(lambda lst: 
                                                            [value for value in lst if pd.notna(value)]).tolist()
        ready_per_run = dat.groupby("Run")[ready_var].apply(lambda lst: 
                                                              [value for value in lst if pd.notna(value)]).tolist()
            
    elif 'GetReady.RTTime' in dat.columns:
        #GE columns
        ready_var = "GetReady.RTTime"
        # Save the values for preptime (main) and getready time (includes volume prep alt)
        # the eprime data has a multiple valunes in GetReady.RTTime, start of scanner is first value of list per run
        # after initial values the onset is the last value
        ready_per_run = [lst[0] for lst in (dat.groupby("Run")[ready_var]
                                    .apply(lambda lst: [value for value in lst if pd.notna(value)]).tolist()) if lst]
        prep_per_run = [lst[-1] for lst in (dat.groupby("Run")[ready_var]
                                    .apply(lambda lst: [value for value in lst if pd.notna(value)]).tolist()) if lst]
        
        
    elif any(col.startswith("Wait4Scanner") and col.endswith(".RTTime") for col in dat.columns):
        #alt columns
        ready_var = [col for col in dat.columns 
                    if ("Wait4Scanner" in col) and (col.endswith(".RTTime"))]
        
        ready_run1 = dat.loc[dat['Run'] == 1, ready_var[0]].iloc[0]
        prep_run1 = dat.loc[dat['Run'] == 1, ready_var[0]].dropna().iloc[-1]
        ready_run2 = dat.loc[dat['Run'] == 2, ready_var[1]].iloc[0]
        prep_run2 = dat.loc[dat['Run'] == 2, ready_var[1]].dropna().iloc[-1]
        
        # Save the values for preptime (main) and getready time (includes volume prep alt)
        ready_per_run = [ready_run1,ready_run2]
        prep_per_run = [prep_run1,prep_run2]
        
    elif 'Waiting4ScannerGE' in dat.columns:
        #alt columns
        BeginFix_Onset = "BeginFix.OnsetTime"
        fix_start_per_run = [lst[0] for lst in (dat.groupby("Run")["BeginFix.OnsetTime"]
                                    .apply(lambda lst: [value for value in lst if pd.notna(value)]).tolist()) if lst]
        
        ready_var = "Waiting4ScannerGE"
        length_ready_var = dat.groupby('Run')['Waiting4ScannerGE'].apply(lambda x: x.notna().sum()).tolist()

        # Save the values for preptime (main) and getready time (includes volume prep alt)
        # the eprime data has a multiple valunes in GetReady.RTTime, start of scanner is first value of list per run
        # after initial values the onset is the last value
        tr_time = 800
        
        ready_per_run = [fix_start_per_run[0]-(length_ready_var[0]*tr_time),
                        fix_start_per_run[1]-(length_ready_var[1]*tr_time)]
        prep_per_run = [fix_start_per_run[0]-(1*tr_time),fix_start_per_run[1]-(1*tr_time)]
        
        
        
    # curate running Trials column for each run, drop NA columns to reduce preptime
    run_trial_cols = [col for col in dat.columns if ("TestList" in col) and (col.endswith("A") or col.endswith("B"))]


    try:
        dat['Trials'] = np.where(dat[run_trial_cols[0]].notnull(), dat[run_trial_cols[0]], 
                               np.where(dat[run_trial_cols[1]].notnull(), dat[run_trial_cols[1]], np.nan))
    except IndexError:
        dat['Trials'] = np.where(dat[run_trial_cols[0]].notnull(), dat[run_trial_cols[0]], np.nan)
    
    
    # Create Subtrial column remove NA subtrial columns
    dat = dat[~dat['Trials'].isna()]
    
    # get unique runs for loop
    uniq_runs = dat['Run'].unique().astype('int')
    
    for r in uniq_runs:
        # selection the first instance over GetReady and PrepTime for each run (see grouping above)
        try:
            dat['TriggerTime'] = np.tile(ready_per_run[r-1], len(dat))
            dat['TriggerTimeAlt'] = np.tile(prep_per_run[r-1], len(dat))
        except (IndexError, TypeError):
            dat['TriggerTime'] = np.tile(ready_per_run[0][r-1], len(dat))
            dat['TriggerTimeAlt'] = np.tile(prep_per_run[0][r-1] + .8, len(dat))
            
        # before conversion, convert values to numeric to ensure not objects 
        df = dat
        df_subset = convert_to_numeric(df)
        df_subset = df[df['Run'] == r]
        
        
        # keep column names for the output behavioral files, modify as needed
        keep_cols = ["Subject","Handedness","Trials","Procedure[SubTrial]","TrialCode",
                     "BeginFix.OnsetTime", "BeginFix.Duration","CorrectAnswer",
                     "EndFix.OffsetTime","EndFix.Duration",
                     "Fix.OnsetTime","Fix.FinishTime","Fix.Duration",
                     "Fix.RESP","Fix.RT",
                     "Go.OnsetTime","Go.Duration","Go.FinishTime","Go.ACC","Go.CRESP",
                     "Go.StartTime","Go.RTTime","Go.RT","Jitter",
                     "SSD.CRESP","SSD.OnsetDelay","SSD.OnsetTime",
                     "Stimulus","SSD.RT","SSD.RTTime","SSDDur",
                     "Stop_nback","StopDur", "StopSignal.ACC","StopSignal.OnsetTime","StopSignal.Duration",
                     "StopSignal.RESP","StopSignal.RT","StopSignal.RTTime",
                     "TriggerTime","TriggerTimeAlt"
                     ]
        cols_to_keep = [col for col in keep_cols if col in df_subset.columns]
        df_subset = df_subset[cols_to_keep]
        
        # subtract Onset/Finish times + adjust duration to seconds
        time_subtract = ["BeginFix.OnsetTime","Fix.OnsetTime","Fix.FinishTime",
                         "Go.OnsetTime","Go.FinishTime","Go.StartTime","Go.RTTime",
                         "SSD.OnsetTime", "StopSignal.OnsetTime", "StopSignal.RTTime"]
        
        duration_subtract = ["BeginFix.Duration","Fix.Duration","Go.Duration",
                            "Jitter"] # leaving SSDDur and StopSignal.Duration in ms
        
        for col_time in time_subtract:
            df_subset[col_time] = (df_subset[col_time] - df_subset['TriggerTimeAlt'])/1000
            
        for dur_time in duration_subtract:
            df_subset[dur_time] = df_subset[dur_time]/1000
        
        
        # create diff between scanner start and task start
        df_subset["DiffTriggerTimes"] = (df_subset["TriggerTimeAlt"]-df_subset["TriggerTime"])/1000
        
        # writeout .tsv per run
        df_subset.to_csv(f"{out_dir}/sub-{sub}_ses-{ses}_task-{task}_run-0{r}_events.tsv", sep = '\t', index = False)
        
elif task == "nback":
    # creating run and trial labels using try/except. Some eprime files are written oddly, so this info isn't always avail
        
    # specify columns that have the start of scanner time (including initial volumes: GetReady; not calib volumes: PrepTime)
    if 'SiemensPad.OnsetTime' in dat.columns:
        #Siemens eprime cols
        ready_var = "SiemensPad.OnsetTime"
        prep_var = "SiemensPad.OffsetTime"
        
        # run info 
        run_labs = [col for col in dat.columns if ("Waiting4Scanner" in col) and (col.endswith(".Cycle"))]

        row_run1_start = (dat[run_labs[0]] == 1).argmax()
        row_run2_start = (dat[run_labs[1]] == 1).argmax()

        if row_run2_start == 0:
            dat['Run'] = 1
        else:
            dat['Run'] = np.where(dat.index < row_run2_start, 1, 2)
        
        
        # create/extract scanner ready/prep details
        prep_per_run = dat.groupby("Run")[prep_var].apply(lambda lst: 
                                                            [value for value in lst if pd.notna(value)]).tolist()
        ready_per_run = dat.groupby("Run")[ready_var].apply(lambda lst: 
                                                              [value for value in lst if pd.notna(value)]).tolist()
            
            
    elif 'GetReady.OnsetTime' in dat.columns:
        #GE columns
        if "Waiting4Scanner.Cycle" in dat.columns:
            run_labs = [col for col in dat.columns if ("Waiting4Scanner" in col) and (col.endswith(".Cycle"))]
            row_run1_start = (dat[run_labs[0]] == 1).argmax()
            row_run2_start = (dat[run_labs[1]] == 1).argmax()

            dat['Run'] = np.where(dat.index < row_run2_start, 1, 2)
            
            # getting onset times
            ready_var = [col for col in dat.columns 
                        if ("GetReady" in col) and (col.endswith(".OffsetTime"))]
            
            ready_run1 = dat.loc[dat['Run'] == 1, ready_var[0]].iloc[0]
            prep_run1 = dat.loc[dat['Run'] == 1, ready_var[0]].dropna().iloc[-1]
            ready_run2 = dat.loc[dat['Run'] == 2, ready_var[1]].iloc[0]
            prep_run2 = dat.loc[dat['Run'] == 2, ready_var[1]].dropna().iloc[-1]
            
        
            # Save the values for preptime (main) and getready time (includes volume prep alt)
            ready_per_run = [ready_run1,ready_run2]
            prep_per_run = [prep_run1,prep_run2]
            
        else:
            ready_var = "GetReady.OnsetTime"
            prep_var = "GetReady.FinishTime"
            run_start = dat.loc[dat['Procedure[Block]'].str.startswith('TRSyncPROC')].index.tolist()
            
            if len(run_start) > 1:
                dat['Run'] = np.where(dat.index < run_start[1], 1, 2)
            else:
                dat['Run'] = np.where(dat.index >= run_start[0], 1, np.nan)
            
            
            # Save the values for preptime (main) and getready time (includes volume prep alt)
            # the eprime data has a multiple valunes in GetReady.RTTime, start of scanner is first value of list per run
            # after initial values the onset is the last value
            ready_per_run = [lst[0] for lst in (dat.groupby("Run")[ready_var]
                                        .apply(lambda lst: [value for value in lst if pd.notna(value)]).tolist()) if lst]
            prep_per_run = [lst[-1] for lst in (dat.groupby("Run")[prep_var]
                                        .apply(lambda lst: [value for value in lst if pd.notna(value)]).tolist()) if lst]
            
            dat = dat[~dat['Run'].isna()]
            
    # create subtrials
    dat['SubTrial'] = dat.loc[~dat['Procedure[Block]'].str.startswith('TRSyncPROC')] \
                       .groupby('Run') \
                       .cumcount() + 1
    
    # remove "Block" suffix in names to keep consistent
    if 'Fix.OnsetTime[Block]' in dat.columns:
        cols_to_rename = ['Running[Block]', 'Stim.ACC[Block]', 'Stim.CRESP[Block]', 'Stim.Duration[Block]',
                  'Stim.DurationError[Block]', 'Stim.FinishTime[Block]', 'Stim.OnsetDelay[Block]',
                  'Stim.OnsetTime[Block]', 'Stim.OnsetToOnsetTime[Block]', 'Stim.RESP[Block]',
                  'Stim.RT[Block]', 'Stim.RTTime[Block]', 'Stim.StartTime[Block]', 'StimType[Block]',
                  'Fix.Duration[Block]', 'Fix.DurationError[Block]', 'Fix.FinishTime[Block]',
                  'Fix.OnsetDelay[Block]', 'Fix.OnsetTime[Block]', 'Fix.OnsetToOnsetTime[Block]',
                  'Fix.StartTime[Block]']

        # Loop over the columns to be renamed and replace "Block" with an empty string
        for col in cols_to_rename:
            if col in dat.columns:
                new_col_name = col.replace("[Block]", "")
                dat = dat.rename(columns={col: new_col_name})
                       
    # remove NA subtrial columns
    dat = dat[~dat['SubTrial'].isna()]
    
    # get unique runs for loop
    uniq_runs = dat['Run'].unique().astype('int')
    
    for r in uniq_runs:
        # selection the first instance over GetReady and PrepTime for each run (see grouping above)
        try:
            dat['TriggerTime'] = np.tile(ready_per_run[r-1], len(dat))
            dat['TriggerTimeAlt'] = np.tile(prep_per_run[r-1], len(dat))
        except (IndexError, TypeError):
            dat['TriggerTime'] = np.tile(ready_per_run[0][r-1], len(dat))
            dat['TriggerTimeAlt'] = np.tile(prep_per_run[0][r-1], len(dat))
            
            
        # before conversion, convert values to numeric to ensure not objects 
        df = dat
        df_subset = convert_to_numeric(df)
        df_subset = df[df['Run'] == r]
        
        # keep column names for the output behavioral files, modify as needed
        keep_cols = ["Subject","Run","SubTrial","Handedness","Block","BlockType","StimType",
                     "TargetType","ConsecNonResp[Block]",
                     "ConsecSameResp","ControlAcc","CorrectResponse",
                     "Cue2Back.OnsetTime","Cue2Back.Duration","Cue2Back.FinishTime",
                     "CueFix.OnsetTime","CueFix.Duration","CueFix.FinishTime","CueFix.RTTime",
                     "CueTarget.OnsetTime","CueTarget.Duration","CueTarget.FinishTime",
                     "Fix.OnsetTime","Fix.Duration","Fix.FinishTime",
                     "Fix15sec.OnsetTime","Fix15sec.Duration",
                     "Stim.OnsetTime","Stim.Duration","Stim.FinishTime",
                     "Procedure[Block]","RunTrialNumber[Block]",
                     "Stim.ACC","Stim.CRESP","Stim.RESP","Stim.RT",
                     "Run3Block1","Run3Block2","Run3Block3","Run3Block4","Run3Block5",
                     "Run3Block6","Run3Block7","Run3Block8",
                     "Run4Block1","Run4Block2","Run4Block3","Run4Block4","Run4Block5",
                     "Run4Block6","Run4Block7","Run4Block8",
                     "TriggerTime","TriggerTimeAlt"
                     ]
        cols_to_keep = [col for col in keep_cols if col in df_subset.columns]
        df_subset = df_subset[cols_to_keep]
        
        # subtract Onset/Finish times + adjust duration to seconds
        time_subtract = ["Cue2Back.OnsetTime","Cue2Back.FinishTime",
        "CueFix.OnsetTime","CueFix.FinishTime",
        "CueTarget.OnsetTime","CueTarget.FinishTime",
        "Fix.OnsetTime","Fix.FinishTime","Fix15sec.OnsetTime",
        "Stim.OnsetTime","Stim.FinishTime"]
        
        duration_subtract = ["Cue2Back.Duration","CueFix.Duration","CueTarget.Duration",
                             "Fix.Duration","Fix15sec.Duration","Stim.Duration"]
        
        # due to column differences for select cases, to avoid errors using try/except
        # adding 800ms to Trigger time as trigger time is when volume is collected not + 800ms duration to next volume
        for col_time in time_subtract:
            try:
                df_subset[col_time] = (df_subset[col_time] - df_subset['TriggerTimeAlt'])/1000
            except Exception as e:
                print(f"Error processing column {col_time}: {e}")

        for dur_time in duration_subtract:
            try:
                df_subset[dur_time] = df_subset[dur_time]/1000
            except Exception as e:
                print(f"Error processing column {dur_time}: {e}")
            
        # create diff between scanner start and task start
        df_subset["DiffTriggerTimes"] = (df_subset["TriggerTimeAlt"]-(df_subset["TriggerTime"]))/1000
        
        # writeout .tsv per run
        df_subset.to_csv(f"{out_dir}/sub-{sub}_ses-{ses}_task-{task}_run-0{r}_events.tsv", sep = '\t', index = False)
        
        
    
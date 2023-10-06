import pandas as pd
import numpy as np
import json
import glob


# Create an empty DataFrame with the desired columns
columns = ['bids_name','subject_id','session_id','task_id']
column_suffixes = ['mrt', 'acc', 'mrt_hit', 'mrt_mis',
                   'cue_acc_lgrew', 'cue_acc_lgpun', 'cue_acc_neut',
                   'cue_acc_smrew', 'cue_acc_smpun',
                   'cue_mrt_lgrew', 'cue_mrt_lgpun', 'cue_mrt_neut',
                   'cue_mrt_smrew', 'cue_mrt_smpun',
                   'feedback_h-lgrew', 'feedback_m-lgrew', 'feedback_h-lgpun', 'feedback_m-lgpun',
                   'feedback_h-neut', 'feedback_m-neut', 'feedback_h-smrew', 'feedback_m-smrew',
                   'feedback_h-smpun', 'feedback_m-smpun']
df = pd.DataFrame(columns=columns)

# List of JSON file paths
json_files = glob.glob(f"../baselineYear1Arm1_MID/*task-mid_beh-descr.json")

for file_path in json_files:
    sub_id = file_path.split('_')[1].split('-')[1]
    ses_id = file_path.split('_')[2].split('-')[1]
    task_id = file_path.split('_')[3].split('-')[1]

    with open(file_path, 'r') as f:
        data = json.load(f)
    runs = []
    for run_name, run_data in data.items():
        if run_name.startswith('Run '):
            run_num = int(run_name[4:])
            runs.append(run_name)
    # Initialize a dictionary to store run-specific data
    run_data_dict = {'bids_name': file_path,
                     'subject_id': sub_id,
                     'session_id': ses_id,
                     'task_id': task_id
                     }

    for run in runs:
        # Extract the Mean RT and Overall Accuracy for the run
        run_data = data[run]
        mean_rt_run = run_data.get('Mean RT', np.nan)
        accuracy_run = run_data.get('Overall Accuracy', np.nan)
        mrt_hit_run = run_data.get('Mean RT by Hit/Miss', {}).get('1', np.nan)
        mrt_miss_run = run_data.get('Mean RT by Hit/Miss', {}).get('0', np.nan)
        acc_cond_run = run_data.get('Accuracy by Cue Condition', {})
        mrt_cond_run = run_data.get('Mean RT by Cue Condition', {})
        feedback_trials_run = run_data.get('Trials Per Feedback Condition', {})
        # Update the dictionary with run-specific data
        run_data_dict[f'mrt_{run.lower().replace(" ", "")}'] = mean_rt_run
        run_data_dict[f'acc_{run.lower().replace(" ", "")}'] = accuracy_run
        run_data_dict[f'mrt_hit-{run.lower().replace(" ", "")}'] = mrt_hit_run
        run_data_dict[f'mrt_mis-{run.lower().replace(" ", "")}'] = mrt_miss_run
        for cond, suffix in zip(['LgReward', 'LgPun', 'Triangle', 'SmallReward', 'SmallPun'], column_suffixes[4:]):
            run_data_dict[f'{suffix}-{run.lower().replace(" ", "")}'] = acc_cond_run.get(cond, np.nan)
        for cond, suffix in zip(['LgReward', 'LgPun', 'Triangle', 'SmallReward', 'SmallPun'], column_suffixes[9:]):
            run_data_dict[f'{suffix}-{run.lower().replace(" ", "")}'] = mrt_cond_run.get(cond, np.nan)
        for cond, suffix in zip(['You earn $5!','You did not earn $5!', 'You earn $0.20!', 'You did not earn $0.20!',
                                'Neutral Hit', 'Neutral Miss',
                                 'You keep $5!','You lose $5!', 'You keep $0.20!', 'You lose $0.20!'],
                                column_suffixes[14:]):
            run_data_dict[f'{suffix}-{run.lower().replace(" ", "")}'] = feedback_trials_run.get(cond, 0)
    # Append the dictionary to the DataFrame
    df = pd.concat([df, pd.DataFrame([run_data_dict])], ignore_index=True)





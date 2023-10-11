"""
This script defines a Python function called `export_csv` that processes JSON files containing task-specific data,
extracts relevant information, and exports the data to a CSV file. The function takes three arguments:

Parameters:
task (str): The name of the task ('MID', 'SST', or 'nback').
folder_path (str): The path to the folder containing JSON files (+ .png files) with task data.
out_path (str): The path where the resulting CSV file will be saved.

The function performs the following steps:

1. Initializes a DataFrame with predefined column names: 'bids_name', 'subject_id', 'session_id', 'task_id',
and 'image_path'.
2. Retrieves a list of JSON file paths in the specified folder that match the task name provided.
3. Counts the occurrences of "Run 1" and "Run 2" within the JSON data.
4. Iterates through each JSON file, extracting subject-specific information and constructing image paths.
5. Loads the JSON data from each file and counts the occurrences of "Run 1" and "Run 2" within the JSON data.
6. Extracts run-specific information from the JSON data based on the specified task ('MID', 'SST', or 'nback').
7. Appends the run-specific data to the DataFrame.
8. Saves the DataFrame to a CSV file with a name based on the task in the specified output directory.

Returns:
    Tuple[int, int, int]: A tuple containing three values:
        - The count of JSON files processed.
        - The count of "Run 1" occurrences.
        - The count of "Run 2" occurrences.

The returned values are used in summaries for the generate_html.py
Author: Michael Demidenko
Date: October 2023
"""

import pandas as pd
import numpy as np
import json
import glob
import os


def export_csv(task: str, folder_path: str, out_path: str):
    # Create an empty DataFrame with the desired columns
    columns = ['bids_name', 'subject_id', 'session_id', 'task_id', 'image_path']

    df = pd.DataFrame(columns=columns)

    # List of JSON file paths
    json_files = glob.glob(f"{folder_path}/*task-{task}_beh-descr.json")

    # count # runs
    r1 = 0
    r2 = 0

    for file_path in json_files:
        # set subject specific information: id, ses, task, basename of file and image path
        sub_id = file_path.split('_')[1].split('-')[1]
        ses_id = file_path.split('_')[2].split('-')[1]
        task_id = file_path.split('_')[3].split('-')[1]
        basename = os.path.splitext(os.path.basename(file_path))[0]
        image_path = f".{folder_path}/{basename}.png"

        # Initialize and assign subject specifics to data dict
        run_data_dict = {'bids_name': file_path,
                         'subject_id': sub_id,
                         'session_id': ses_id,
                         'task_id': task_id,
                         'image_path': image_path
                         }
        # load json data
        with open(file_path, 'r') as f:
            data = json.load(f)
        # count # of Run values within json (# available run counts)
        for item in data:
            if "Run 1" in json.dumps(item):
                r1 += 1
            if "Run 2" in json.dumps(item):
                r2 += 1

        # slicing run level interger info only from Run values within json
        runs = []
        for run_name, run_data in data.items():
            if run_name.startswith('Run '):
                # run_num = int(run_name[4:])
                runs.append(run_name)

        if task == 'MID':
            column_suffixes = ['mrt', 'acc', 'mrt_hit', 'mrt_mis',
                               'cue_acc_lgrew', 'cue_acc_lgpun', 'cue_acc_neut',
                               'cue_acc_smrew', 'cue_acc_smpun',
                               'cue_mrt_lgrew', 'cue_mrt_lgpun', 'cue_mrt_neut',
                               'cue_mrt_smrew', 'cue_mrt_smpun',
                               'feedback_h-lgrew', 'feedback_m-lgrew', 'feedback_h-lgpun', 'feedback_m-lgpun',
                               'feedback_h-neut', 'feedback_m-neut', 'feedback_h-smrew', 'feedback_m-smrew',
                               'feedback_h-smpun', 'feedback_m-smpun']
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
                # looping over some suffix to create condition specific summaries
                for cond, suffix in zip(['LgReward', 'LgPun', 'Triangle', 'SmallReward', 'SmallPun'],
                                        column_suffixes[4:]):
                    run_data_dict[f'{suffix}-{run.lower().replace(" ", "")}'] = acc_cond_run.get(cond, np.nan)
                for cond, suffix in zip(['LgReward', 'LgPun', 'Triangle', 'SmallReward', 'SmallPun'],
                                        column_suffixes[9:]):
                    run_data_dict[f'{suffix}-{run.lower().replace(" ", "")}'] = mrt_cond_run.get(cond, np.nan)
                for cond, suffix in zip(['You earn $5!', 'You did not earn $5!', 'You earn $0.20!',
                                         'You did not earn $0.20!', 'Neutral Hit', 'Neutral Miss',
                                         'You keep $5!', 'You lose $5!', 'You keep $0.20!', 'You lose $0.20!'],
                                        column_suffixes[14:]):
                    run_data_dict[f'{suffix}-{run.lower().replace(" ", "")}'] = feedback_trials_run.get(cond, 0)
            # Append the dictionary to the DataFrame
            df = pd.concat([df, pd.DataFrame([run_data_dict])], ignore_index=True)

        elif task == 'SST':
            for run in runs:
                run_data = data[run]
                go_acc = run_data.get('Go Accuracy', np.nan)
                go_rt = run_data.get('Go MRT', np.nan)
                stopsig_acc = run_data.get('Stop Signal Accuracy', np.nan)
                stopsigfail_mrt = run_data.get('Stop Signal MRT', np.nan)
                ssrt = run_data.get('Stop Signal Reaction Time', np.nan)
                ssd_min = run_data.get('Stop Signal Delay Durations', {}).get("min", np.nan)
                ssd_max = run_data.get('Stop Signal Delay Durations', {}).get("max", np.nan)

                # Update the dictionary with run-specific data
                run_data_dict[f'acc_go-{run.lower().replace(" ", "")}'] = go_acc
                run_data_dict[f'acc_stop-{run.lower().replace(" ", "")}'] = stopsig_acc
                run_data_dict[f'mrt_go-{run.lower().replace(" ", "")}'] = go_rt
                run_data_dict[f'mrt_stopfail-{run.lower().replace(" ", "")}'] = stopsigfail_mrt
                run_data_dict[f'ssrt_{run.lower().replace(" ", "")}'] = ssrt
                run_data_dict[f'ssd_min-{run.lower().replace(" ", "")}'] = ssd_min
                run_data_dict[f'ssd_max-{run.lower().replace(" ", "")}'] = ssd_max

            df = pd.concat([df, pd.DataFrame([run_data_dict])], ignore_index=True)

        elif task == 'nback':
            for run in runs:
                # Extract variables for the specific run (Run 1 or Run 2)
                overall_acc = data[run]['Overall Accuracy']
                n0back_acc = data[run]['Block Accuracy']['0-Back']
                n2back_acc = data[run]['Block Accuracy']['2-Back']
                negface_acc = data[run]['Stimulus Accuracy']['NegFace']
                neutface_acc = data[run]['Stimulus Accuracy']['NeutFace']
                posface_acc = data[run]['Stimulus Accuracy']['PosFace']
                place_acc = data[run]['Stimulus Accuracy']['Place']

                # Extract MRTs for the specific run (Run 1 or Run 2)
                overall_mrt = data[run]['Overall Mean RT']
                n0back_mrt = data[run]['Block Mean RT']['0-Back']
                n2back_mrt = data[run]['Block Mean RT']['2-Back']
                negface_mrt = data[run]['Stimulus Mean RT']['NegFace']
                neutface_mrt = data[run]['Stimulus Mean RT']['NeutFace']
                posface_mrt = data[run]['Stimulus Mean RT']['PosFace']
                place_mrt = data[run]['Stimulus Mean RT']['Place']

                # Extract dprime vars for the run
                n0back_dprime = data[run]['D-prime']['0back']
                n2back_dprime = data[run]['D-prime']['2back']

                # Update the dictionary with run-specific data
                # add acc
                run_data_dict[f'acc_all-{run.lower().replace(" ", "")}'] = overall_acc
                run_data_dict[f'acc_n0back-{run.lower().replace(" ", "")}'] = n0back_acc
                run_data_dict[f'acc_n2back-{run.lower().replace(" ", "")}'] = n2back_acc
                run_data_dict[f'acc_negface-{run.lower().replace(" ", "")}'] = negface_acc
                run_data_dict[f'acc_neutface-{run.lower().replace(" ", "")}'] = neutface_acc
                run_data_dict[f'acc_posface-{run.lower().replace(" ", "")}'] = posface_acc
                run_data_dict[f'acc_place-{run.lower().replace(" ", "")}'] = place_acc
                # add mrt
                run_data_dict[f'mrt_all-{run.lower().replace(" ", "")}'] = overall_mrt
                run_data_dict[f'mrt_n0back-{run.lower().replace(" ", "")}'] = n0back_mrt
                run_data_dict[f'mrt_n2back-{run.lower().replace(" ", "")}'] = n2back_mrt
                run_data_dict[f'mrt_negface-{run.lower().replace(" ", "")}'] = negface_mrt
                run_data_dict[f'mrt_neutface-{run.lower().replace(" ", "")}'] = neutface_mrt
                run_data_dict[f'mrt_posface-{run.lower().replace(" ", "")}'] = posface_mrt
                run_data_dict[f'mrt_place-{run.lower().replace(" ", "")}'] = place_mrt
                # add d-prime
                run_data_dict[f'd-prime_0back-{run.lower().replace(" ", "")}'] = n0back_dprime
                run_data_dict[f'd-prime_2back-{run.lower().replace(" ", "")}'] = n2back_dprime

            df = pd.concat([df, pd.DataFrame([run_data_dict])], ignore_index=True)

    # save csv to group output path
    df.to_csv(f'{out_path}/group_{task}.csv', index=False)

    return len(json_files), r1, r2

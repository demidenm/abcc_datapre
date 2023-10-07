"""
Behavioral Plotting Script

This Python script generates behavioral plots for research purposes. Some of the components used in this script, including group templates, plotting functions, CSS styles, and JavaScript scripts, have been adopted and, in some cases, tailored from the MRIQC project.

MRIQC is an open-source software project for quality control and analysis of MRI data, and the components used in this script were sourced from the MRIQC repository: https://github.com/nipreps/mriqc/tree/f360d1eb46909626c4ca9dbeb5514c3d8fab29e7/mriqc/reports

The adoption of these components has been done with proper attribution and adherence to their respective licenses and terms of use.

Author: Michael Demidenko

Date: Current Date
"""

from . import group
from pathlib import Path
import pandas as pd
import numpy as np
import json
import glob
import os

print(os.getcwd())

task = 'MID'
folder_path = f"./baselineYear1Arm1_{task}"
out_path = "./beh_html"
html_desc = f"./scripts/templates/describe_report_{task}.txt"

# Create an empty DataFrame with the desired columns
columns = ['bids_name', 'subject_id', 'session_id', 'task_id', 'image_path']
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
json_files = glob.glob(f"{folder_path}/*task-{task}_beh-descr.json")

# count # runs
r1 = 0
r2 = 0
for file_path in json_files:
    sub_id = file_path.split('_')[1].split('-')[1]
    ses_id = file_path.split('_')[2].split('-')[1]
    task_id = file_path.split('_')[3].split('-')[1]
    basename = os.path.splitext(os.path.basename(file_path))[0]
    image_path = f".{folder_path}/{basename}.png"

    with open(file_path, 'r') as f:
        data = json.load(f)

    for item in data:
        if "Run 1" in json.dumps(item):
            r1 += 1
        if "Run 2" in json.dumps(item):
            r2 += 1

    runs = []
    for run_name, run_data in data.items():
        if run_name.startswith('Run '):
            run_num = int(run_name[4:])
            runs.append(run_name)
    # Initialize a dictionary to store run-specific data
    run_data_dict = {'bids_name': file_path,
                     'subject_id': sub_id,
                     'session_id': ses_id,
                     'task_id': task_id,
                     'image_path': image_path
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
        for cond, suffix in zip(['You earn $5!', 'You did not earn $5!', 'You earn $0.20!', 'You did not earn $0.20!',
                                 'Neutral Hit', 'Neutral Miss',
                                 'You keep $5!', 'You lose $5!', 'You keep $0.20!', 'You lose $0.20!'],
                                column_suffixes[14:]):
            run_data_dict[f'{suffix}-{run.lower().replace(" ", "")}'] = feedback_trials_run.get(cond, 0)
    # Append the dictionary to the DataFrame
    df = pd.concat([df, pd.DataFrame([run_data_dict])], ignore_index=True)

df.to_csv(f'{folder_path}/group_{task}.csv', index=False)

items = {
    "MID": [
        (["Acc", "acc_run1", "acc_run2"], '%'),
        (["Cue_Acc", "cue_acc_lgrew-run1", "cue_acc_lgpun-run1", "cue_acc_neut-run1", "cue_acc_smrew-run1",
          "cue_acc_smpun-run1", ],
         '% - Run1'),
        (["Cue_Acc", "cue_acc_lgrew-run2", "cue_acc_lgpun-run2", "cue_acc_neut-run2", "cue_acc_smrew-run2",
          "cue_acc_smpun-run2", ],
         '% - Run1'),
        (["MRT", "mrt_run1", "mrt_run2", ], 'ms'),
        (["MRT",
          "mrt_hit-run1",
          "mrt_mis-run1",
          "mrt_hit-run2",
          "mrt_mis-run2", ],
         'ms - 1/0'),
        (["Cue_MRT",
          "cue_mrt_lgrew-run1", "cue_mrt_lgpun-run1",
          "cue_mrt_neut-run1", "cue_mrt_smrew-run1",
          "cue_mrt_smpun-run1", ],
         'ms'),
        (["Cue_MRT",
          "cue_mrt_lgrew-run2", "cue_mrt_lgpun-run2",
          "cue_mrt_neut-run2", "cue_mrt_smrew-run2",
          "cue_mrt_smpun-run2", ],
         'ms'),
        (["Feedback",
          "feedback_h-lgrew-run1", "feedback_lgrew_nmiss_run1", "feedback_h-smrew-run1", "feedback_m-smrew-run1",
          "feedback_h-neut-run1", "feedback_m-neut-run1", "feedback_h-smpun-run1", "feedback_m-smpun-run1",
          "feedback_h-lgpun-run1", "feedback_m-lgpun-run1", ],
         'n - 1/0'),
        (["Feedback",
          "feedback_h-lgrew-run2", "feedback_lgrew_nmiss_run2", "feedback_h-smrew-run2", "feedback_m-smrew-run2",
          "feedback_h-neut-run2", "feedback_m-neut-run2", "feedback_h-smpun-run2", "feedback_m-smpun-run2",
          "feedback_h-lgpun-run2", "feedback_m-lgpun-run2", ],
         'n - 1/0'),
    ],
    "nback": [
        (["efc"], None),
        (["fber"], None),
        (["fwhm", "fwhm_x", "fwhm_y", "fwhm_z"], "mm"),
    ],
}

with open(html_desc, "r", encoding="utf-8") as input_html:
    html_content = input_html.read()

modality = [task]

for mod in modality:
    csv_path = Path(f'{folder_path}/group_{task}.csv')
    out_html = f"{out_path}/group_{mod}.html"
    group.gen_html(
        csv_file=csv_path,
        mod=mod,
        n_subjects=len(json_files),
        runs=[r1, r2],
        description=html_content,
        qc_items=items,
        out_file=out_html
    )
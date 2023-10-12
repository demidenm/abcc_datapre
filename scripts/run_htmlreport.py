"""
Behavioral Plotting Script

This Python script generates behavioral plots for research purposes. Some components used in this script,
including group templates, plotting functions, CSS styles, and JavaScript scripts, have been adopted and, in
some cases, tailored from the MRIQC project.

MRIQC is an open-source software project for quality control and analysis of MRI data, and the components used in this
script were sourced from the MRIQC repository:
https://github.com/nipreps/mriqc/tree/f360d1eb46909626c4ca9dbeb5514c3d8fab29e7/mriqc/reports

The adoption of these components has been done with proper attribution and adherence to their respective licenses and
terms of use.

Author: Michael Demidenko
Date: October 2023
"""

from . import (group_html, groupcsv_output)
from pathlib import Path
import argparse

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="This script runs calculates subject specific information "
                                             "and generates group csv that is used in creating the html report(s).")


parser.add_argument("-t", "--task_name", type=str, required=True,
                    help="task name, e.g. MID, SST, nback")
parser.add_argument("-i", "--in_dir", type=str, required=True,
                    help="Input directory where subject specific .json & .png files are, "
                         "e.g. ./baseline_sst")
parser.add_argument("-d", "--task_desc", type=str, required=True,
                    help="path to file contain html description for task, "
                         "e.g. ./scripts/templates/describe_report_sst.txt ")
parser.add_argument("-o", "--out_dir", type=str, required=True,
                    help="output directory for aggregated .csv file")

args = parser.parse_args()

# Assign values to variables
task = args.task_name
folder_path = args.in_dir
html_desc = args.task_desc
out_path = args.out_dir

# create task specific .csv file
json_n, r1, r2 = groupcsv_output.export_csv(task=task, folder_path=folder_path, out_path=out_path)

items = {
    "MID": [
        (["Acc", "acc_run1", "acc_run2"], '%'),
        (["Cue_Acc", "cue_acc_lgrew-run1", "cue_acc_lgrew-run2",
          "cue_acc_lgpun-run1", "cue_acc_lgpun-run2", "cue_acc_neut-run1", "cue_acc_neut-run2",
          "cue_acc_smrew-run1", "cue_acc_smrew-run2", "cue_acc_smpun-run1", "cue_acc_smpun-run2", ],
         '%'),
        (["MRT", "mrt_run1", "mrt_run2", ], 'ms'),
        (["MRT",
          "mrt_hit-run1",
          "mrt_hit-run2",
          "mrt_mis-run1",
          "mrt_mis-run2", ],
         'ms'),
        (["Cue_MRT",
          "cue_mrt_lgrew-run1", "cue_mrt_lgrew-run2", "cue_mrt_lgpun-run1", "cue_mrt_lgpun-run2",
          "cue_mrt_neut-run1", "cue_mrt_neut-run2", "cue_mrt_smrew-run1", "cue_mrt_smrew-run2",
          "cue_mrt_smpun-run1", "cue_mrt_smpun-run2", ],
         'ms'),
        (["Feedback",
          "feedback_h-lgrew-run1", "feedback_m-lgrew-run1", "feedback_h-smrew-run1", "feedback_m-smrew-run1",
          "feedback_h-neut-run1", "feedback_m-neut-run1", "feedback_h-smpun-run1", "feedback_m-smpun-run1",
          "feedback_h-lgpun-run1", "feedback_m-lgpun-run1", ],
         'n'),
        (["Feedback",
          "feedback_h-lgrew-run2", "feedback_m-lgrew-run2", "feedback_h-smrew-run2", "feedback_m-smrew-run2",
          "feedback_h-neut-run2", "feedback_m-neut-run2", "feedback_h-smpun-run2", "feedback_m-smpun-run2",
          "feedback_h-lgpun-run2", "feedback_m-lgpun-run2", ],
         'n'),
    ],
    "SST": [
        (["Go Acc", "acc_go-run1", "acc_go-run2", ], 'Go %'),
        (["Stop Acc", "acc_stop-run1", "acc_stop-run2"], 'Stop %'),
        (["Go MRT", "mrt_go-run1", "mrt_go-run2", ], 'Go ms'),
        (["Fail Stop MRT", "mrt_stopfail-run1", "mrt_stopfail-run2", ], 'Stop ms'),
        (["SSRT", "ssrt_run1", "ssrt_run2", ], 'ms'),
        (["SSD Min", "ssd_min-run1", "ssd_min-run2", ], "min ms"),
        (["SSD Max", "ssd_max-run1", "ssd_max-run2"], "max ms"),

    ],
    "nback": [
        (["Block Acc",
          "acc_all-run1", "acc_all-run2",
          'acc_n0back-run1', 'acc_n0back-run2',
          'acc_n2back-run1', 'acc_n2back-run2', ], 'Block %'),
        (["Cue Acc" 
          "acc_posface-run1", "acc_posface-run2",
          "acc_neutface-run1", "acc_neutface-run2",
          "acc_negface-run1", "acc_negface-run2",
          "acc_place-run1", "acc_place-run2", ], 'Cue %'),
        (["Block MRT",
          "mrt_all-run1", "mrt_all-run2",
          'mrt_n0back-run1', 'mrt_n0back-run2',
          'mrt_n2back-run1', 'mrt_n2back-run2', ], 'Block ms'),
        (["Cue MRT"
          "mrt_posface-run1", "mrt_posface-run2",
          "mrt_neutface-run1", "mrt_neutface-run2",
          "mrt_negface-run1", "mrt_negface-run2",
          "mrt_place-run1", "mrt_place-run2", ], 'Cue ms'),
        (["D'",
          "d-prime_0back-run1", "d-prime_0back-run2",
          "d-prime_2back-run1", "d-prime_2back-run2", ], None),
    ],
}

with open(html_desc, "r", encoding="utf-8") as input_html:
    html_content = input_html.read()

csv_path = Path(f'{out_path}/group_{task}.csv')
out_html = f"{out_path}/group_{task}.html"
group_html.gen_html(
    csv_file=csv_path,
    mod=task,
    n_subjects=json_n,
    runs="Run1 = {} & Run 2 = {}".format(r1, r2),
    description=html_content,
    qc_items=items,
    out_file=out_html
)

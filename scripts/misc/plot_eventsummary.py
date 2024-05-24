import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
plt.switch_backend('Agg')


def parse_args():
    parser = argparse.ArgumentParser(description="This script creates QC plots for events files for tasks/sess/runs")
    parser.add_argument("--beh_inp", type=str, required=True,
                        help="Input directory path subdir (e.g. baselineYear1Arm1_MID) beh folders exist ")
    parser.add_argument("--summ_out", type=str, required=True,
                        help="Path to folder where to write out summary files that aggrate info")
    parser.add_argument("--fig_out", type=str, required=True,
                        help="path to folder where to save figures")
    parser.add_argument("--nda_file", type=str, required=True,
                        help="Path to csv w/ NDA info that contains colums: subject, site, scanner, software, session."
                             "subjects IDs should note contain underscore. Session should be replaced to match ABCC "
                             "values, e.g. baselineYear1Arm1 or 2YearFollowUpYArm1")

    return parser.parse_args()


def create_behdat(task_name, cue_start, events_folds, list_runs, list_sessions, nda_info, beh_outdir):
    for run in list_runs:
        for sess in list_sessions:
            subs = glob(f'{events_folds}/{sess}_{task_name}/*run-{run}_events.tsv')
            n = len(subs)
            # Create a filename based on the session and run
            filename = f'n-{n}_summary-{task_name}_timings_{sess}_run-{run}.csv'
            save_path = os.path.join(beh_outdir, filename)  # Ensure 'output_directory' exists
            if not os.path.exists(save_path):
                onset_times = []
                for sub in subs:
                    base = os.path.basename(sub)
                    sub_id = base.split('_')[0].split('-')[1]

                    try:
                        beh_file = pd.read_csv(sub, sep='\t')
                        task_onset = beh_file[cue_start].iloc[0]
                        diff_times = beh_file[diff_trigger].iloc[0]
                        trig_col = beh_file[scantrig].iloc[0]
                        cali_col = beh_file[calibrend].iloc[0]
                        read_col = beh_file[read_type].iloc[0]
                        site = nda_info.query(f"subject == '{sub_id}' & session == '{sess}'")["site"].values[0]
                        scanner = nda_info.query(f"subject == '{sub_id}' & session == '{sess}'")["scanner"].values[0]
                        software = nda_info.query(f"subject == '{sub_id}' & session == '{sess}'")["software"].values[0]

                        onset_times.append({
                            'task_onset': task_onset,
                            'diff_triggers': diff_times,
                            'site': site,
                            'scanner': scanner,
                            'software': software,
                            'calibrend_col': cali_col,
                            'trig_col': trig_col,
                            'read_col': read_col
                        })
                    except Exception as e:
                        print(f"Error {e}: for {sub_id}")
                        continue

                # Convert the collected data to a DataFrame
                df_onset_times = pd.DataFrame(onset_times)
                df_onset_times.to_csv(save_path, index=False)
                print("Created and saved: ", save_path)
            else:
                print("Not creating, file exists:\n", save_path)


def plt_run_sess(taskname, x_name, y_name, list_runs, list_sessions, plot_out,
                 summ_path, add_y_mean=False, x_ticksangle=75):
    n_cols = len(list_sessions)
    n_rows = len(list_runs)
    sns.set(style='white', font='DejaVu Serif')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), dpi=300)

    for sess in list_sessions:
        for i, run in enumerate(list_runs):
            path_str = glob(f'{summ_path}/n-*_summary-{taskname}_timings_{sess}_run-{run}.csv')[0]
            n_file = os.path.basename(path_str).split('_')[0].split('-')[1]
            onsets_df = pd.read_csv(path_str, sep=',')
            onsets_df['software'] = onsets_df['software'].apply(
                lambda x: x.split(':', 1)[1].strip() if isinstance(x, str) and 'Software release:' in x else x)
            col = list_sessions.index(sess)
            # create plot
            ax = axes[i, col] if n_rows > 1 else axes[col]
            sns.boxplot(ax=ax, x=x_name, y=y_name, data=onsets_df, dodge=False)
            if add_y_mean:
                ax.axhline(onsets_df[y_name].mean(), color='r', linestyle='--', linewidth=1.5,
                           label=f'All Avg {y_name}')
            if y_name == "diff_triggers":
                ax.axhline(6.4, color='r', linestyle='--', linewidth=1.5,
                           label="6.4sec Trigger T")
                ax.axhline(12, color='b', linestyle='--', linewidth=1.5,
                           label="12sec Trigger T")

            ax.set_title(f'Ses-{sess} & Run-{run} (n = {n_file})')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_ticksangle, fontsize=8)
            ax.set_xlabel(" ")
            ax.set_ylabel(f'{y_name} (sec)')
            ax.legend()

    fig.suptitle(f'{taskname}: Boxplot of {y_name} ~ {x_name}')
    plt.tight_layout()
    plt.savefig(plot_out)
    plt.close()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    beh_inp = args.beh_inp
    summ_out = args.summ_out
    fig_out = args.fig_out
    nda_file = args.nda_file

    assert os.path.exists(beh_inp), f"Input directory {beh_inp} does not exist."
    if not os.path.exists(summ_out):
        print(f"Output directory {summ_out} does not exist. Creating it now.")
        os.makedirs(summ_out)
    if not os.path.exists(fig_out):
        print(f"Output directory {fig_out} does not exist. Creating it now.")
        os.makedirs(fig_out)

    # set lists
    onset_cols = {
        'MID': 'Cue.OnsetTime',
        'nback': 'CueFix.OnsetTime',
        'SST': 'BeginFix.OnsetTime'
    }

    plot_by_xy = {
        'task_onset': ['site', 'scanner', 'software', 'calibrend_col', 'trig_col', 'read_col'],
        'diff_triggers': ['software', 'scanner']
    }

    run_list = ['01', '02']
    sess_list = ['baselineYear1Arm1', '2YearFollowUpYArm1']
    task_list = ['MID', 'SST', 'nback']
    diff_trigger = 'DiffTriggerTimes'
    scantrig = 'scantrig_col'
    calibrend = 'calibrend_col'
    read_type = 'eprime_readtype'

    nda_df = pd.read_csv(nda_file, sep=',')
    print(f'Running the following list of tasks, sessions and runs', task_list, sess_list, run_list)
    for task in task_list:
        onset_name = onset_cols[task]
        create_behdat(task_name=task, events_folds=beh_inp, cue_start=onset_name,
                      list_runs=run_list, list_sessions=sess_list, beh_outdir=summ_out, nda_info=nda_df)

        for y_type in ['task_onset', 'diff_triggers']:
            if y_type == 'task_onset':
                add_mean = True
            if y_type == 'diff_triggers':
                add_mean = False
            for x_type in plot_by_xy[y_type]:
                figpath_name = f'{fig_out}/plt_task-{task}_axis-{y_type}by{x_type}.png'
                plt_run_sess(taskname=task, x_name=x_type, y_name=y_type,
                             list_runs=run_list, list_sessions=sess_list,
                             plot_out=figpath_name, summ_path=summ_out,
                             add_y_mean=add_mean, x_ticksangle=75)

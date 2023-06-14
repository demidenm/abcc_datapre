# ABCC ScriptThis is a working folder building on scripts for the [ABCD-BIDS dataset](https://collection3165.readthedocs.io/en/stable/)Current Scripts/files:1) eprimetotsv.py: Converts edat/edat2 E-prime *EventsRelatedInformation.txt files pulled from NDA to *_events.tsv files2) summary_taskonsets.py: Sanity check to ensure distribution of onsets for tasks is within an expected range as deviations of seconds can cause BOLD GLM issues.3) eventsbeh_describe.py: Script that summarizes metrics from subject and run specific task events.tsv files, generative summary .json files and .png images    - describe_tsv.sh: Script to simplying running eventsbeh_describe.py by providing input/output/task information.4) behave_htmlrep.py: Script to compile subject information from .json data (based on input folder) for specific session and task to generate group figure, task description and subject specific plots/summaries    - describe_report_MID.txt: tentative summary of the Monetary Incetive Delay task    - describe_report_SST.txt: tentative summary of the Stop Signal Task    - describe_report_nback.txt: tentative summary of the Emotional N-back task        Coming soon:1) tempaltes and sbatch scripts used to run MRIQC v21.0.0 on ABCD-BIDS input data2) tempaltes and sbatch scripts used to run fMRIprep v21.1.0 on ABCD-BIDS input dataThese scripts are a work in-progress and will change (some more than others). Currently, the scripts do not reflect the core ABCC procedures.
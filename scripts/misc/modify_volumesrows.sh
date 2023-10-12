# get volume info from the preprocessed images for the MID task
# Siemens scan = 411 volumes, Hagler et al "8 frame make up pre-scan" 411-8 = 403
echo "#### Modifying runs .nii and .tsv to match target length ####"
target_vols=403
declare -A vols_run
declare -A diff_vols_run
declare -A len_conf_run
declare -A newvols_run

# Loop over the runs
for run in $(seq 1 2); do
	echo "Modifying Run: ${run} "
	# Calculate vols and diff_vols
	vols_run[$run]=$(fslinfo "${tmp_in}/${sub}/${ses}/func/${sub}_${ses}_task-MID_run-0${run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz" \
		| grep -w "dim4" \
		| awk -F" " '{ print $2 }')

	diff_vols_run[$run]=$((vols_run[$run] - target_vols))

	# Output the difference
	echo "Difference from target 407. Run${run}: ${diff_vols_run[$run]}"

	# Extract volumes using fslroi and cut fmriprep confound rows
	fslroi "${tmp_in}/${sub}/${ses}/func/${sub}_${ses}_task-MID_run-0${run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz" \
		"${tmp_in}/${sub}/${ses}/func/${sub}_${ses}_task-MID_run-0${run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz" \
		${diff_vols_run[$run]} ${target_vols}

	remove_run=$((diff_vols_run[$run] + 1))
	sed -i "2,${remove_run}d" "${tmp_in}/${sub}/${ses}/func/${sub}_${ses}_task-MID_run-0${run}_desc-confounds_timeseries.tsv"
	len_conf_run[$run]=$(cat "${tmp_in}/${sub}/${ses}/func/${sub}_${ses}_task-MID_run-0${run}_desc-confounds_timeseries.tsv" | wc -l)

	# new vols calc
	newvols_run[$run]=$(fslinfo "${tmp_in}/${sub}/${ses}/func/${sub}_${ses}_task-MID_run-0${run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz" \
                | grep -w "dim4" \
                | awk -F" " '{ print $2 }')
done

# Output summary statistics
echo
echo "SUMMARY STATISTICS:"
echo -e "run1_dim4 = ${vols_run[1]}\trun2_dim4 = ${vols_run[2]}\nrun1_vols_cut = ${diff_vols_run[1]}\trun2_vols_cut = ${diff_vols_run[2]}"
echo -e "conf_len_run1 = ${len_conf_run[1]}\tconf_len_run2 = ${len_conf_run[2]}"

# Append summary statistics to the volume info file
echo -e "${sub}\t${vols_run[1]}\t${newvols_run[1]}\t${vols_run[2]}\t${newvols_run[2]}\t${diff_vols_run[1]}\t${diff_vols_run[2]}\t${len_conf_run[1]}\t${len_conf_run[2]}" >> ${vol_info_file}
echo

# Check if the value in ${newvols_run[1]} is less than 403
if (( ${newvols_run[1]} < 403 )); then
    echo "Failed, volumes below expected 403. Check data."
# Check if the value in ${newvols_run[2]} is less than 403
elif (( ${newvols_run[2]} < 403 )); then
    echo "Failed, volumes below expected 403. Check data."
else
    echo "Values are >= 403. Proceeding..."
fi

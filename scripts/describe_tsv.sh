#!/bin/bash -l

module load python

dir=`pwd`
log_out=${dir}/../logs
curr_date=$(date +"%m/%d/%Y")

echo "Starting script..."

echo -e "Which session? Options: 2YearFollowUpYArm1, baselineYear1Arm1."
read ses
echo -e "Which task? Options: MID, SST, nback." 
read task

# create output fold based on prompts

out=${dir}/../${ses}_${task}
err_log=${log_out}/${ses}_${task}.err
out_log=${log_out}/${ses}_${task}.out
beh_dir=${dir}/../../events_to_tsv
in_dir=${beh_dir}/${ses}_${task}
ev_log=${beh_dir}/logs/${ses}_${task}.out
sub_list=${dir}/../sub_lists/${ses}_${task}.txt

# check if sub list exists, if not, create from completed log IDs
if [ ! -f ${sub_list} ] ;  then
	cat ${ev_log} | awk -F'\t' '{ print $1 }' > $sub_list
fi

echo "Options selected: ses: $ses, task: $task "

echo -e "Which you like to proced with options: \n\tses-${ses} & task-${task} in ${sub_list} ? y or n "
read response

if [[ "$response" =~ ^[Yy]$ ]]; then
	echo "  Running Script... check output in: "
	echo "	$out "
else
	echo " Abort script..."
	exit 1
fi

# make directors if do not exist
if [ ! -d "${out}" ]; then
    mkdir -p "${out}"
fi

if [ ! -d "${log_out}" ]; then
    mkdir -p "${log_out}"
fi


# run python script
c=0

cat ${sub_list} | while read line ; do 
	c=$((c+1))
	sub=$(echo $line )
	echo "     Starting $sub [${c}]"
	
	# run python script and save error for troubleshooting 
	error_msg=$(python ./abcc_beh_describe.py -i ${in_dir} -o ${out} -s ${sub} -e ${ses} -t ${task} 2>&1 | tr -d '\n')
    	if [[ ! -z "${error_msg}" ]]; then
        	echo -e "${sub}\t${ses}\t${task}\t${curr_date}\t${error_msg}" >> ${err_log}
    	else
		com_msg=".py script completed w/o error"
		out_file=$(echo "${out}/sub-${sub}_ses-${ses}_task-${task}_*")
		echo -e "${sub}\t${ses}\t${task}\t${curr_date}\t${comp_msg}" >> ${out_log}
	fi
	

done

echo
err_len=$(cat $err_log | wc -l )
echo "Completed. Check logs, ${err_len} errors."
echo

#!/bin/bash -l

module load python

dir=`pwd`
tmp=${dir}/tmp
files=${dir}/files
log_out=${dir}/logs
curr_date=$(date +"%m/%d/%Y")

echo "Starting script..."

echo -e "Which session? Options: 2YearFollowUpYArm1, baselineYear1Arm1."
read ses
echo -e "Which task? Options: MID, SST, nback." 
read task
echo -e "Which subject ID list? Provide file with list of full path \n to *EventsInfo.txt"
read subs

echo "Options selected: ses: $ses, task: $task, sub: $subs ."

echo -e "Which you like to proced with options: \n\tses-${ses} & task-${task}? y or n" 
read response

# create output fold based on prompts
out=${dir}/${ses}_${task}
err_log=${log_out}/${ses}_${task}.err
out_log=${log_out}/${ses}_${task}.out

echo 
if [[ "$response" =~ ^[Yy]$ ]]; then
	echo "  Running Script... check output in: "
	echo "	$out "
else
	echo " Abort script..."
	exit 1
fi

# make directors if do not exist
if [ ! -d "${tmp}" ]; then
    mkdir -p "${tmp}"
fi

if [ ! -d "${out}" ]; then
    mkdir -p "${out}"
fi

if [ ! -d "${log_out}" ]; then
    mkdir -p "${log_out}"
fi

# run python script
c=0
cat ${subs} | while read line ; do 
	dir=$(dirname $line )
	file=$(basename $line )
	sub=$(echo $file | awk -F'_' '{ print $1}' | awk -F'-' '{ print $2}') 
	ses=$(echo $file | awk -F'_' '{ print $2}' | awk -F'-' '{ print $2}')
	run=$(echo $file | awk -F'_' '{ print $4}' | awk -F'-' '{ print $2}')
	c=$((c+1))

	echo "     Starting $sub [${c}]"
	
	cp $line $tmp
	# run python script and save error for troubleshooting 
	error_msg=$(python ../eprimetotsv.py -i ${tmp} -o ${out} -s ${sub} -e ${ses} -r ${run} -t ${task} 2>&1 | tr -d '\n')
    	if [[ ! -z "${error_msg}" ]]; then
        	echo -e "${sub}\t${ses}\t${task}\t${run}\t${curr_date}\t${error_msg}" >> ${err_log}
    	else
		com_msg=".py script completed w/o error"
		out_file=$(echo "${out}/sub-${sub}_ses-${ses}_task-${task}_run-${run}_events.tsv")
		echo -e "${sub}\t${ses}\t${task}\t${run}\t${curr_date}\t${comp_msg}" >> ${out_log}
	fi
	
	rm ${tmp}/${file}

done

echo

len_comp=$(cat $out_log | wc -l )
len_err=$(cat $err_log | wc -l )
echo "Script completed without error $len_comp subjects. Errors in ${len_err}"

echo

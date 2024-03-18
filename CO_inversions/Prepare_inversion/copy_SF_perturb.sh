#!/bin/bash

# Define the source directory
source_dir="/u/bbyrne1/python_codes/Canada_Fires_2023/Byrne_etal_codes/CO_inversions/Prepare_inversion/SF_perturb"
# Define the destination directory
dest_dir="/nobackup/bbyrne1/GHGF-CMS-3day-COinv-MonteCarlo"

# Iterate over QFED and GFAS
for model in "QFED" "GFAS"; do
    # Define the destination subdirectory based on the model
    
    # Iterate over ens_member from 01 to 40
    for ((ens_member=1; ens_member<=40; ens_member++)); do
	
	# Pad the ens_member with leading zeros if necessary
        ens_member_padded=$(printf "%02d" $ens_member)
        
        # Define the source and destination files
        source_file="SF_perturb_${model}_2023_${ens_member_padded}.nc"
        model_dest_dir="${dest_dir}/Run_COinv_${model}_${ens_member_padded}"
	dest_file_old="SF_perturb_${ens_member_padded}.nc"
	dest_file="SF_perturb.nc"

	rm ${model_dest_dir}/${dest_file_old}
        cp ${source_dir}/${source_file} ${model_dest_dir}/${dest_file}
    done
done

#!/bin/bash

root="/vinai/phucph11/shared/GarmentCode/GarmentCodeData_v2"

n_folders=36
sub_folders=("default_body" "random_body")

for ((idx=0; idx<n_folders; idx++)); do
    for sub_folder in "${sub_folders[@]}"; do
        folder_name="${root}/garments_5000_${idx}/${sub_folder}"
        mkdir -p "${folder_name}/data"
        tar --use-compress-program=pigz -xvf "${folder_name}/data.tar.gz" -C "${folder_name}/data"
    done
done

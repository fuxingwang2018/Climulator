#!/bin/bash

# Define the base input directory
resolution="3km"
input_base_dir="/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/original/"${resolution}
output_base_dir="/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/"${resolution}

mkdir -p "$output_base_dir"  # Create output base directory if it doesn't exist

# Define the cropping range (x_start:x_end, y_start:y_end)
if [[ "$resolution" == "12km" ]]; then
    x_range="1,106"  # Replace with your desired range
    y_range="1,88"   # Replace with your desired range
elif [[ "$resolution" == "3km" ]]; then
    x_range="1,424"  # Replace with your desired range
    y_range="1,352"   # Replace with your desired range
fi

# Load modules if needed (e.g., on HPC systems)
module load CDO/2.3.0-eccodes-aec-cmor-fftw-hpc2-intel-2023a-eb

# Function to process NetCDF files in subdirectories
process_directory() {
    local input_dir="$1"
    local output_dir="$2"

    # Create the corresponding output directory
    mkdir -p "$output_dir"

    # Loop through all NetCDF files in the current directory
    for file in "$input_dir"/*.nc; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")  # Extract the filename
            output_file="$output_dir/$filename"

            # Use cdo to crop the NetCDF file
            cdo selindexbox,$x_range,$y_range "$file" "$output_file"
            if [ $? -eq 0 ]; then
                echo "Cropped $file and saved to $output_file"
            else
                echo "Error cropping $file"
            fi
        fi
    done

    # Recursively process subdirectories
    for subdir in "$input_dir"/*/; do
        if [ -d "$subdir" ]; then
            sub_output_dir="$output_dir/$(basename "$subdir")"
            process_directory "$subdir" "$sub_output_dir"
        fi
    done
}

# Start processing from the base input directory
process_directory "$input_base_dir" "$output_base_dir"

echo "Cropping completed for all files and subdirectories."



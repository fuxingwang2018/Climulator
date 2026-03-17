#!/bin/bash
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -J CropNorCP
#SBATCH -e slurm_error.txt
#SBATCH -o slurm_output.txt
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log_stats/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
###SBATCH --qos=low
#SBATCH -A rossby


# Define the base input directory
EXP='NorCP'
#EXP='FPS'
#DOMAIN="Emilia_Romagna"
#DOMAIN="NorCP_SSE"
DOMAIN="NorCP_full"
resolution="3km" #12km
#GCM="ECMWF-ERAINT" 
#GCM="ICHEC-EC-EARTH_HIST" 
#GCM="ICHEC-EC-EARTH_RCP85_MC"
#GCM="ICHEC-EC-EARTH_RCP85_LC"
#GCM="ICHEC-EC-EARTH_RCP45_MC"
GCM="ICHEC-EC-EARTH_RCP45_LC"
input_base_dir="/nobackup/rossby26/users/sm_fuxwa/AI/"${DOMAIN}"/original/"${GCM}"/"${resolution}
output_base_dir="/nobackup/rossby26/users/sm_fuxwa/AI/"${DOMAIN}"/cropped/"${GCM}"/"${resolution}

mkdir -p "$output_base_dir"  # Create output base directory if it doesn't exist

# Define the cropping range (x_start:x_end, y_start:y_end)
#if [[ "$EXP" == "FPS" ]]; then
if [[ "$DOMAIN" == "Emilia_Romagna" ]]; then
    if [[ "$resolution" == "12km" ]]; then
        x_range="1,106"   # Replace with your desired range
        y_range="1,88"    # Replace with your desired range
    elif [[ "$resolution" == "3km" ]]; then
        x_range="1,424"   # Replace with your desired range
        y_range="1,352"   # Replace with your desired range
    fi
#elif [[ "$EXP" == "NorCP" ]]; then
elif [[ "$DOMAIN" == "NorCP_SSE" ]]; then
    if [[ "$resolution" == "12km" ]]; then
        x_range="1,17"   # Replace with your desired range
        y_range="1,23"   # Replace with your desired range
    elif [[ "$resolution" == "3km" ]]; then
        x_range="4,71"   # Replace with your desired range
        y_range="1,92"   # Replace with your desired range
    fi
elif [[ "$DOMAIN" == "NorCP_full" ]]; then
    if [[ "$resolution" == "12km" ]]; then
        x_range="1,155"   # Replace with your desired range
        y_range="1,209"   # Replace with your desired range
    elif [[ "$resolution" == "3km" ]]; then
        x_range="2,621"   # Replace with your desired range
        y_range="2,837"   # Replace with your desired range
    fi
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
    #for file in "$input_dir"/*950*_*.nc; do
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



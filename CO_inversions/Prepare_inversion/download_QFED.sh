#!/bin/bash

# ######################################
# Download QFED biomass burning dataset
# ######################################

# Loop over years
for yyyy in {2023..2023}; do
    # Loop over months
    for mm in {09..09}; do
        # Get the number of days in the current month
        days_in_month=$(cal $mm $yyyy | awk 'NF {DAYS = $NF}; END {print DAYS}')

        # Loop over days in the current month
        for dd in $(seq -w 01 $days_in_month); do
            url="https://portal.nccs.nasa.gov/datashare/iesa/aerosol/emissions/QFED/v2.6r1/0.25/QFED/Y${yyyy}/M${mm}/qfed2.emis_co.061.${yyyy}${mm}${dd}.nc4"
            locPath="/u/bbyrne1/BiomassBurning_datasets/QFED_v2.6r1_0.25res/Y${yyyy}/M${mm}/"
            echo "Downloading: ${url}"
            wget -P "$locPath" "$url"
        done
    done
done

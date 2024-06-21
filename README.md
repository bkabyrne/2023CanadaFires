## README

**Code to reproduce the results of "Vast 2023 Canadian forest fire carbon emissions"**
**contact: Brendan Byrne**
**email: brendan.k.byrne@jpl.nasa.gov**

Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

## Overview:

This repository contains python programs used to 1. Process TROPOMI CO retrievals (./TROPOMI_superobs), 2. generate the input data for the CO inversions (./CO_inversions/Prepare_inversion), 3. process the output from the CO inversions (CO_inversions/Process_output), and 4. make figures shown in the paper (plot_figures). Climate data were downloaded as described in the methods sections. Inverse estimates can be reproduced using the GEOS-Chem adjoint, which can be downloaded from here: http://wiki.seas.harvard.edu/geos-chem/index.php/Quick_Start_Guide, following the inversion set-up described in the methods. All of the results of the manuscript can be reproduced from the data provided (provided separately upon manuscript publication). The python codes can reproduce the figures from the manuscript from those data.

# Contents:

-----
## Contents of ./CO_inversions/Prepare_inversion
- **download_GFAS.py:**
> Download GFAS fire inventory
- **download_QFED.sh:**
> Download QFED fire inventory
- **regrid_QFED_to_2x25.py:**
> Regrid QFED to 2 x 2.5
- **regrid_GFAS_to_2x25.py:**
> Regrid GFAS to 2 x 2.5
- **write_total_prior_with_3Day_UNCr_flux.py:**
> Combines prior fire, fossil and biomass fluxes for inversion (3-day optimization)
- **create_fluxes_with_injh.py:**
> Maps fire emissions to injection height from GFAS IS4fires
- **write_fire_emissions_at_injection_height.py:**
> Adds injection height to emission data (for sensitivity tests)
- **write_otherFlux_emissions_posterior.py**
> Writes fire and biomass emissions (used for tests with fire emitted at injection height)
- **prepare_SF_perturb.py:**
> Writes perturbed fluxes for Monte Carlo experiments (3-day optimization)
- **prepare_SF_perturb_7day.py:**
> Writes perturbed fluxes for Monte Carlo experiments (7-day optimization)

-----
## Contents of ./CO_inversions/Process_output
- **write_posterior_fire_COandCO2_3day_MonteCarlo.py:**
> Writes posterior fire for Monte Carlo ensemble (3-day)
- **write_posterior_fire_COandCO2_7day_MonteCarlo.py:**
> Writes posterior fire for Monte Carlo ensemble (7-day)
- **write_posterior_fire_COandCO2_3day.py:**
> Writes posterior fire emissions (3-day opt)
- **write_posterior_fire_COandCO2_7day.py:**
> Writes posterior fire emissions (7-day opt)
- **write_posterior_otherFlux_COandCO2_7day.py:**
> Writes posterior non-fire emissions (7-Day opt)
- **write_posterior_otherFlux_COandCO2_3day.py:**
> Writes posterior non-fire emissions (3-Day opt)
- **write_for_archiving.py:**
> Writes CO and CO2 emissions for archiving (easy use format)

-----
## Contents of ./TROPOMI_superobs
- **calc_TROPOMI_mole_fractions_2x25.py:**
> Maps TROPOMI obs to GEOS-Chem grid and calculates super-obs
- **Scripts for generating representativeness errors:**
  1. **write_daily_YHx_prior.py:**
  > Save simulated CO data
  2. **calculate_representativeness_error_Heald.py:**
  > Calculates representativeness errors from simulated CO data
  3. **write_TROPOMI_w_representativeness_errors.py:**
  > Re-writes TROPOMI super-obs with representativeness errors 
- **make_TROPOMI_MonteCarlo.py:**
> Calculates TROPOMI data with perturbations for Monte Carlo tests

-----
## Contents of ./plot_figures
- **plot_Fire_emissions_revised.py:**
> Creates figures 1, S1 and S2
- **plot_climate_anomalies.py:**
> Creates figures 2, S4 and S5
- **plot_Fire_and_climate.py:**
> Creates figures 3 and S6
- **plot_inventory.py:**
> Creates figure 4
- **plot_FigS3.py**
> Creates figure S3
- **plot_posterior_TROPOMI_maps.py**
> Creates figures S7-S8
- **plot_TCCON_cosamples.py**
> Creates figures S9-S10
- **plt_CMPI5_data.py:**
> Creates figure S11
- **plot_FireMaps_FigSY.py**
> Creates figure S13
- **plot_Canada_Managed_land.py:**
> Regrids managed land mask and makes some plots (which are combined to make Fig. S12)


## Required content:

1. System Requirements

This code has been tested on the NASA Pleiades computing system (Linux). The python routined were run using python3/3.9.12

2. Installation guide

No instillation is required. This code should run on any system with python3/3.9.12 and loaded modules

3. Demo

Python routines can be run to re-create all of the figures in the paper. All of the python codes should run in under 10 minutes except TROPOMI super-obs aggregation code.

4. Instructions for use

Execute each python routine and the Figures shown in the manuscript and supplement will be written to the Figures sub-directory. The header of each python code describes the figures made by that particular code.


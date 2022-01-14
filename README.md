# TomoMIST
Wrapper for processing a large number of projections with MIST.

The MIST_II (directional) technique is presented in Pavlov, K. M., D. M. Paganin, K. S. Morgan, H. Li, S. Berujon, L. Quénot and E. Brun (2021). "Directional dark-field implicit x-ray speckle tracking using an anisotropic-diffusion Fokker-Planck equation." Physical Review A 104(5): 053505.

These Python functions permit to process large consecutive numbers of projections.
The Shell scripts permit to distribute the work on parallel jobs using the OAR task manager of the ESRF clusters.

The core function were written initially by L. Quenot and are also available at at https://github.com/DoctorEmmetBrown/popcorn/tree/main/popcorn/phase_retrieval  as part of a larger package for phase retrieval in X-ray imaging. 


The two main functions for processing projections MISTII_1 & 2 differs a bit though from the implementation found in the popcorn package. I have PARALLELIZED the core calculation (the resolution of the systems for each pixels) in order to greatly speed up the calculation. This version should be faster if you’re working on a multicore architecture (it will use all your available CPUs).





Files: 
Tomo_MIST.ini: is an initialization file where you should fill your case with the parameters. Use the template already available.
TomoMIST.py: process the projections
            argin: $1 = INIFILEPATH, $2 = STUDYCASE (name as in the ini file), $3 = first projection number in the scan, $4 = last projection number
OAR_general.sh: script generating a job for each OAR core. Each job will ask for 8 cores on the clusters as to also limit the memory demanded on each machine.
OARsubmit.rec: should be called to call for the processing of the case on OAR, it will call OAR_general.sh to generate the jobs.
=> INIFILEPATH & PYTOMOSCRIPT are two hard path that you should adjust in the script before using.

===================================================================================================================
TO USE:
Navigate to your folder.
Fill and check the INIFILEPATH & PYTOMOSCRIPT parameters in the OARsubmit.rec if you're using it for the first time.

Fill the ini file with your case, follow the other ones'structure.

Call OARsubmit.rec

With 2 parameters $1 = the case name as written in the ini file.
                      $2 = the number of projections in the scan (todo: to move to the ini file)

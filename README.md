# LSH-Product-Duplicate-Detection
This repository is used in the paper for the course "Computer Science for Business Analytics" (FEM21037). It combines LSH and MSMP methods to find duplicates products in a dataset of TVs on multiple websites.

The repository is easy-to-use and structured as follows:
- "data" contains the original dataset "TVs-all-merged" (1624 TV products, 1262 different model IDs, 399 duplicate pairs)
- "smalldata" contains a smaller version of this dataset with around 10% the size of the original one
- main.py is set up such that it can be run instantly, creating all necessary results for reproduction.
- The LSH, MSMP and MSMP-FAST algorithms can all be found in python files under the same name.
- All other .py files contain helper functions for the code to work, don't rename or move them!
- All plots and graphs in the paper can be found in the folder "data/plots/12-13-2024_04u18m16s".
- If you would like to replicate the results from scratch, simply create a new data folder and smalldata folder,
    copy the data files (both named "TVs-all-merged") to their paths and make sure to change the directories
    in the main.py file. These directories are initialised at the very top, so you don't need to search in the code!
    running on 50 bootstraps takes about 2 hours and 20 minutes, the print statements when running indicate where
    the results (graphs, figures and other plots) can be found for that run.
- This code has been run on a standard Intel i5 processor (CPU), which does not have a lot of computing power compared to GPUs.


# Data Analysis

This project contains Python scripts and Jupyter notebooks for analysis of publicly available target-disease
associations datasets provided by the Open Targets team.

# Project Structure

The overall structure of the project is as follows:

```bazaar

project
│   README.md
│   main.py
│   download_datasets.py
│   Journey.ipynb
│   disease_target.json
│   evidence_stats.json
│
└───datasets
        └───evidences
        └───diseases
        └───targets
```

The two Python scripts `main.py` and `download_datasets.py` are main scripts for downloading datasets and preforming
analysis. Final results of the analysis is saved in `evidence_stats.json` and `disease_target.json`. The former contains
Json objects of evidence statistics, while the later include disease-target pair stats. File `Journey.ipynb` contains a comprehensive description of my approach for this project. I tried to compare
different options and justify my decisions.
# How to run the script

In order to run the `main.py` script, you should follow these steps:

1. Clone this repository and change your current working directory as follows

```bash
git clone https://github.com/arman-yekkehkhani/data_analysis
cd data_analysis
```

2. Create a new Python virtual environment and install the dependencies from `requirements.text`. Here, I use
   Python >= 3.7 and pip as a package manager.

```bash
python3 venv -m env
source env/bin/activate
pip install -r requirements.txt
```

3. run `main.py` with desired args.
```bash
python main.py

# removes directories containing the datasets in the current dir
# fetch datasets
python main.py --over-write true
```

# Expected Output

The final output of each files is as follows:

1.evidence_stats.json
```json
{
   "diseaseId":"EFO_0000095",
   "targetId":"ENSG00000082898",
   "median":0.7,
   "top3":[0.7,0.7,0.7]
}
...
```
2.disease_target.json
```json
{
   "diseaseId":"EFO_0003847",
   "targetId":"ENSG00000284299",
   "median":0.0
}
...
```
Another important result is the number of target-target pairs sharing a connection to at least two diseases, which is
printed when `main.py` is finished.
```json
Number of target-target pairs share a connection to at least two diseases : 142015
```




# Data-Challenge-2-code

## The data
The following datasets have been changed or collected by hand and therefore are already in this github so you do not have to download them:
- PAS question dataset (No link available)
- The UK police data repository (https://data.police.uk/data/archive/)

The following datasets should be downloaded yourself and placed in the `data` folder:
- PAS survey data (https://data.london.gov.uk/dataset/mopac-surveys, the .xlsx file)
- Ethnic groups by Borough (https://data.london.gov.uk/dataset/ethnic-groups-borough)

## Before you start
This github provides the user with both .py and jupiter notebook files. To run the .py files, all necessary packages are in the `requirements.txt` file.

## loading and cleaning the data
Run the following files in order:
- `load_data_to_SQL.py`
- `BoroughFinder.py`
- `clean_data.py`

This repository contains the template code for the TU/e course JBG050 Data Challenge 2.
Please read this document carefully as it has been filled out with important information.

## How to clone the project
1. Copy the following HTTPS key: https://github.com/stiverthijmen163/DC2-Group-15.git
2. Make a folder in which you want to store the project
3. Open cmd
4. Type ‘git clone https://github.com/stiverthijmen163/DC2-Group-15.git’

## Starting Jupyter Lab (Optional if you would like to use notebooks! Not useful when you work with .py files, use VS code or Pycharm instead)
1. Open Anaconda and launch Jupyter Lab
2. When Jupyter Lab is launches, nagivate the the cloned project

## Pusing from Jupyter Lab when no new files are added*
1. Start a new terminal session inside Jupyter Lab (new launcher tab, next to your python tab)
2. Type: git commit -m “message”
3. Type: git push
*When using gitlab TU/e for the first time, you might get a pop up to login. Your username is your 20----- number (not your student number!), password is your regular TU/e password.

## Pusing when new files are added:
1. Start a new terminal session inside Jupyter Lab if noone is active (new launcher tab, next to your python tab)
2. Type: git add .
3. Type: git commit -m “message”
4. Type: git push

## Pulling:
1. Start a new terminal session inside Jupyter Lab if noone is active (new launcher tab, next to your python tab)
2. Type: git pull origin master
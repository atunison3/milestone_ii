# Milestone 2️⃣
Collaboration repo for MADS Milestone II

## Setting up the Virtual Environment

Please note, this project requires Python 3.10.

### Windows
1. Create the virtual environment:
  ```sh
  python3.10 -m venv .venv
  ```
2. Activate the virtual environment:
  ```sh
  .venv\Scripts\activate
  ```
3. Install requirements.txt
  ```sh
  pip install -r requirements.txt
  ```
4. Build source code and install local pacakge
  ```sh
  python -m build
  pip install -e .
  ```
  
### Mac
1. Create the virtual environment:
  ```sh
  python3.10 -m venv .venv
  ```
2. Activate the virtual environment:
  ```sh
  source .venv/bin/activate
  ```
3. Install requirements.txt
  ```sh
  pip install -r requirements.txt
  ```
4. Build source code and install local pacakge
  ```sh
  python -m build
  pip install -e .
  ```

## How To Download the Data 📊
1. Ensure the virtual environment is activated.
2. Run the main script:
  ```sh
  python main.py
  # data should download under /data
  ```
3. Feel free to use any of the exploratory notebooks to check out the data

> Data will automatically be downloaded within the main script. There will be no CSVs stored directly in this repo


## Supervised Learning

Running the supervised learning portion of this project is accomplished by running `supervised.py`. 
This file has two arguments (`path_to_assets`, and `path_to_results`). `path_to_assets` has a 
default of `assets` as an assets directory was established under team17 (see troubleshooting for 
more info). The path to assets is a string that points to a folder with all the evwatts data in it. 
The `path_to_results` is also a string. It points to a directory on your computer that will be 
used to store the results generated from the analysis. 

## Unsupervised Learning

Unsupervised learning is performed very much the same way as supervised. Running the `unsupervised.py` 
module will take care of the whole thing. This module has a few arguments. The big two are the 
`path_to_results` and `path_to_assets`. These are set up just like the supervised learning. 

## Troubleshooting 

Due to time constraints, insufficient testing was performed. Scripts were typically run in the 
directory they reside. The `.gitignore` file ignores all folders named `assets`. During 
development, a directory `assets` with all the evwatts data was established in the `team17` 
directory. 
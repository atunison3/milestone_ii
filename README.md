# Milestone 2️⃣
Collaboration repo for MADS Milestone II

## Setting up the Virtual Environment

### Windows
1. Create the virtual environment:
  ```sh
  python -m venv .venv
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
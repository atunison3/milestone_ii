import subprocess

def get_data() -> None:
  """
  This function runs a shell script to download data from the EV Watts API.
  The data is saved in the 'data' folder.
  """

  # Define the shell script to run
  shell_script = './evwatts.public.sh'

  # Run the shell script
  subprocess.run(['sh', shell_script], check=True)

  print(f"Data has been downloaded and saved in the data folder.")

if __name__ == '__main__':
  get_data()
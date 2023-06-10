import subprocess

for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023]:
    cmd = ['python', 'run_scraping.py', year]
    subprocess.Popen(cmd)

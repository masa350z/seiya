import subprocess

for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023]:
    for d in range(31):
        cmd = ['python', 'run_scraping.py', str(year), str(d)]
        subprocess.Popen(cmd)

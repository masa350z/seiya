import subprocess

for d in range(31):
    cmd = ['python', 'run_scraping.py', str(d+1)]
    subprocess.Popen(cmd)

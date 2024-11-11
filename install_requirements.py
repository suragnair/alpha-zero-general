import subprocess

with open('requirements.txt', "r") as f:
    requirements = f.readlines()

for requirement in requirements:
    requirement = requirement.strip()
    try:
        subprocess.check_call(["pip", "install", requirement])
        print(f"Installed {requirement}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {requirement}")
        continue
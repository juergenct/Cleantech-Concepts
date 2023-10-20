import os
import pandas as pd
import pyalex
from joblib import Parallel, delayed

# Create a list of fake emails
emails = [
    "hallo61@byom.de",
    "hallo62@byom.de",
    "hallo63@byom.de",
    "hallo64@byom.de",
    "hallo65@byom.de",
    "hallo66@byom.de",
    "hallo67@byom.de",
    "hallo68@byom.de",
    "hallo69@byom.de",
    "hallo70@byom.de",
    "hallo71@byom.de",
    "hallo72@byom.de",
    "hallo73@byom.de",
    "hallo74@byom.de",
    "hallo75@byom.de",
    "hallo76@byom.de",
    "hallo77@byom.de",
    "hallo78@byom.de",
    "hallo79@byom.de",
    "hallo80@byom.de",
    "hallo81@byom.de",
    "hallo82@byom.de"
]

# Create a list of all Cleantech works
df = pd.read_csv('/Users/juergenthiesen/Documents/Patentsview/Cleantech Concepts/df_oaid_Cleantech_Y02.csv')
works = df['oaid'].tolist()

# Create a list of all commands
commands = []
for i in range(0, 20):
    command = f"python institutions_works.py --email {emails[i]} --csv data/chunks/institutions_filtered_{i}.csv --chunk {i}"
    commands.append(command)

os._exit(1)
    
# Run the commands in parallel
try:
    Parallel(n_jobs=6)(delayed(os.system)(command) for command in commands)
except Exception as e:
    print("Error!")
    print(e)
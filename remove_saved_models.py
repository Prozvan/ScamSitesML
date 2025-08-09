import json
import os
import shutil




# Read json file
with open('stats.json', 'r') as f:
    data = json.load(f)


# Get all paths that are records in stats
PATHS = [data[i]["file_name"] for i in data]



# Go through all saved models
DIR = "saved_models"
for folder in os.listdir(DIR):
    
    if folder not in PATHS:
        shutil.rmtree(os.path.join(DIR, folder))
        print(f"Removed: {folder}")
        

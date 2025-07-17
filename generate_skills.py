import pandas as pd
import json

# Load the file (tab-separated)
df = pd.read_csv("Skills.txt", sep="\t", usecols=["Element Name"])

# Clean and deduplicate skill names
skills = df["Element Name"].dropna().unique().tolist()

# Optional: sort them alphabetically
skills = sorted(set(skills))

# Save to JSON
with open("onet_skills.json", "w") as f:
    json.dump(skills, f, indent=2)

print(f"✅ Extracted {len(skills)} unique skills from O*NET.")

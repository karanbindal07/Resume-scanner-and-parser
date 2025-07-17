import pandas as pd
import json
import re

# 1. Load your job descriptions CSV
df = pd.read_csv("ultra_diverse_job_descriptions.csv")

# Print available column names for clarity
print("📌 CSV Columns:", df.columns.tolist())

# Replace with the actual column name for job descriptions
column_name = "description"  # Change this based on the printed columns

# 2. Extract non-empty JDsimport pandas as pd
import json
import re

# 1. Load your job descriptions CSV
df = pd.read_csv("ultra_diverse_job_descriptions.csv")

# Print available column names for clarity
print("📌 CSV Columns:", df.columns.tolist())

# Replace with the actual column name for job descriptions
column_name = "description"  # Change this based on the printed columns

# 2. Extract non-empty JDs
job_descriptions = df[column_name].dropna().tolist()
total = len(job_descriptions)
print(f"📄 Loaded {total} job descriptions.\n")

# 3. Load your O*NET skill list
with open("onet_skills.json", "r") as f:
    all_skills = json.load(f)

# 4. Skill matching function
def extract_skills(text, skills):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    matched = []
    for skill in skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text):
            matched.append(skill)
    return matched

# 5. Process and show progress every 10
training_data = []

for i, jd in enumerate(job_descriptions):
    matched_skills = extract_skills(jd, all_skills)
    training_data.append({
        "job_description": jd,
        "skills": matched_skills
    })

    # Progress update every 10
    if (i + 1) % 10 == 0 or (i + 1) == total:
        print(f"✅ Processed {i + 1}/{total} job descriptions", flush=True)

# 6. Save result
with open("jd_skill_training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

print("\n🎉 Done! Saved labeled training data to jd_skill_training_data.json")

job_descriptions = df[column_name].dropna().tolist()
total = len(job_descriptions)
print(f"📄 Loaded {total} job descriptions.\n")

# 3. Load your O*NET skill list
with open("onet_skills.json", "r") as f:
    all_skills = json.load(f)

# 4. Skill matching function
def extract_skills(text, skills):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    matched = []
    for skill in skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text):
            matched.append(skill)
    return matched

# 5. Process and show progress every 10
training_data = []

for i, jd in enumerate(job_descriptions):
    matched_skills = extract_skills(jd, all_skills)
    training_data.append({
        "job_description": jd,
        "skills": matched_skills
    })

    # Progress update every 10
    if (i + 1) % 10 == 0 or (i + 1) == total:
        print(f"✅ Processed {i + 1}/{total} job descriptions", flush=True)

# 6. Save result
with open("jd_skill_training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

print("\n🎉 Done! Saved labeled training data to jd_skill_training_data.json")

import json
import re

# 1. Load original labeled data
with open("jd_skill_training_data.json", "r") as f:
    data = json.load(f)

# 2. Load new skills (missed ones)
with open("missing_tools_full.txt", "r") as f:
    missed_skills = [line.strip() for line in f if line.strip()]

# 3. Add missed skills where matched
for i, item in enumerate(data):
    jd_text = item["job_description"].lower()
    current_skills = set([s.lower() for s in item.get("skills", [])])

    for skill in missed_skills:
        skill_clean = skill.lower()
        if re.search(r'\b' + re.escape(skill_clean) + r'\b', jd_text):
            if skill_clean not in current_skills:
                item["skills"].append(skill)

    if (i + 1) % 10 == 0 or (i + 1) == len(data):
        print(f"🔍 Updated {i + 1}/{len(data)} job descriptions", flush=True)

# 4. Save the updated result
with open("jd_skill_training_data_updated.json", "w") as f:
    json.dump(data, f, indent=2)

print("\n✅ Done! Saved updated file as jd_skill_training_data_updated.json")


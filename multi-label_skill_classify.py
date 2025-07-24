import json

jd_file = "jd_skill_training_data_updated.json"
vocab_file = "skill_vocab.json"

# Load JD data
with open(jd_file, "r") as f:
    data = json.load(f)

# Extract unique skills
all_skills = set()
for entry in data:
    all_skills.update([s.lower() for s in entry["skills"]])

# Map each skill to an ID
skill_to_id = {skill: idx for idx, skill in enumerate(sorted(all_skills))}

# Save to JSON
with open(vocab_file, "w") as f:
    json.dump(skill_to_id, f, indent=2)

print(f"✅ Saved {len(skill_to_id)} skills to {vocab_file}")

import json
import os
import re
import random
from tqdm import tqdm

# === Parameters ===
resume_dir = "generated_resumes"
jd_skill_path = "jd_skill_training_data_updated.json"
output_dir = "chunked_training_data"
chunk_size = 200
jd_limit = 5000
resume_start_chunk = 32  # ⬅️ Start from chunk 32

# === Make sure output folder exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load or generate random JD sample ===
jd_sample_path = os.path.join(output_dir, "sampled_5000_jds.json")

if os.path.exists(jd_sample_path):
    with open(jd_sample_path, "r") as f:
        jd_skill_data = json.load(f)
    print("📦 Loaded existing random 5k JDs.")
else:
    with open(jd_skill_path, "r") as f:
        all_jds = json.load(f)
        jd_skill_data = random.sample(all_jds, jd_limit)
    with open(jd_sample_path, "w") as f:
        json.dump(jd_skill_data, f, indent=2)
    print("🎲 Random 5k JDs selected and saved for consistency.")

# === Helper: clean text ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

# === Load resumes ===
resume_files = sorted([f for f in os.listdir(resume_dir) if f.endswith(".txt")])
total_chunks = len(resume_files) // chunk_size + (1 if len(resume_files) % chunk_size else 0)

# === Resume from desired chunk ===
for chunk_idx in range(resume_start_chunk - 1, total_chunks):
    chunk_start = chunk_idx * chunk_size
    chunk_end = min(chunk_start + chunk_size, len(resume_files))
    chunk_resumes = resume_files[chunk_start:chunk_end]

    chunk_filename = f"chunk_{chunk_idx+1}.json"
    chunk_path = os.path.join(output_dir, chunk_filename)

    if os.path.exists(chunk_path):
        print(f"✅ Skipping {chunk_filename} (already exists)")
        continue

    print(f"🚀 Processing Chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_resumes)} resumes)...")
    chunk_data = []

    for resume_file in tqdm(chunk_resumes, desc=f"🔍 Chunk {chunk_idx+1}"):
        with open(os.path.join(resume_dir, resume_file), "r") as f:
            resume_text = f.read()
        cleaned_resume = clean_text(resume_text)

        for jd_entry in jd_skill_data:
            jd_text = jd_entry["job_description"]
            true_skills = list(set([s.lower() for s in jd_entry["skills"]]))

            matched_skills = [s for s in true_skills if re.search(r'\b' + re.escape(s) + r'\b', cleaned_resume)]
            missing_skills = list(set(true_skills) - set(matched_skills))

            chunk_data.append({
                "resume": resume_text,
                "job_description": jd_text,
                "true_skills": true_skills,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills
            })

    with open(chunk_path, "w") as f:
        json.dump(chunk_data, f, indent=2)

    print(f"✅ Saved {len(chunk_data)} records to {chunk_filename}")

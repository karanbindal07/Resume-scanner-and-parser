import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os

# === Model ===
from model import ResumeJDMatcher  # Make sure this imports your model architecture

# === Config ===
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_SKILLS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_DIR = "chunked_training_data"
VOCAB_PATH = "skill_vocab.json"

# === Load skill vocab ===
with open(VOCAB_PATH) as f:
    skill_to_id = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === Dataset ===
class ResumeJDDataset(Dataset):
    def __init__(self, data, skill_to_id):
        self.data = data
        self.skill_to_id = skill_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        resume_text = sample["resume"]
        jd_text = sample["job_description"]
        true_skills = sample["true_skills"]

        resume_enc = tokenizer(resume_text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
        jd_enc = tokenizer(jd_text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

        label_vector = torch.zeros(NUM_SKILLS)
        for s in true_skills:
            skill_id = self.skill_to_id.get(s.lower())
            if skill_id is not None:
                label_vector[skill_id] = 1

        return {
            "resume_input": {k: v.squeeze(0) for k, v in resume_enc.items()},
            "jd_input": {k: v.squeeze(0) for k, v in jd_enc.items()},
            "labels": label_vector
        }

# === Initialize Model, Loss, Optimizer ===
model = ResumeJDMatcher(num_skills=NUM_SKILLS).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
regression_loss = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# === Training Loop Over Chunks ===
chunk_files = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".json") and f.startswith("chunk_")])

for epoch in range(EPOCHS):
    print(f"\n🌍 Starting Epoch {epoch + 1}/{EPOCHS}")

    for chunk_file in chunk_files:
        print(f"\n📦 Loading {chunk_file}")
        with open(os.path.join(CHUNK_DIR, chunk_file), "r") as f:
            chunk_data = json.load(f)

        dataset = ResumeJDDataset(chunk_data, skill_to_id)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        total_samples = len(dataset)

        with tqdm(total=total_samples, desc=f"🔁 Training {chunk_file}", unit="samples") as pbar:
            for batch in dataloader:
                model.train()
                resume_input = {k: v.to(DEVICE) for k, v in batch["resume_input"].items()}
                jd_input = {k: v.to(DEVICE) for k, v in batch["jd_input"].items()}
                labels = batch["labels"].to(DEVICE)

                optimizer.zero_grad()
                skill_logits, effectiveness_score = model(resume_input, jd_input)

                loss1 = criterion(skill_logits, labels)
                dummy_score = torch.rand_like(effectiveness_score)
                loss2 = regression_loss(effectiveness_score, dummy_score)

                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item(), skill_loss=loss1.item(), eff_loss=loss2.item())
                pbar.update(len(labels))

# === Save the model ===
torch.save(model.state_dict(), "resume_jd_model_trained_on_chunks.pth")
print("✅ Model training complete and saved.")

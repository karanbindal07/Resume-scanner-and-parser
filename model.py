import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# === Config ===
MODEL_NAME = "distilbert-base-uncased"
HIDDEN_SIZE = 768
DROPOUT = 0.3
NUM_SKILLS = 1000  # Number of unique skills in your label space

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === Multi-Layer Cross Attention Block ===
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, q, k, v):
        attn_output, _ = self.attn(q, k, v)
        out = self.norm(q + self.dropout(attn_output))
        return out

# === Deep Resume-JD Matching Model ===
class ResumeJDMatcher(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_skills=NUM_SKILLS):
        super().__init__()
        self.resume_encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.jd_encoder = AutoModel.from_pretrained(MODEL_NAME)

        # Stack multiple cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_size) for _ in range(3)
        ])

        # MLP Heads
        self.skill_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, num_skills)
        )

        self.effectiveness_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, resume_input, jd_input):
        # Encode text
        resume_out = self.resume_encoder(**resume_input).last_hidden_state  # (B, L, H)
        jd_out = self.jd_encoder(**jd_input).last_hidden_state

        # Cross-attention
        x = resume_out
        for layer in self.cross_attn_layers:
            x = layer(x, jd_out, jd_out)

        # Pooling (mean pooling)
        pooled = x.mean(dim=1)

        # Outputs
        skill_logits = self.skill_classifier(pooled)
        effectiveness_score = self.effectiveness_regressor(pooled).squeeze(-1)

        return skill_logits, effectiveness_score


# === Example Usage ===
def tokenize_inputs(resume_text, jd_text):
    resume_input = tokenizer(resume_text, padding=True, truncation=True, return_tensors="pt")
    jd_input = tokenizer(jd_text, padding=True, truncation=True, return_tensors="pt")
    return resume_input, jd_input


# === Instantiate the model ===
model = ResumeJDMatcher()

# === Forward pass demo ===
resume_sample = "Experienced Python developer skilled in machine learning, TensorFlow, and SQL."
jd_sample = "Looking for a backend engineer with knowledge of ML, databases, and deep learning."

resume_input, jd_input = tokenize_inputs(resume_sample, jd_sample)

with torch.no_grad():
    skill_logits, effectiveness_score = model(resume_input, jd_input)

print("🔧 Skill Predictions (Raw Logits):", skill_logits)
print("📈 Effectiveness Score:", effectiveness_score.item())

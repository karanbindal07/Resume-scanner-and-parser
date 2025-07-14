import random
import pandas as pd

# Define roles, industries, responsibilities, requirements
job_roles = [
    "Software Engineer", "Data Scientist", "Product Manager", "UX Designer",
    "Marketing Manager", "Sales Representative", "DevOps Engineer",
    "Cybersecurity Analyst", "Finance Analyst", "HR Business Partner",
    "Project Manager", "Interior Designer", "Mechanical Engineer",
    "Electrical Engineer", "Pharmacist", "Nurse Practitioner",
    "Clinical Psychologist", "Lawyer", "Digital Content Creator",
    "Video Game Developer"
]

industries = [
    "FinTech", "Healthcare", "Education", "E‑commerce", "Gaming",
    "Automotive", "Retail", "Telecommunications", "Government", "Energy",
    "Logistics", "Entertainment", "Hospitality", "Manufacturing", "Biotech",
    "Non-profit", "Aerospace", "Construction", "Real Estate", "Insurance"
]

responsibilities_pool = [
    "Lead the design and development of scalable solutions.",
    "Analyze complex datasets and generate data-driven insights.",
    "Manage cross-functional teams and align objectives.",
    "Design user-centered interfaces and improve usability.",
    "Develop marketing strategies and execute campaigns.",
    "Generate sales leads and build strong client relationships.",
    "Optimize CI/CD pipelines and automate system deployments.",
    "Monitor security breaches and implement preventative measures.",
    "Perform financial modeling and risk assessments.",
    "Manage employee relations and support HR initiatives.",
    "Plan and execute project implementations within scope and budget.",
    "Conceptualize interior layouts and coordinate with vendors.",
    "Design and prototype mechanical systems and components.",
    "Develop electrical systems and ensure compliance with standards.",
    "Dispense medications and counsel patients on proper usage.",
    "Provide primary healthcare services and manage patient care.",
    "Conduct therapeutic sessions and psychological assessments.",
    "Advise clients on legal matters and draft documentation.",
    "Produce engaging digital content across platforms.",
    "Write code for core gameplay mechanics and design engine features."
]

requirements_pool = [
    "Bachelor’s degree in relevant field (Engineering, Business, Psychology, etc.)",
    "5+ years of experience in a similar role or industry.",
    "Strong analytical and problem-solving skills.",
    "Excellent verbal and written communication abilities.",
    "Proficiency with industry-specific tools (e.g., AWS, AutoCAD, SAP).",
    "Demonstrated leadership capabilities in team environments.",
    "Experience managing projects using Agile or Waterfall frameworks.",
    "Understanding of compliance and regulatory requirements.",
    "Ability to work in fast-paced, dynamic environments.",
    "Strong attention to detail and organizational skills.",
    "Proficiency in programming languages or design software.",
    "Certification preferred (PMP, AWS Certified, CPA, LEED, etc.).",
    "Ability to manage budgets and vendor relationships."
]

# Generate 100 descriptions
descriptions = []
for i in range(1, 10001):
    role = random.choice(job_roles)
    industry = random.choice(industries)
    responsibilities = random.sample(responsibilities_pool, k=4)
    requirements = random.sample(requirements_pool, k=4)
    
    desc = f"Job ID: {i}\n"
    desc += f"Role: {role} in {industry}\n\n"
    desc += "Responsibilities:\n"
    for r in responsibilities:
        desc += f"- {r}\n"
    desc += "\nRequirements:\n"
    for r in requirements:
        desc += f"- {r}\n"
    
    descriptions.append({
        "job_id": i,
        "role": role,
        "industry": industry,
        "description": desc.strip()
    })

# Save as DataFrame
df = pd.DataFrame(descriptions)
df.to_csv("job_descriptions.csv", index=False)
print("✅ Generated 100 job descriptions and saved to job_descriptions.csv")

import os
import csv
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

prompt = """
Generate realistic 20 synthetic user investment profiles for a robo-advisor app.
Each profile should include:
user_id, age (25-70), annual_income (USD, 20000-200000),
risk_tolerance (1-10), investment_horizon (years, 1-30),
savings (USD), financial_goal (retirement, house, education, wealth_growth).
Return as CSV rows without headers or explanations.
"""

response = model.generate_content(prompt)
output_text = response.text.strip()

output_file = os.path.join(os.path.dirname(__file__), "user_profiles.csv")
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for line in output_text.split("\n"):
        writer.writerow(line.split(","))

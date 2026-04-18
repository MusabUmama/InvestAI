import os
import csv
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

prompt = """
Generate 20 realistic synthetic user portfolios.
Each row should include:
portfolio_id, user_id, asset_symbol, allocation_percentage (0-100).
Ensure allocations per portfolio add up to 100%.
Return as CSV rows without headers or explanations.
"""

response = model.generate_content(prompt)
output_text = response.text.strip()

output_file = os.path.join(os.path.dirname(__file__), "portfolios.csv")
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for line in output_text.split("\n"):
        writer.writerow(line.split(","))
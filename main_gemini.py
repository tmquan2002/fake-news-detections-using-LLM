import os
import textwrap
import google.generativeai as genai
from dotenv import load_dotenv # Thư viện đọc file .env

# Load API Key from file .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API Key not found, please check .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

cot_prompt = """
You are an expert AI Fact-Checker. Perform a Chain-of-Thought (CoT) analysis:
1. Claim Extraction
2. Logic & Style Analysis
3. Knowledge Verification
4. Final Verdict
News Snippet:
"""

sample_text = """
BREAKING NEWS: Pope Francis just shocked the world by explicitly endorsing Donald Trump. 
He stated Trump is "the only hope". Vatican officials confirmed this.
"""

print("Gemini is thinking...")
try:
    response = model.generate_content(cot_prompt + sample_text)
    print("\n" + "="*50)
    print(response.text)
    print("="*50)
    
    # Save result of predictions
    with open("images/gemini_output.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Result saved in: images/gemini_output.txt")

except Exception as e:
    print(f"Error: {e}")
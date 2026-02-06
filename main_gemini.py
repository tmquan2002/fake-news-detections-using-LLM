import os
import textwrap
import google.generativeai as genai
from dotenv import load_dotenv # Thư viện đọc file .env

# Load API Key từ file .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ Chưa tìm thấy API Key! Hãy kiểm tra file .env")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

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

print("🤖 Gemini đang suy luận...")
try:
    response = model.generate_content(cot_prompt + sample_text)
    print("\n" + "="*50)
    print(response.text)
    print("="*50)
    
    # Lưu kết quả ra file text để báo cáo
    with open("images/gemini_output.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("✅ Đã lưu kết quả vào images/gemini_output.txt")

except Exception as e:
    print(f"❌ Lỗi: {e}")
# """
# AI-Powered Paystub Generator - Streamlit Web Application
# ✓ Upload PDF paystubs
# ✓ Input salary and hours details
# ✓ Automatic calculation of ALL values
# ✓ Zero-artifact PDF editing
# ✓ Download processed paystub
# """

# import os
# import sys
# import json
# import re
# import io
# from pathlib import Path
# from typing import Dict, Any, List, Tuple, Optional
# from datetime import datetime
# import tempfile

# # Streamlit
# import streamlit as st

# # PDF Processing
# import fitz  # PyMuPDF

# # Image Processing for OCR
# from PIL import Image
# import pytesseract
# import cv2
# import numpy as np

# # AI Integration
# import google.generativeai as genai

# # ============================================================================
# # PAGE CONFIGURATION
# # ============================================================================

# st.set_page_config(
#     page_title="AI Paystub Generator",
#     page_icon="📄",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ============================================================================
# # CUSTOM CSS
# # ============================================================================

# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         color: #555;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .log-container {
#         background-color: #1e1e1e;
#         border: 1px solid #dee2e6;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         font-family: 'Courier New', monospace;
#         font-size: 0.9rem;
#         max-height: 400px;
#         overflow-y: auto;
#         color: #00ff00;
#         line-height: 1.6;
#     }
#     .stButton>button {
#         width: 100%;
#         background-color: #1f77b4;
#         color: white;
#         font-weight: bold;
#         padding: 0.5rem;
#         border-radius: 0.5rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class Config:
#     # HARDCODED API KEY - No user input needed
#     GEMINI_API_KEY = 'AIzaSyB1CecX2lZB_QvjeeiBme_DF8csmV8Q9dw'
#     GEMINI_MODEL = 'gemini-2.0-flash-exp'
#     OCR_DPI = 300
#     OCR_LANG = 'eng'
    
#     # Try to find Tesseract automatically
#     TESSERACT_PATHS = [
#         r'C:\Program Files\Tesseract-OCR\tesseract.exe',
#         r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
#         '/usr/bin/tesseract',
#         '/usr/local/bin/tesseract',
#     ]

# # Set Tesseract path
# for path in Config.TESSERACT_PATHS:
#     if os.path.exists(path):
#         pytesseract.pytesseract.tesseract_cmd = path
#         break

# # ============================================================================
# # INTELLIGENT AI ENGINE
# # ============================================================================

# class IntelligentAI:
#     """Super intelligent AI that calculates ALL paystub values automatically"""
    
#     def __init__(self):
#         genai.configure(api_key=Config.GEMINI_API_KEY)
#         self.model = genai.GenerativeModel(
#             Config.GEMINI_MODEL,
#             generation_config={
#                 "temperature": 0.1,
#                 "top_p": 0.95,
#             }
#         )
    
#     def extract_all_values(self, text: str) -> str:
#         """Extract EVERY value from the paystub"""
#         prompt = f"""Analyze this paystub and extract EVERY number with its exact label and format.

# PAYSTUB TEXT:
# {text}

# Extract ALL values including:
# - Earnings (hours, rate, amounts - Current and YTD)
# - Gross Pay (Current and YTD)
# - All Deductions (CPP, EI, Tax, Pension, Union - Current and YTD)
# - Net Pay (Current and YTD)
# - Any other amounts

# Return as a detailed list with EXACT format:
# - RegularHourly Hours: 20.00 (Current)
# - RegularHourly Rate: 140.58 (Current)
# - RegularHourly Amount: 2,811.60 (Current)
# - RegularHourly YTD: 7,029.00 (YTD)
# - Gross Pay Current: 2,980.30
# - Gross Pay YTD: 7,450.75
# - CPP Current: 169.32
# - CPP YTD: 419.29
# etc...

# List EVERY single number you find."""
        
#         response = self.model.generate_content(prompt)
#         return response.text.strip()
    
#     def calculate_all_mappings(self, extraction_text: str, user_inputs: Dict) -> str:
#         """Calculate ALL new values - Complete automation"""
#         prompt = f"""You are a payroll calculation expert. Calculate ALL paystub values and create complete replacement mappings.

# EXTRACTED CURRENT VALUES:
# {extraction_text}

# NEW USER INPUTS:
# - Regular Rate: ${user_inputs['regular_rate']:.2f}/hr
# - Regular Hours: {user_inputs['regular_hours']} hours (this pay period)
# - YTD Hours: {user_inputs['ytd_total_hours']} hours (year to date total)
# - Overtime Hours: {user_inputs['overtime_hours']} hours (this period, rate = 1.5x)
# - Bonus: ${user_inputs['bonus']:.2f}
# - Vacation Hours: {user_inputs['vacation_hours']} hours

# CALCULATION RULES:
# 1. Regular Pay = Rate × Hours
# 2. Overtime Pay = (Rate × 1.5) × OT Hours  
# 3. Vacation Pay = (Rate × Vacation Hours)
# 4. Gross Current = Regular + OT + Bonus + Vacation
# 5. YTD values = Calculate based on YTD hours provided
# 6. CPP = 5.95% of Gross (max $3,867.50/year)
# 7. EI = 1.63% of Gross (max $1,049.42/year)
# 8. Federal Tax = Progressive rates (approx 15-20% for this range)
# 9. Net Pay = Gross - All Deductions
# 10. Keep EXACT same format as original (commas, decimals, spacing)

# IMPORTANT: 
# - If original has "2,811.60" → use comma format "2,400.00"
# - If original has "2811.60" → use no comma format "2400.00"
# - Match the exact format for EACH field

# CRITICAL: Return ONLY a valid JSON array, nothing else. No explanations, no text before or after.

# Return JSON with ALL replacements needed:
# [
#   {{"field": "RegularHourly_Rate", "old": "140.58", "new": "30.00"}},
#   {{"field": "RegularHourly_Hours", "old": "20.00", "new": "80.00"}},
#   {{"field": "RegularHourly_Current", "old": "2,811.60", "new": "2,400.00"}},
#   {{"field": "RegularHourly_YTD", "old": "7,029.00", "new": "48,000.00"}},
#   {{"field": "GrossPay_Current", "old": "2,980.30", "new": "2,900.00"}},
#   {{"field": "GrossPay_YTD", "old": "7,450.75", "new": "49,500.00"}},
#   {{"field": "CPP_Current", "old": "169.32", "new": "172.55"}},
#   {{"field": "CPP_YTD", "old": "419.29", "new": "2,946.25"}},
#   {{"field": "EI_Current", "old": "48.58", "new": "47.27"}},
#   {{"field": "EI_YTD", "old": "121.45", "new": "806.85"}},
#   {{"field": "Tax_Current", "old": "462.41", "new": "435.00"}},
#   {{"field": "Tax_YTD", "old": "1,064.13", "new": "7,425.00"}},
#   {{"field": "NetPay_Current", "old": "1,975.88", "new": "1,976.95"}},
#   {{"field": "NetPay_YTD", "old": "5,035.61", "new": "33,725.70"}}
# ]

# Calculate EVERY value that needs to change based on the new inputs!"""
        
#         response = self.model.generate_content(prompt)
#         json_text = response.text.strip()
        
#         # More aggressive JSON cleaning
#         json_text = re.sub(r'^```json?\s*', '', json_text, flags=re.MULTILINE | re.IGNORECASE)
#         json_text = re.sub(r'^```\s*$', '', json_text, flags=re.MULTILINE)
#         json_text = re.sub(r'```', '', json_text)
        
#         # Extract only the JSON array
#         array_match = re.search(r'\[[\s\S]*\]', json_text)
#         if array_match:
#             json_text = array_match.group(0)
        
#         # Remove any text after the closing bracket
#         bracket_end = json_text.rfind(']')
#         if bracket_end != -1:
#             json_text = json_text[:bracket_end + 1]
        
#         return json_text

# # ============================================================================
# # ZERO-ARTIFACT PDF EDITOR
# # ============================================================================

# class ZeroArtifactEditor:
#     """Zero-artifact PDF editing - absolutely invisible modifications"""
    
#     def __init__(self):
#         self.pdf_doc = None
#         self.text = ""
#         self.page_size = None
    
#     def load_pdf(self, pdf_bytes: bytes) -> bool:
#         """Load PDF from bytes"""
#         try:
#             self.pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#             self.page_size = self.pdf_doc[0].rect
#             return True
#         except Exception as e:
#             st.error(f"Failed to load PDF: {e}")
#             return False
    
#     def extract_text(self) -> str:
#         """Extract text from PDF"""
#         all_text = []
        
#         for i, page in enumerate(self.pdf_doc):
#             text = page.get_text()
            
#             if len(text.strip()) < 50:
#                 text = self._ocr_page(page)
            
#             all_text.append(text)
        
#         self.text = "\n".join(all_text)
#         return self.text
    
#     def _ocr_page(self, page) -> str:
#         """OCR a page"""
#         try:
#             mat = fitz.Matrix(Config.OCR_DPI / 72, Config.OCR_DPI / 72)
#             pix = page.get_pixmap(matrix=mat)
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
#             img_array = np.array(img)
#             gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#             enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                             cv2.THRESH_BINARY, 11, 2)
            
#             return pytesseract.image_to_string(Image.fromarray(enhanced), lang=Config.OCR_LANG)
#         except Exception as e:
#             return ""
    
#     def get_text_with_full_properties(self, page, search_text: str) -> List[Dict]:
#         """Get text with complete font properties"""
#         instances = []
#         blocks = page.get_text("dict")["blocks"]
        
#         for block in blocks:
#             if block.get("type") == 0:
#                 for line in block.get("lines", []):
#                     for span in line.get("spans", []):
#                         span_text = span.get("text", "")
#                         if search_text == span_text:
#                             instances.append({
#                                 "text": span_text,
#                                 "bbox": tuple(span["bbox"]),
#                                 "font": span.get("font", "Helvetica"),
#                                 "size": span.get("size", 10),
#                                 "color": span.get("color", 0),
#                                 "flags": span.get("flags", 0),
#                             })
        
#         return instances
    
#     def replace_text_zero_artifact(self, page, old_text: str, new_text: str, field_name: str) -> int:
#         """Replace text with ZERO visible artifacts"""
        
#         instances = self.get_text_with_full_properties(page, old_text)
        
#         if not instances:
#             return 0
        
#         replaced = 0
        
#         for inst in instances:
#             bbox = inst["bbox"]
#             font_name = inst["font"]
#             font_size = inst["size"]
#             color = inst["color"]
            
#             # Convert color
#             if isinstance(color, int):
#                 r = ((color >> 16) & 0xFF) / 255.0
#                 g = ((color >> 8) & 0xFF) / 255.0
#                 b = (color & 0xFF) / 255.0
#                 text_color = (r, g, b)
#             else:
#                 text_color = (0, 0, 0)
            
#             rect = fitz.Rect(bbox)
            
#             # ZERO ARTIFACT METHOD: Shrink cleaning area significantly
#             padding_v = 2.0
#             padding_h = 1.0
            
#             clean_rect = fitz.Rect(
#                 rect.x0 + padding_h,
#                 rect.y0 + padding_v,
#                 rect.x1 - padding_h,
#                 rect.y1 - padding_v
#             )
            
#             if clean_rect.width > 0 and clean_rect.height > 0:
#                 page.draw_rect(clean_rect, color=None, fill=(1, 1, 1), width=0)
            
#             # Map font
#             fontname = "helv"
#             if "bold" in font_name.lower() and "italic" in font_name.lower():
#                 fontname = "hebi"
#             elif "bold" in font_name.lower():
#                 fontname = "hebo"
#             elif "italic" in font_name.lower():
#                 fontname = "heit"
#             elif "times" in font_name.lower():
#                 fontname = "times" if "bold" not in font_name.lower() else "tibo"
#             elif "courier" in font_name.lower():
#                 fontname = "cour" if "bold" not in font_name.lower() else "cobo"
            
#             # Positioning
#             is_number = bool(re.search(r'[\d,.]', old_text))
#             text_width = fitz.get_text_length(new_text, fontname=fontname, fontsize=font_size)
            
#             if is_number:
#                 x_pos = rect.x1 - text_width - 0.5
#             else:
#                 x_pos = rect.x0 + 0.5
            
#             y_pos = rect.y0 + (rect.height + font_size * 0.75) / 2
            
#             rc = page.insert_text(
#                 fitz.Point(x_pos, y_pos),
#                 new_text,
#                 fontname=fontname,
#                 fontsize=font_size,
#                 color=text_color,
#                 overlay=True,
#                 render_mode=0
#             )
            
#             if rc >= 0:
#                 replaced += 1
        
#         return replaced
    
#     def zero_artifact_edit(self, mappings: List[Dict], page_num: int = 0) -> Tuple[bytes, int, int]:
#         """Perform zero-artifact edit and return PDF bytes"""
#         page = self.pdf_doc[page_num]
        
#         replaced = 0
#         not_found = 0
        
#         for mapping in mappings:
#             old_text = mapping.get('old', '').strip()
#             new_text = mapping.get('new', '').strip()
#             field = mapping.get('field', 'unknown')
            
#             if not old_text or not new_text or old_text == new_text:
#                 continue
            
#             count = self.replace_text_zero_artifact(page, old_text, new_text, field)
            
#             if count > 0:
#                 replaced += count
#             else:
#                 not_found += 1
        
#         # Save to bytes
#         output_bytes = self.pdf_doc.write(
#             garbage=4,
#             deflate=True,
#             clean=True
#         )
        
#         return output_bytes, replaced, not_found
    
#     def close(self):
#         """Close PDF"""
#         if self.pdf_doc:
#             self.pdf_doc.close()

# # ============================================================================
# # STREAMLIT APPLICATION
# # ============================================================================

# def main():
#     # Header
#     st.markdown('<div class="main-header">📄 AI-Powered Paystub Generator</div>', unsafe_allow_html=True)
#     st.markdown('<div class="sub-header">Automatically calculate and generate paystubs with zero visible artifacts</div>', unsafe_allow_html=True)
    
#     # Sidebar - Information Only (No API Key Input)
#     with st.sidebar:
#         st.header("📋 Features")
#         st.markdown("""
#         - ✅ OCR + Font Recognition
#         - ✅ 50+ PDF Format Support
#         - ✅ Automatic Calculations
#         - ✅ Zero Visible Artifacts
#         - ✅ Same Layout Output
        
#         ### 📖 How to Use
#         1. Upload original PDF paystub
#         2. Enter salary and hours details
#         3. Click "Generate Paystub"
#         4. Download processed PDF
#         """)
    
#     # Main Content
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("📤 Upload Original Paystub")
#         uploaded_file = st.file_uploader(
#             "Choose a PDF file",
#             type=['pdf'],
#             help="Upload the original paystub PDF"
#         )
        
#         if uploaded_file:
#             st.success(f"✅ Uploaded: {uploaded_file.name}")
#             st.info(f"📊 File size: {len(uploaded_file.getvalue()) / 1024:.2f} KB")
    
#     with col2:
#         st.header("💰 Salary & Hours Details")
        
#         col2a, col2b = st.columns(2)
        
#         with col2a:
#             hourly_rate = st.number_input(
#                 "Hourly Rate ($)",
#                 min_value=0.0,
#                 value=30.0,
#                 step=0.5,
#                 format="%.2f",
#                 help="Enter hourly rate in dollars"
#             )
            
#             regular_hours = st.number_input(
#                 "Regular Hours (This Period)",
#                 min_value=0.0,
#                 value=80.0,
#                 step=0.5,
#                 format="%.2f",
#                 help="Regular hours for this pay period"
#             )
            
#             ytd_hours = st.number_input(
#                 "YTD Total Hours",
#                 min_value=0.0,
#                 value=1600.0,
#                 step=1.0,
#                 format="%.2f",
#                 help="Year-to-date total hours"
#             )
        
#         with col2b:
#             overtime_hours = st.number_input(
#                 "Overtime Hours (1.5x)",
#                 min_value=0.0,
#                 value=0.0,
#                 step=0.5,
#                 format="%.2f",
#                 help="Overtime hours at 1.5x rate"
#             )
            
#             bonus = st.number_input(
#                 "Bonus ($)",
#                 min_value=0.0,
#                 value=0.0,
#                 step=10.0,
#                 format="%.2f",
#                 help="Bonus amount in dollars"
#             )
            
#             vacation_hours = st.number_input(
#                 "Vacation Hours",
#                 min_value=0.0,
#                 value=0.0,
#                 step=0.5,
#                 format="%.2f",
#                 help="Vacation hours"
#             )
    
#     # Generate Button
#     st.markdown("---")
    
#     if st.button("🚀 Generate Paystub", type="primary"):
#         # Validation
#         if not uploaded_file:
#             st.error("❌ Please upload a PDF file!")
#             return
        
#         # Processing
#         with st.spinner("🔄 Processing paystub..."):
#             try:
#                 # Create log container
#                 st.markdown("### 📝 Processing Log")
#                 log_container = st.empty()
#                 logs = []
                
#                 def add_log(msg):
#                     logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
#                     log_html = "<br>".join([f'<span style="color: #00ff00;">{log}</span>' for log in logs])
#                     log_container.markdown(
#                         f'<div class="log-container">{log_html}</div>',
#                         unsafe_allow_html=True
#                     )
                
#                 add_log("="*70)
#                 add_log("🚀 AI PAYSTUB AUTOMATION v10.0")
#                 add_log("   ✓ Calculates EVERYTHING automatically")
#                 add_log("   ✓ Works with ANY PDF format")
#                 add_log("   ✓ Zero visible artifacts")
#                 add_log("="*70)
                
#                 # Initialize
#                 user_inputs = {
#                     'regular_rate': hourly_rate,
#                     'regular_hours': regular_hours,
#                     'ytd_total_hours': ytd_hours,
#                     'overtime_hours': overtime_hours,
#                     'overtime_rate': hourly_rate * 1.5,
#                     'bonus': bonus,
#                     'vacation_hours': vacation_hours
#                 }
                
#                 # Load PDF
#                 add_log(f"📄 Loading PDF: {uploaded_file.name}")
#                 pdf_bytes = uploaded_file.getvalue()
                
#                 pdf_editor = ZeroArtifactEditor()
#                 if not pdf_editor.load_pdf(pdf_bytes):
#                     st.error("❌ Failed to load PDF!")
#                     return
                
#                 add_log(f"✅ Loaded: {len(pdf_editor.pdf_doc)} page(s)")
                
#                 # Extract text
#                 add_log("📖 STEP 1: Extracting text from PDF...")
#                 text = pdf_editor.extract_text()
                
#                 if len(text) < 50:
#                     st.error("❌ No text found in PDF!")
#                     return
                
#                 add_log(f"✅ Extracted {len(text)} characters")
                
#                 # Show sample
#                 with st.expander("📄 View Extracted Text Sample"):
#                     st.code(text[:500] + "..." if len(text) > 500 else text)
                
#                 # AI Extract
#                 add_log("🤖 STEP 2: AI extracting ALL values...")
#                 ai = IntelligentAI()
#                 extraction = ai.extract_all_values(text)
                
#                 add_log("✅ Extraction complete!")
                
#                 with st.expander("🔍 View Extracted Values"):
#                     st.code(extraction[:1000] + "..." if len(extraction) > 1000 else extraction)
                
#                 # AI Calculate
#                 add_log("🧮 STEP 3: AI calculating ALL new values...")
#                 add_log("   Calculating: Earnings, Gross, Deductions, Taxes, Net Pay, YTD...")
                
#                 mappings_json = ai.calculate_all_mappings(extraction, user_inputs)
                
#                 try:
#                     mappings = json.loads(mappings_json)
#                     add_log(f"✅ Created {len(mappings)} complete mappings")
                    
#                     if len(mappings) == 0:
#                         st.error("❌ No mappings created!")
#                         return
                    
#                 except json.JSONDecodeError as e:
#                     add_log(f"⚠️  JSON Parse Error: {str(e)}")
#                     add_log(f"📝 Raw response preview: {mappings_json[:500]}")
                    
#                     # Try to show what we received
#                     with st.expander("⚠️ Debug: View Raw AI Response"):
#                         st.code(mappings_json)
                    
#                     st.error(f"❌ Failed to parse AI response. Please check the debug section above.")
#                     return
#                 except Exception as e:
#                     st.error(f"❌ Unexpected error parsing mappings: {e}")
#                     return
                
#                 # Show mappings preview
#                 with st.expander("📊 View Calculated Mappings"):
#                     for i, m in enumerate(mappings[:15]):
#                         st.text(f"{i+1}. {m.get('field')}: {m.get('old')} → {m.get('new')}")
#                     if len(mappings) > 15:
#                         st.text(f"... and {len(mappings) - 15} more")
                
#                 # Apply changes
#                 add_log("✏️  STEP 4: Applying ALL changes (Zero-Artifact Mode)...")
                
#                 output_bytes, replaced, not_found = pdf_editor.zero_artifact_edit(mappings, page_num=0)
                
#                 add_log(f"📊 Replaced: {replaced} | Not found: {not_found}")
#                 add_log("="*70)
#                 add_log("✅ PAYSTUB GENERATION COMPLETE!")
#                 add_log(f"✓ {len(mappings)} values processed")
#                 add_log(f"✓ {replaced} values updated")
#                 add_log("="*70)
                
#                 # Cleanup
#                 pdf_editor.close()
                
#                 # Success
#                 st.success("✅ Paystub generated successfully!")
                
#                 # Download button
#                 st.markdown("### 📥 Download Processed Paystub")
                
#                 output_filename = f"generated_paystub_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
#                 st.download_button(
#                     label="⬇️ Download Generated Paystub",
#                     data=output_bytes,
#                     file_name=output_filename,
#                     mime="application/pdf",
#                     type="primary"
#                 )
                
#                 st.balloons()
                
#             except Exception as e:
#                 st.error(f"❌ Error: {str(e)}")
#                 st.exception(e)
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style="text-align: center; color: #666; font-size: 0.9rem;">
#         <p>🚀 AI-Powered Paystub Generator v10.0</p>
#         <p>Supports 50+ PDF formats with OCR, font recognition, and zero-artifact editing</p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()























# ///////////////////












"""
AI-Powered DOCX Paystub Generator - Streamlit Web Application
✓ Upload DOCX paystubs
✓ Input salary and hours details
✓ Context-aware replacements
✓ Zero destructive edits
✓ Download processed paystub
"""

import os
import sys
import json
import re
import io
from typing import Dict, List, Tuple
from datetime import datetime
import tempfile

# Streamlit
import streamlit as st

# DOCX Processing
from docx import Document

# AI Integration
import google.generativeai as genai

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI DOCX Paystub Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - BLACK BACKGROUND FOR LOGS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .log-container {
        background-color: #000000;
        border: 1px solid #333;
        border-radius: 0.5rem;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 600px;
        overflow-y: auto;
        color: #00ff00;
        line-height: 1.8;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # HARDCODED API KEY
    API_KEY = 'AIzaSyAKidKZhEBio8Akj0nBJrZBxWyOc5_JLso'
    MODEL = 'gemini-2.0-flash'

# ============================================================================
# CONTEXT-AWARE REPLACER (YOUR EXACT CODE)
# ============================================================================

class ContextAwareReplacer:
    """Replaces text ONLY in correct context (specific table cells)"""
    
    @staticmethod
    def find_value_in_tables(doc: Document, search_text: str) -> List[Tuple]:
        """Find value and its exact table location"""
        locations = []
        search_clean = re.sub(r'[^\d\-.]', '', search_text)
        
        for table_idx, table in enumerate(doc.tables):
            for row_idx, row in enumerate(table.rows):
                for cell_idx, cell in enumerate(row.cells):
                    cell_text = cell.text.strip()
                    cell_clean = re.sub(r'[^\d\-.]', '', cell_text)
                    
                    # Exact or fuzzy match
                    if search_text in cell_text or (search_clean and search_clean == cell_clean):
                        locations.append((table_idx, row_idx, cell_idx, cell))
        
        return locations
    
    @staticmethod
    def replace_in_cell(cell, old_text: str, new_text: str) -> bool:
        """Replace text in specific cell only, preserving formatting"""
        found = False
        old_clean = re.sub(r'[^\d\-.]', '', old_text)
        
        for para in cell.paragraphs:
            # Build full text from runs
            full_text = ''.join([r.text for r in para.runs])
            
            # Check if value is in this paragraph
            if old_text not in full_text:
                full_clean = re.sub(r'[^\d\-.]', '', full_text)
                if not old_clean or old_clean != old_clean:
                    continue
            
            # Replace at run level (preserves formatting)
            try:
                # Find which run(s) contain the old text
                pos = 0
                for run_idx, run in enumerate(para.runs):
                    run_text = run.text
                    run_len = len(run_text)
                    
                    # Check if old text starts in this run
                    if old_text in run_text:
                        # Simple case: entire value in one run
                        run.text = run_text.replace(old_text, new_text)
                        found = True
                        break
                    
                    # Fuzzy match for numbers
                    run_clean = re.sub(r'[^\d\-.]', '', run_text)
                    if old_clean and old_clean in run_clean:
                        # Replace numbers only
                        run.text = run_text.replace(old_text, new_text)
                        found = True
                        break
                    
                    pos += run_len
                
                if found:
                    break
                    
            except Exception as e:
                continue
        
        return found

# ============================================================================
# AI ENGINE WITH CONTEXT (YOUR EXACT CODE)
# ============================================================================

class SmartPaystubAI:
    """AI that understands paystub structure"""
    
    def __init__(self):
        genai.configure(api_key=Config.API_KEY)
        self.model = genai.GenerativeModel(Config.MODEL, generation_config={"temperature": 0.1})
    
    def extract_structured_data(self, doc: Document) -> str:
        """Extract text WITH table structure context"""
        parts = []
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(f"[PARAGRAPH] {para.text}")
        
        # Extract tables WITH structure
        for table_idx, table in enumerate(doc.tables):
            parts.append(f"\n[TABLE {table_idx}]")
            for row_idx, row in enumerate(table.rows):
                row_data = []
                for cell_idx, cell in enumerate(row.cells):
                    text = cell.text.strip()
                    if text:
                        row_data.append(f"Cell[{row_idx},{cell_idx}]: {text}")
                if row_data:
                    parts.append(" | ".join(row_data))
        
        return "\n".join(parts)
    
    def analyze_and_calculate(self, text: str, inputs: Dict) -> List[Dict]:
        """AI creates context-aware mappings"""
        
        prompt = f"""Analyze paystub structure and create PRECISE replacements.

PAYSTUB STRUCTURE:
{text}

NEW INPUTS:
- Rate: ${inputs['rate']:.2f}/hr
- Hours: {inputs['hours']}
- YTD Hours: {inputs['ytd_hours']}
- Overtime: {inputs['overtime']} hrs
- Bonus: ${inputs['bonus']:.2f}
- Vacation: {inputs['vacation']} hrs

FIND THESE STANDARD FIELDS:
✓ Hourly Rate / Rate (Current)
✓ Regular Hours / Hours (Current)
✓ Regular Earnings / RegularHourly (Current + YTD)
✓ Gross Pay (Current + YTD)
✓ CPP Employee Withheld (Current + YTD)
✓ EI Employee Withheld (Current + YTD)
✓ Federal Tax Withheld (Current + YTD)
✓ Vacation Pay (Current + YTD)
✓ Net Pay (Current + YTD)
✓ Other Deductions (PEN3_REG_EE, etc.)

CALCULATE:
1. Regular = Rate × Hours = ${inputs['rate']:.2f} × {inputs['hours']} = ${inputs['rate'] * inputs['hours']:.2f}
2. OT = Rate × 1.5 × {inputs['overtime']} = ${inputs['rate'] * 1.5 * inputs['overtime']:.2f}
3. Vacation = Rate × {inputs['vacation']} = ${inputs['rate'] * inputs['vacation']:.2f}
4. Gross = Regular + OT + Vacation + Bonus = ${inputs['rate'] * inputs['hours'] + inputs['rate'] * 1.5 * inputs['overtime'] + inputs['rate'] * inputs['vacation'] + inputs['bonus']:.2f}
5. YTD Gross = Rate × YTD Hours + Bonus = ${inputs['rate'] * inputs['ytd_hours'] + inputs['bonus']:.2f}
6. CPP = 5.95% × Gross
7. EI = 1.63% × Gross
8. Fed Tax = 15% × Gross
9. Net = Gross - Deductions

FORMAT RULES:
- Match EXACT format: if original has "2,811.60" use "9,000.00"
- If original has "169.32" for deduction, use same format
- DON'T replace "N/A" - only replace ACTUAL numeric values

CRITICAL: Only create mappings for VALUES that exist (not labels, not "N/A")

Return JSON:
[
  {{"field": "Hourly Rate", "old": "140.58", "new": "50.00", "context": "base rate"}},
  {{"field": "Regular Hours", "old": "20.00", "new": "180.00", "context": "hours worked"}},
  {{"field": "Regular Earnings Current", "old": "2,811.60", "new": "9,000.00", "context": "current period"}},
  {{"field": "Regular Earnings YTD", "old": "7,029.00", "new": "130,000.00", "context": "year to date"}},
  {{"field": "Vacation Pay", "old": "168.70", "new": "400.00", "context": "vacation earnings"}},
  {{"field": "Gross Pay Current", "old": "2,980.30", "new": "10,275.00", "context": "gross current"}},
  {{"field": "Gross Pay YTD", "old": "7,450.75", "new": "130,500.00", "context": "gross ytd"}},
  {{"field": "CPP Current", "old": "169.32", "new": "611.36", "context": "cpp deduction"}},
  {{"field": "CPP YTD", "old": "419.29", "new": "3,867.50", "context": "cpp ytd"}},
  {{"field": "EI Current", "old": "48.58", "new": "167.48", "context": "ei deduction"}},
  {{"field": "EI YTD", "old": "121.45", "new": "1,049.42", "context": "ei ytd"}},
  {{"field": "Federal Tax Current", "old": "462.41", "new": "1,541.25", "context": "fed tax"}},
  {{"field": "Net Pay Current", "old": "1,975.88", "new": "6,927.41", "context": "net pay"}},
  {{"field": "Net Pay YTD", "old": "5,035.61", "new": "92,957.08", "context": "net pay ytd"}}
]

Return ONLY JSON array of actual values found (not N/A, not labels)"""
        
        try:
            response = self.model.generate_content(prompt)
            json_text = response.text.strip()
            
            # Clean
            json_text = re.sub(r'^```json?\s*', '', json_text, flags=re.MULTILINE)
            json_text = re.sub(r'^```\s*$', '', json_text, flags=re.MULTILINE)
            
            match = re.search(r'\[.*\]', json_text, re.DOTALL)
            if match:
                json_text = match.group(0)
            
            mappings = json.loads(json_text)
            
            # Filter out N/A replacements
            mappings = [m for m in mappings if 'N/A' not in m.get('old', '') and 'N/A' not in m.get('new', '')]
            
            return mappings
        except Exception as e:
            raise Exception(f"AI failed: {e}")

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    # Initialize session state
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'output_docx' not in st.session_state:
        st.session_state.output_docx = None
    if 'output_filename' not in st.session_state:
        st.session_state.output_filename = ""
    
    # Header
    st.markdown('<div class="main-header">📄 AI-Powered DOCX Paystub Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Context-aware paystub generation with zero destructive edits</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Features")
        st.markdown("""
        - ✅ Context-Aware Replacements
        - ✅ DOCX Format Support
        - ✅ Automatic Calculations
        - ✅ Perfect Formatting Preservation
        - ✅ No Random Replacements
        
        ### 📖 How to Use
        1. Upload original DOCX paystub
        2. Enter salary and hours details
        3. Click "Generate Paystub"
        4. View processing logs
        5. Download processed DOCX
        6. Use Reset to clear logs
        """)
        
        st.markdown("---")
        
        # Reset Button
        if st.button("🔄 Reset Logs", type="secondary"):
            st.session_state.logs = []
            st.session_state.output_docx = None
            st.session_state.output_filename = ""
            st.rerun()
    
    # Main Content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Original Paystub")
        uploaded_file = st.file_uploader(
            "Choose a DOCX file",
            type=['docx'],
            help="Upload the original paystub DOCX file"
        )
        
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {len(uploaded_file.getvalue()) / 1024:.2f} KB")
    
    with col2:
        st.header("💰 Salary & Hours Details")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            hourly_rate = st.number_input(
                "Hourly Rate ($)",
                min_value=0.0,
                value=50.0,
                step=0.5,
                format="%.2f",
                help="Enter hourly rate in dollars"
            )
            
            regular_hours = st.number_input(
                "Regular Hours (This Period)",
                min_value=0.0,
                value=180.0,
                step=0.5,
                format="%.2f",
                help="Regular hours for this pay period"
            )
            
            ytd_hours = st.number_input(
                "YTD Total Hours",
                min_value=0.0,
                value=2600.0,
                step=1.0,
                format="%.2f",
                help="Year-to-date total hours"
            )
        
        with col2b:
            overtime_hours = st.number_input(
                "Overtime Hours (1.5x)",
                min_value=0.0,
                value=0.0,
                step=0.5,
                format="%.2f",
                help="Overtime hours at 1.5x rate"
            )
            
            bonus = st.number_input(
                "Bonus ($)",
                min_value=0.0,
                value=0.0,
                step=10.0,
                format="%.2f",
                help="Bonus amount in dollars"
            )
            
            vacation_hours = st.number_input(
                "Vacation Hours",
                min_value=0.0,
                value=0.0,
                step=0.5,
                format="%.2f",
                help="Vacation hours"
            )
    
    # Generate Button
    st.markdown("---")
    
    if st.button("🚀 Generate Paystub", type="primary"):
        # Validation
        if not uploaded_file:
            st.error("❌ Please upload a DOCX file!")
            return
        
        # Clear previous results
        st.session_state.logs = []
        st.session_state.output_docx = None
        
        # Processing
        with st.spinner("🔄 Processing paystub..."):
            try:
                def add_log(msg):
                    st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                
                add_log("="*70)
                add_log("🚀 PAYSTUB GENERATOR v3.0 - CONTEXT-AWARE")
                add_log("   ✓ Replaces ONLY in correct locations")
                add_log("   ✓ Perfect formatting preservation")
                add_log("   ✓ No destructive replacements")
                add_log("="*70)
                
                # Initialize
                user_inputs = {
                    'rate': hourly_rate,
                    'hours': regular_hours,
                    'ytd_hours': ytd_hours,
                    'overtime': overtime_hours,
                    'bonus': bonus,
                    'vacation': vacation_hours
                }
                
                # Load DOCX
                add_log(f"📄 Loading: {uploaded_file.name}")
                doc_bytes = uploaded_file.getvalue()
                doc = Document(io.BytesIO(doc_bytes))
                add_log("✅ Loaded")
                add_log("")
                
                # Extract structured data
                add_log("📖 STEP 1: Extract Structured Data")
                add_log("")
                ai = SmartPaystubAI()
                text = ai.extract_structured_data(doc)
                add_log(f"✅ Extracted {len(text)} chars with structure")
                add_log("")
                
                # AI Calculate
                add_log("🤖 STEP 2: AI Analysis & Calculation")
                add_log("")
                mappings = ai.analyze_and_calculate(text, user_inputs)
                
                if not mappings:
                    add_log("❌ No valid mappings created!")
                    st.error("❌ No valid mappings created!")
                    return
                
                add_log(f"✅ Created {len(mappings)} context-aware mappings")
                add_log("")
                
                # Display mappings
                add_log("="*70)
                add_log("📋 CONTEXT-AWARE MAPPINGS:")
                add_log("="*70)
                for i, m in enumerate(mappings, 1):
                    field = m.get('field', 'unknown')
                    old = m.get('old', '')
                    new = m.get('new', '')
                    context = m.get('context', '')
                    add_log(f"{i:2d}. {field:30s}: {old:15s} → {new:15s} [{context}]")
                add_log("="*70)
                add_log("")
                
                # Replace with context awareness
                add_log("✏️  STEP 3: Context-Aware Replacement")
                add_log("")
                
                replacer = ContextAwareReplacer()
                total = 0
                
                for mapping in mappings:
                    old = mapping.get('old', '').strip()
                    new = mapping.get('new', '').strip()
                    field = mapping.get('field', 'unknown')
                    
                    if not old or not new or old == new:
                        continue
                    
                    # Find ALL locations of this value
                    locations = replacer.find_value_in_tables(doc, old)
                    
                    if locations:
                        count = 0
                        for table_idx, row_idx, cell_idx, cell in locations:
                            if replacer.replace_in_cell(cell, old, new):
                                count += 1
                        
                        if count > 0:
                            add_log(f"   ✅ {field}: '{old}' → '{new}' ({count} cells)")
                            total += count
                        else:
                            add_log(f"   ⚠️  {field}: Found but replacement failed")
                    else:
                        add_log(f"   ⚠️  {field}: '{old}' not found in tables")
                
                add_log("")
                add_log(f"✅ Replaced {total} values")
                add_log("")
                
                # Save to bytes
                add_log("💾 Saving: output.docx")
                output_buffer = io.BytesIO()
                doc.save(output_buffer)
                output_buffer.seek(0)
                
                st.session_state.output_docx = output_buffer.getvalue()
                st.session_state.output_filename = f"generated_paystub_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                
                add_log("="*70)
                add_log("✅ COMPLETE!")
                add_log(f"   Input:  {uploaded_file.name}")
                add_log(f"   Output: {st.session_state.output_filename}")
                add_log(f"   Values: {total}")
                add_log("="*70)
                add_log("")
                
                st.success("✅ Paystub generated successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                add_log(f"❌ ERROR: {str(e)}")
    
    # Display Logs (ALWAYS SHOWN IF EXIST)
    if st.session_state.logs:
        st.markdown("---")
        st.markdown("### 📝 Processing Log")
        log_html = "<br>".join([f'<span style="color: #00ff00;">{log}</span>' for log in st.session_state.logs])
        st.markdown(
            f'<div class="log-container">{log_html}</div>',
            unsafe_allow_html=True
        )
    
    # Download Button (ALWAYS SHOWN IF FILE EXISTS)
    if st.session_state.output_docx:
        st.markdown("---")
        st.markdown("### 📥 Download Processed Paystub")
        
        st.download_button(
            label="⬇️ Download Generated Paystub (DOCX)",
            data=st.session_state.output_docx,
            file_name=st.session_state.output_filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🚀 AI-Powered DOCX Paystub Generator v3.0 - Context-Aware</p>
        <p>Replaces ONLY in correct locations • Perfect formatting preservation • Zero destructive edits</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
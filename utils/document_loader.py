

import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
import io
import re
from typing import Union

def load_document(file) -> Union[str, pd.DataFrame]:
    """Enhanced text extraction with better PDF handling and table detection"""
    try:
        if file.name.endswith(".pdf"):
            full_text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        full_text += "\nTABLE:\n" + df.to_string() + "\n"
            
            return full_text if full_text else "Could not extract text from PDF"
        
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
            return df.to_string()
        
        elif file.name.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(io.BytesIO(file.read()))
            text = pytesseract.image_to_string(img)
            return text if text else "Could not extract text from image"
        
        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            return df.to_string()
        
        else:  # TXT and other text formats
            return file.getvalue().decode("utf-8")
    
    except Exception as e:
        raise ValueError(f"Error processing document: {str(e)}")

def extract_financial_sections(text: str) -> dict:
    """Extract key financial sections from document text"""
    sections = {
        "income_statement": "",
        "balance_sheet": "",
        "cash_flow": "",
        "management_discussion": "",
        "risks": ""
    }
    
    # Try to find section headers (case insensitive)
    patterns = {
        "income_statement": r"(consolidated statements? of income|income statements?|statements? of operations)",
        "balance_sheet": r"(consolidated balance sheets?|balance sheets?)",
        "cash_flow": r"(consolidated statements? of cash flows?|cash flows?)",
        "management_discussion": r"(management'?s discussion and analysis|MD&A)",
        "risks": r"(risk factors|risks and uncertainties)"
    }
    
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            next_section = min(
                [m.start() for m in re.finditer("|".join(p for p in patterns.values() if p != pattern), text[start_pos:], re.IGNORECASE)] or [len(text)])
            sections[section] = text[start_pos:start_pos+next_section].strip()
    
    return sections
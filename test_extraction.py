import pandas as pd
import re
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

def extract_text_from_page_ocr(page, dpi=300):
    """Extract text from a PDF page using OCR"""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes('png')
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return ""

def extract_name_from_pdf(pdf_path):
    """Extract name from second line of first page"""
    try:
        doc = fitz.open(pdf_path)
        first_page = doc[0]
        text = extract_text_from_page_ocr(first_page)
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        for i, line in enumerate(lines):
            if 'Candidate' in line and i + 1 < len(lines):
                name_line = lines[i + 1]
                name = re.split(r'\s*[\(@]', name_line)[0].strip()
                doc.close()
                return name

        doc.close()
    except Exception as e:
        print(f"Error extracting name: {str(e)}")
        return ""

    return ""

def parse_star_rating(text_snippet):
    """Parse star rating from OCR text"""
    star_count = 0

    # Count asterisks (each * is one star)
    star_count += text_snippet.count('*')

    # Count unicode stars
    star_count += text_snippet.count('★')
    star_count += text_snippet.count('⭐')

    if star_count > 0:
        return star_count

    # Check for OCR patterns if no direct stars found
    star_patterns = [
        (r'kkk', 3),  # Three stars as "KKK"
        (r'kk', 2),   # Two stars as "KK"
        (r'>\s*oo\.\s*[0-9]', 3),  # Pattern like "> oo. 4"
        (r'>\s*oo', 3),  # Pattern like "> oo"
    ]

    text_lower = text_snippet.lower()
    for pattern, score in star_patterns:
        if re.search(pattern, text_lower):
            return score

    return 0

def parse_skills_from_text(text):
    """Parse skills and scores from extracted OCR text"""
    skills = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    in_skills_section = False
    i = 0

    while i < len(lines):
        line = lines[i]

        if 'Skills Feedback' in line or 'Feedback' in line:
            in_skills_section = True
            i += 1
            continue

        if 'AI Assessment' in line or 'Summary of Questions' in line or 'Overall Feedback' in line:
            break

        if in_skills_section:
            skill_match = re.match(r'^([A-Za-z\s/]+?)\s+([\*Kk>oo\.\s0-9]+)$', line)

            if skill_match:
                skill_name = skill_match.group(1).strip()
                rating_text = skill_match.group(2).strip()
                score = parse_star_rating(rating_text)

                if skill_name and score > 0:
                    skill_name = skill_name.replace('  ', ' ').strip()

                    # Filter out invalid skill names (containing x, numbers, etc.)
                    if not re.search(r'[x0-9]', skill_name.lower()):
                        skills.append({'skill': skill_name, 'score': score})

        i += 1

    return skills

def extract_skills_and_scores(pdf_path):
    """Extract assessment areas and scores from page 3 onwards"""
    skills_data = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_num in range(2, total_pages):
            page = doc[page_num]
            text = extract_text_from_page_ocr(page)
            print(f"\n=== PAGE {page_num + 1} OCR ===")
            print(text[:500])
            print("\n=== Parsed skills ===")
            page_skills = parse_skills_from_text(text)
            for skill in page_skills:
                print(f"{skill['skill']}: {skill['score']}")
            skills_data.extend(page_skills)

        doc.close()
    except Exception as e:
        print(f"Error extracting skills: {str(e)}")

    return skills_data

# Test extraction
pdf_path = 'Mukesh Kumar S_MonjinInterviewDoc.pdf'

print("Extracting name...")
name = extract_name_from_pdf(pdf_path)
print(f"Name: {name}\n")

print("Extracting skills and scores...")
skills = extract_skills_and_scores(pdf_path)

print(f"\n=== FINAL RESULTS ===")
print(f"Total skills found: {len(skills)}")

# Create DataFrame
rows = []
for skill in skills:
    rows.append({
        'Name': name,
        'Interview Level': 'L1',
        'Assessment Area': skill['skill'],
        'Score': skill['score']
    })

df = pd.DataFrame(rows)
print("\n", df.to_string())

# Save to Excel
df.to_excel('test_output.xlsx', index=False)
print("\nSaved to test_output.xlsx")

import streamlit as st
import pandas as pd
import re
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

def extract_text_from_page_ocr(page, dpi=300):
    """Extract text from a PDF page using OCR"""
    try:
        # Convert page to image
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes('png')
        img = Image.open(io.BytesIO(img_data))

        # Perform OCR
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"OCR error: {str(e)}")
        return ""

def extract_name_from_pdf(pdf_bytes):
    """Extract name from second line of first page"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        first_page = doc[0]

        # Extract text using OCR
        text = extract_text_from_page_ocr(first_page)

        # Parse the name from text
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        for i, line in enumerate(lines):
            if 'Candidate' in line and i + 1 < len(lines):
                name_line = lines[i + 1]
                # Extract just the name part (before email or special chars)
                name = re.split(r'\s*[\(@]', name_line)[0].strip()
                return name

        doc.close()
    except Exception as e:
        st.error(f"Error extracting name: {str(e)}")
        return ""

    return ""

def parse_star_rating(text_snippet):
    """Parse star rating from OCR text - handles various OCR interpretations of stars"""
    # Stars can be recognized as: *, **, ***, Kk, > oo, etc.
    # Count individual star characters
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

def extract_skills_and_scores(pdf_bytes):
    """Extract assessment areas and scores from page 3 onwards"""
    skills_data = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        # Process pages 3 onwards (index 2+)
        for page_num in range(2, total_pages):
            page = doc[page_num]
            text = extract_text_from_page_ocr(page)

            # Extract skills from text
            skills_data.extend(parse_skills_from_text(text))

        doc.close()
    except Exception as e:
        st.error(f"Error extracting skills: {str(e)}")

    return skills_data

def parse_skills_from_text(text):
    """Parse skills and scores from extracted OCR text"""
    skills = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    in_skills_section = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if we're in a skills feedback section
        if 'Skills Feedback' in line or 'Feedback' in line:
            in_skills_section = True
            i += 1
            continue

        # Stop at certain sections
        if 'AI Assessment' in line or 'Summary of Questions' in line or 'Overall Feedback' in line:
            break

        if in_skills_section:
            # Look for skill patterns: "Skill Name * *" or "Skill Name Kk" etc.
            # Common patterns from OCR:
            # - "Cucumber * *"
            # - "Core Java Kk"
            # - "Automation Framework > oo. 4"

            # Match lines that have skill names followed by star-like patterns
            skill_match = re.match(r'^([A-Za-z\s/]+?)\s+([\*Kk>oo\.\s0-9]+)$', line)

            if skill_match:
                skill_name = skill_match.group(1).strip()
                rating_text = skill_match.group(2).strip()

                # Parse the star rating
                score = parse_star_rating(rating_text)

                if skill_name and score > 0:
                    # Clean up skill name
                    skill_name = skill_name.replace('  ', ' ').strip()

                    # Filter out invalid skill names (containing x, numbers, etc.)
                    if not re.search(r'[x0-9]', skill_name.lower()):
                        skills.append({'skill': skill_name, 'score': score})

        i += 1

    return skills

def create_output_dataframe(name, skills_data):
    """Create output dataframe in the required format"""
    rows = []

    for skill in skills_data:
        rows.append({
            'Name': name,
            'Interview Level': 'L1',
            'Assessment Area': skill['skill'],
            'Score': skill['score']
        })

    return pd.DataFrame(rows)

def main():
    st.title("Monjin - PDF Interview Assessment Extractor")
    st.write("Upload a PDF file to extract interview assessment data")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        # Process file button
        if st.button("Process File"):
            with st.spinner("Processing PDF..."):
                try:
                    # Read PDF bytes
                    pdf_bytes = uploaded_file.read()

                    # Extract name
                    name = extract_name_from_pdf(pdf_bytes)

                    if name:
                        st.write(f"**Extracted Name:** {name}")
                    else:
                        st.warning("Could not extract name automatically. Using 'Unknown' as default.")
                        name = "Unknown"

                    # Extract skills and scores
                    skills_data = extract_skills_and_scores(pdf_bytes)

                    if skills_data:
                        # Create output dataframe
                        df = create_output_dataframe(name, skills_data)

                        # Display output
                        st.subheader("Extracted Assessment Data")
                        st.dataframe(df)

                        # Create download button for Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Assessment')

                        output.seek(0)

                        st.download_button(
                            label="Download Excel Output",
                            data=output,
                            file_name=f"{name.replace(' ', '_')}_assessment.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        st.success("Processing completed successfully!")
                    else:
                        st.warning("No skills with star ratings found in the PDF")

                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()

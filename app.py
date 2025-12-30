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
    # Stars can be recognized as: *, **, ***, Kk, KKK, !, !!, !!!, > oo, etc.

    # First check for OCR patterns (Kk, bo &, etc.) which represent 3 stars
    # This takes priority over counting individual symbols
    text_lower = text_snippet.lower()

    star_patterns = [
        (r'k\s*k\s*k', 3),  # Three K's with possible spaces
        (r'k\s*k', 3),   # Two K's often means 3 stars (Kk from OCR of ★★★)
        (r'bo\s*&', 3),  # "bo &" is OCR misreading of 3 stars
        (r'b\s*o', 3),   # "bo" or "b o" variation
        (r'>\s*oo\.\s*[0-9]', 3),  # Pattern like "> oo. 4"
        (r'>\s*oo', 3),  # Pattern like "> oo"
    ]

    for pattern, score in star_patterns:
        if re.search(pattern, text_lower):
            return score

    # If no OCR patterns found, count individual star characters
    star_count = 0

    # Count asterisks (each * is one star)
    star_count += text_snippet.count('*')

    # Count unicode stars
    star_count += text_snippet.count('★')
    star_count += text_snippet.count('⭐')

    # Count exclamation marks (OCR sometimes reads stars as !)
    star_count += text_snippet.count('!')

    # Check for repeated characters that might be stars
    if star_count == 0:
        repeated_char_patterns = [
            (r'[!|l]{3,}', 3),  # Three or more !, |, or l characters
            (r'[!|l]{2}', 2),  # Two !, |, or l characters
        ]

        for pattern, score in repeated_char_patterns:
            if re.search(pattern, text_lower):
                return score

    return star_count if star_count > 0 else 0

def extract_skills_and_scores(pdf_bytes):
    """Extract assessment areas and scores from page 2 onwards"""
    skills_data = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        # Process pages 2 onwards (index 1 = page 2, index 2 = page 3, etc.)
        for page_num in range(1, total_pages):
            page = doc[page_num]
            text = extract_text_from_page_ocr(page)

            # Extract skills from text
            skills_data.extend(parse_skills_from_text(text))

        doc.close()
    except Exception as e:
        st.error(f"Error extracting skills: {str(e)}")

    return skills_data

def parse_skills_from_text(text):
    """Parse skills and scores from extracted OCR text - only from JD Skills Feedback section"""
    skills = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    in_jd_skills_section = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if we're entering the JD Skills Feedback section
        if 'JD Skills Feedback' in line:
            in_jd_skills_section = True
            i += 1
            continue

        # Stop at other feedback sections or end sections
        if in_jd_skills_section and ('Timeline Skills Feedback' in line or
                                      'AI Assessment' in line or
                                      'Summary of Questions' in line or
                                      'Overall Feedback' in line):
            break

        if in_jd_skills_section:
            # Pattern 1: Skill name and stars on the same line
            # e.g., "Cucumber * *" or "Core Java Kk"
            skill_match = re.match(r'^([A-Za-z\s/\(\),]+?)\s+([\*Kk>oo!\|lb&x\.\s0-9]+)$', line)

            if skill_match:
                skill_name = skill_match.group(1).strip()
                rating_text = skill_match.group(2).strip()

                # Parse the star rating
                score = parse_star_rating(rating_text)

                if skill_name and score > 0:
                    # Clean up skill name
                    skill_name = skill_name.replace('  ', ' ').strip()
                    # Filter out invalid skill names (containing numbers)
                    if not re.search(r'\d', skill_name):
                        skills.append({'skill': skill_name, 'score': score})

            # Pattern 2: Skill name on one line, stars on the next line
            # e.g., "Kafka" followed by "xk" on next line
            elif re.match(r'^[A-Za-z\s/\(\),]+$', line) and i + 1 < len(lines):
                # Check if next line looks like a star rating
                next_line = lines[i + 1].strip()
                if re.match(r'^[\*Kk>oo!\|lb&x\.\s0-9]+$', next_line):
                    skill_name = line.strip()
                    rating_text = next_line

                    # Parse the star rating
                    score = parse_star_rating(rating_text)

                    if skill_name and score > 0:
                        # Filter out invalid skill names
                        if not re.search(r'\d', skill_name):
                            skills.append({'skill': skill_name, 'score': score})
                        i += 1  # Skip the next line since we've processed it

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

    # Debug mode toggle
    debug_mode = st.checkbox("Enable Debug Mode (show OCR output)", value=False)

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

                    # Show debug info if enabled
                    if debug_mode:
                        st.subheader("Debug: OCR Output from Page 2")
                        try:
                            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                            if len(doc) > 1:
                                page = doc[1]
                                text = extract_text_from_page_ocr(page)
                                st.text_area("Page 2 OCR Text", text, height=300)
                            doc.close()
                        except Exception as e:
                            st.error(f"Debug error: {str(e)}")

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

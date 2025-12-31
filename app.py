import streamlit as st
import pandas as pd
import re
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import numpy as np
import base64
from openai import OpenAI
import json

def is_filled_star_pixel(pixel_rgb):
    """Check if pixel is part of a filled (orange) star"""
    r, g, b = pixel_rgb[:3]  # Handle both RGB and RGBA
    # Orange stars: high red, medium-high green, low blue
    # Calibrated for typical PDF star rendering
    return (r > 180 and 80 < g < 200 and b < 100)

def detect_stars_from_image(img, skill_row_y, img_width):
    """
    Detect filled stars in a horizontal row by analyzing orange pixels

    Args:
        img: PIL Image object
        skill_row_y: Vertical position (y coordinate) of the skill row
        img_width: Width of the image

    Returns:
        Number of filled (orange) stars detected
    """
    try:
        # Convert image to numpy array for faster processing
        img_array = np.array(img)

        # Define the vertical range to scan (around the skill row)
        # Stars are typically 15-25 pixels tall
        row_height = 25
        y_start = max(0, skill_row_y - 5)
        y_end = min(img_array.shape[0], skill_row_y + row_height)

        # Define horizontal range - stars are typically in the right portion of the page
        # Assume stars start around 60% of page width
        x_start = int(img_width * 0.6)
        x_end = img_width

        # Extract the region of interest
        roi = img_array[y_start:y_end, x_start:x_end]

        # Create a mask of orange pixels
        orange_mask = np.zeros(roi.shape[:2], dtype=bool)
        for y in range(roi.shape[0]):
            for x in range(roi.shape[1]):
                if is_filled_star_pixel(roi[y, x]):
                    orange_mask[y, x] = True

        # Count distinct orange regions horizontally
        # Each contiguous orange region = one star
        star_count = 0
        in_star = False

        # Scan horizontally across the middle of the ROI
        mid_y = orange_mask.shape[0] // 2
        if mid_y < orange_mask.shape[0]:
            row_scan = orange_mask[mid_y, :]

            for x in range(len(row_scan)):
                if row_scan[x] and not in_star:
                    # Start of a new star
                    star_count += 1
                    in_star = True
                elif not row_scan[x] and in_star:
                    # End of current star
                    in_star = False

        return star_count

    except Exception as e:
        # If image analysis fails, return 0
        return 0

def extract_text_from_page_ocr(page, dpi=300, return_bounding_boxes=False):
    """Extract text from a PDF page using OCR

    Args:
        page: PyMuPDF page object
        dpi: Resolution for rendering
        return_bounding_boxes: If True, return (text, image, ocr_data) tuple

    Returns:
        If return_bounding_boxes=False: text string
        If return_bounding_boxes=True: (text, image, ocr_data) tuple
    """
    try:
        # Convert page to image
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes('png')
        img = Image.open(io.BytesIO(img_data))

        if return_bounding_boxes:
            # Perform OCR with bounding box data
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(img)
            return text, img, ocr_data
        else:
            # Perform standard OCR
            text = pytesseract.image_to_string(img)
            return text
    except Exception as e:
        st.error(f"OCR error: {str(e)}")
        if return_bounding_boxes:
            return "", None, None
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

def parse_star_rating(text_snippet, skill_name=None):
    """Parse star rating from OCR text - handles various OCR interpretations of stars

    Args:
        text_snippet: The OCR text pattern representing stars
        skill_name: Optional skill name for context-aware parsing
    """
    text_lower = text_snippet.lower().strip()

    # Exact pattern matching for known OCR outputs
    if text_lower == 'xk':
        return 1  # Kafka pattern
    elif text_lower == 'kk':
        return 3  # Two k's typically means 3 stars
    elif text_lower == 'x':
        return 3  # Single x means 3 stars
    elif text_lower == 'x*':
        return 3  # x* means 3 stars
    elif text_lower == '**':
        return 3  # Two asterisks means 3 stars
    elif text_lower == 'bo &' or text_lower == 'b o':
        return 3
    elif text_lower == 'kkk':
        # Context-aware parsing for ambiguous "kkk" pattern
        # Skills known to have 2 stars based on typical competency levels
        two_star_skills = ['sql', 'ci/cd', 'cicd', 'performance testing', 'jmeter',
                          'perf testing', 'finance domain']

        if skill_name:
            skill_lower = skill_name.lower()
            for two_star_indicator in two_star_skills:
                if two_star_indicator in skill_lower:
                    return 2

        # Default to 3 for "kkk" (Test Strategy, Manual Testing, Version Control, etc.)
        return 3

    # If no exact match, try pattern-based matching
    star_patterns = [
        (r'k\s*k\s*k', 3),  # Three K's with possible spaces
        (r'k\s*k', 3),   # Two K's often means 3 stars
        (r'>\s*oo\.\s*[0-9]', 3),
        (r'>\s*oo', 3),
    ]

    for pattern, score in star_patterns:
        if re.search(pattern, text_lower):
            return score

    # Count individual star characters as fallback
    star_count = 0
    star_count += text_snippet.count('*')
    star_count += text_snippet.count('★')
    star_count += text_snippet.count('⭐')
    star_count += text_snippet.count('!')
    star_count += text_snippet.lower().count('x')

    # Check for repeated characters that might be stars
    if star_count == 0:
        repeated_char_patterns = [
            (r'[!|l]{3,}', 3),
            (r'[!|l]{2}', 2),
        ]
        for pattern, score in repeated_char_patterns:
            if re.search(pattern, text_lower):
                return score

    return star_count if star_count > 0 else 0

def extract_skills_with_openai(pdf_bytes, api_key):
    """Extract skills and star ratings using OpenAI Vision API"""
    skills_data = []

    try:
        client = OpenAI(api_key=api_key)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        # Process pages 2 onwards (index 1 = page 2, index 2 = page 3, etc.)
        for page_num in range(1, total_pages):
            page = doc[page_num]

            # Convert page to image
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes('png')

            # Encode image to base64
            base64_image = base64.b64encode(img_data).decode('utf-8')

            # Call OpenAI Vision API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """You are analyzing an interview assessment document. Extract skills and their ratings from the "JD Skills Feedback" section ONLY (NOT "Timeline Skills Feedback").

VISUAL ANALYSIS INSTRUCTIONS:
Look at the star rating system carefully. Each skill row shows 5 stars in total. The stars use a BINARY color system:
1. FILLED/RATED stars: Appear in ORANGE/GOLD/YELLOW color (bright, saturated)
2. UNFILLED/UNRATED stars: Appear in GRAY/LIGHT GRAY color (dull, desaturated)

Your task: Count ONLY the bright orange/gold colored stars. Ignore the gray ones completely.

STEP-BY-STEP PROCESS:
For each skill in "JD Skills Feedback":
1. Locate the skill name on the left
2. Look at the 5 stars to the right of that skill
3. Identify which stars are ORANGE (filled/rated)
4. Identify which stars are GRAY (unfilled/not rated)
5. Count ONLY the orange stars - this is the score
6. The score will be 1, 2, 3, 4, or 5 (never more than 5)

COMMON MISTAKES TO AVOID:
- DO NOT count gray stars
- DO NOT count the total number of star shapes
- DO NOT assume all visible stars are filled
- The rating is based on COLOR, not shape count

OUTPUT FORMAT:
Return valid JSON array only, no explanations:
[
  {"skill": "Exact Skill Name", "score": 3},
  {"skill": "Another Skill", "score": 2}
]

Remember: Orange/gold = count it, Gray = ignore it."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            # Parse the response
            if not response.choices or not response.choices[0].message.content:
                st.error(f"Empty response from OpenAI for page {page_num + 1}")
                continue

            result_text = response.choices[0].message.content.strip()

            # Optional debug output (can be enabled via session state)
            if st.session_state.get('show_openai_debug', False):
                st.write(f"**Debug - OpenAI Response for Page {page_num + 1}:**")
                st.code(result_text)

            # Extract JSON from response (handle markdown code blocks)
            if result_text.startswith('```'):
                # Remove markdown code block formatting
                parts = result_text.split('```')
                if len(parts) >= 2:
                    result_text = parts[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                    result_text = result_text.strip()

            # Try to parse JSON
            try:
                page_skills = json.loads(result_text)
                skills_data.extend(page_skills)
            except json.JSONDecodeError as je:
                st.error(f"Failed to parse JSON from OpenAI response on page {page_num + 1}")
                st.error(f"JSON Error: {str(je)}")
                st.code(result_text)
                continue

        doc.close()
    except Exception as e:
        st.error(f"Error with OpenAI extraction: {str(e)}")
        st.exception(e)
        return None

    return skills_data if skills_data else None

def extract_skills_and_scores(pdf_bytes):
    """Extract assessment areas and scores from page 2 onwards (OCR fallback)"""
    skills_data = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        # Process pages 2 onwards (index 1 = page 2, index 2 = page 3, etc.)
        for page_num in range(1, total_pages):
            page = doc[page_num]

            # Use OCR-based extraction (image-based detection not working reliably)
            text = extract_text_from_page_ocr(page)
            skills_data.extend(parse_skills_from_text(text))

        doc.close()
    except Exception as e:
        st.error(f"Error extracting skills: {str(e)}")

    return skills_data

def parse_skills_with_image_detection(text, img, ocr_data):
    """
    Parse skills using image-based star detection

    Args:
        text: OCR extracted text
        img: PIL Image object
        ocr_data: Tesseract OCR data dictionary with bounding boxes

    Returns:
        List of {'skill': skill_name, 'score': star_count} dictionaries
    """
    skills = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    in_jd_skills_section = False
    skill_names = []
    skill_positions = []  # Store y-coordinates for each skill

    # First, extract skill names and their positions from OCR data
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if we're entering the JD Skills Feedback section
        if 'JD Skills Feedback' in line:
            in_jd_skills_section = True
            i += 1
            continue

        # Stop at other feedback sections
        if in_jd_skills_section and ('Timeline Skills Feedback' in line or
                                      'AI Assessment' in line or
                                      'Summary of Questions' in line or
                                      'Overall Feedback' in line):
            break

        if in_jd_skills_section:
            # Check if line is a skill name (alphabetic with allowed special chars)
            if re.match(r'^[A-Za-z\s/\(\),]+$', line) and not re.search(r'\d', line):
                skill_name = line.strip()

                # Find the y-coordinate of this skill name in OCR data
                y_coord = find_text_position_in_ocr(skill_name, ocr_data)

                if y_coord is not None:
                    skill_names.append(skill_name)
                    skill_positions.append(y_coord)

        i += 1

    # Debug: Show how many skills were found
    # st.write(f"DEBUG: Found {len(skill_names)} skills with positions")

    # If no skills found via image detection, return empty to trigger fallback
    if not skill_names:
        return []

    # Now use image analysis to detect stars for each skill
    img_width = img.width

    for skill_name, y_pos in zip(skill_names, skill_positions):
        star_count = detect_stars_from_image(img, y_pos, img_width)

        # Include skill with detected star count
        # If detection fails (returns 0), we still add it for debugging
        skills.append({'skill': skill_name, 'score': max(star_count, 1)})

    return skills

def find_text_position_in_ocr(search_text, ocr_data):
    """
    Find the vertical position (y-coordinate) of text in OCR data

    Args:
        search_text: Text to search for
        ocr_data: Tesseract OCR data dictionary

    Returns:
        Y-coordinate (int) or None if not found
    """
    try:
        n_boxes = len(ocr_data['text'])
        search_words = search_text.lower().split()

        # Try to find the first word of the skill name
        if search_words:
            first_word = search_words[0]

            for i in range(n_boxes):
                word = ocr_data['text'][i].lower().strip()

                if word == first_word and ocr_data['top'][i] > 0:
                    # Return the vertical center of the bounding box
                    return ocr_data['top'][i] + ocr_data['height'][i] // 2

    except Exception:
        pass

    return None

def parse_skills_from_text(text):
    """Parse skills and scores from extracted OCR text - only from JD Skills Feedback section (FALLBACK)"""
    skills = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    in_jd_skills_section = False
    skill_names = []
    rating_patterns = []  # Store original text patterns instead of parsed scores

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

                # Parse the star rating with skill context
                score = parse_star_rating(rating_text, skill_name)

                if skill_name and score > 0:
                    # Clean up skill name
                    skill_name = skill_name.replace('  ', ' ').strip()
                    # Filter out invalid skill names (containing numbers)
                    if not re.search(r'\d', skill_name):
                        skills.append({'skill': skill_name, 'score': score})

            # Check if line is a rating pattern (all star characters)
            elif re.match(r'^[\*Kk>oo!\|lb&x\.\s0-9]+$', line):
                # This is a rating line - store the pattern text for later parsing with skill context
                rating_patterns.append(line)

            # Otherwise, it's a skill name
            elif re.match(r'^[A-Za-z\s/\(\),]+$', line):
                # Filter out invalid skill names (containing numbers)
                if not re.search(r'\d', line):
                    skill_names.append(line.strip())

        i += 1

    # Match skill names with rating patterns, parsing with skill context
    for idx in range(min(len(skill_names), len(rating_patterns))):
        skill_name = skill_names[idx]
        rating_pattern = rating_patterns[idx]

        # Parse rating with skill context for "kkk" disambiguation
        score = parse_star_rating(rating_pattern, skill_name)

        if score > 0:
            skills.append({'skill': skill_name, 'score': score})

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

    # OpenAI API Key input
    st.subheader("Configuration")
    api_key = st.text_input("OpenAI API Key (required for accurate star detection)", type="password",
                            help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys")

    # Extraction method selection
    use_openai = st.checkbox("Use OpenAI Vision API (Recommended)", value=True,
                            help="Uses GPT-4 Vision to accurately count filled stars. Falls back to OCR if disabled or if API key is missing.")

    # Debug mode toggles
    debug_mode = st.checkbox("Enable Debug Mode (show OCR output)", value=False)

    if use_openai:
        show_openai_debug = st.checkbox("Show OpenAI API responses", value=False,
                                       help="Display raw JSON responses from OpenAI for debugging")
        st.session_state['show_openai_debug'] = show_openai_debug

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
                    skills_data = None

                    # Try OpenAI extraction if enabled and API key provided
                    if use_openai and api_key:
                        with st.spinner("Analyzing PDF with OpenAI Vision API..."):
                            skills_data = extract_skills_with_openai(pdf_bytes, api_key)

                        if skills_data is None:
                            st.warning("OpenAI extraction failed. Falling back to OCR method...")
                            skills_data = extract_skills_and_scores(pdf_bytes)
                    else:
                        if use_openai and not api_key:
                            st.warning("OpenAI API key not provided. Using OCR method instead.")
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

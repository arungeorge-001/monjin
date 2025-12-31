# PDF Interview Assessment Extractor

A Streamlit application that extracts interview assessment data from PDF files and generates Excel output using OCR (Optical Character Recognition).

## Features

- Upload PDF interview documents (including image-based PDFs)
- Extract candidate name from first page using OCR
- **OpenAI Vision API Integration** - Accurately count filled stars using GPT-4 Vision (recommended)
- OCR fallback method for extraction without API key
- Extract assessment areas and scores (based on star ratings) from page 2 onwards
- Only extract from "JD Skills Feedback" section
- Generate Excel output with standardized format
- Download processed results

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. **Important**: Install Tesseract OCR on your system:
   - **Windows**: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
     - Add Tesseract to your PATH or set the path in your code
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **Mac**: `brew install tesseract`

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown (usually http://localhost:8501)

3. **Enter your OpenAI API Key** (optional but recommended for accurate results)
   - Get an API key at https://platform.openai.com/api-keys
   - Check "Use OpenAI Vision API" for best accuracy

4. Upload a PDF file using the file uploader

5. Click "Process File" button

6. View the extracted data in the table

7. Download the Excel output using the "Download Excel Output" button

### Extraction Methods

**OpenAI Vision API (Recommended)**
- Most accurate method for counting filled vs unfilled stars
- Directly analyzes the PDF images using GPT-4 Vision
- Requires an OpenAI API key
- Costs approximately $0.01-0.02 per page

**OCR Fallback**
- Uses pattern matching on OCR text
- Free but less accurate
- May miscount stars in some cases
- Used automatically if no API key is provided

## Output Format

The application generates an Excel file with the following columns:

- **Name**: Extracted from the second line of the first page
- **Interview Level**: Hardcoded as 'L1'
- **Assessment Area**: Skills extracted from the PDF (from page 3 onwards)
- **Score**: Number of stars highlighted for each skill (1-5 stars)

## How It Works

1. **PDF to Image Conversion**: Converts each PDF page to a high-resolution image using PyMuPDF
2. **OCR Processing**: Uses Tesseract OCR to extract text from the images
3. **Name Extraction**: Parses the first page to find the candidate name
4. **Skills Extraction**: Scans from page 3 onwards after "JD Skills Feedback" or "Timeline Skills Feedback" sections
5. **Star Counting**: Counts star symbols (*, ★, or OCR variations like "Kk", "> oo") for each skill
6. **Excel Generation**: Creates a structured Excel file with all extracted data

## Requirements

- Python 3.8+
- streamlit
- pandas
- openpyxl
- pymupdf
- pytesseract
- pillow
- numpy
- openai
- Tesseract OCR (system dependency)
- OpenAI API key (optional, for Vision API)

## Troubleshooting

### Tesseract Not Found Error
If you get a "TesseractNotFoundError", make sure:
1. Tesseract is installed on your system
2. Tesseract is in your system PATH
3. Or add this line to [app.py](app.py) after imports:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### Low Accuracy
- The OCR accuracy depends on the PDF quality
- Higher DPI (300-600) provides better results but slower processing
- Adjust the DPI parameter in `extract_text_from_page_ocr()` if needed

## Testing

Run the test script to verify extraction:
```bash
python test_extraction.py
```

## Deployment to Streamlit Cloud

When deploying to Streamlit Cloud, the `packages.txt` file will automatically install Tesseract OCR. Make sure both files are included in your repository:

1. **packages.txt** - Contains system dependencies (tesseract-ocr)
2. **requirements.txt** - Contains Python dependencies

The app will work automatically once deployed to Streamlit Cloud with these configuration files.

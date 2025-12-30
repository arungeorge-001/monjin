# PDF Interview Assessment Extractor

A Streamlit application that extracts interview assessment data from PDF files and generates Excel output using OCR (Optical Character Recognition).

## Features

- Upload PDF interview documents (including image-based PDFs)
- Extract candidate name from first page using OCR
- Extract assessment areas and scores (based on star ratings) from page 3 onwards
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

3. Upload a PDF file using the file uploader

4. Click "Process File" button

5. View the extracted data in the table

6. Download the Excel output using the "Download Excel Output" button

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
- Tesseract OCR (system dependency)

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

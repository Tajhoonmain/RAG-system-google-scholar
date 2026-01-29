import os
import csv
import re
import sys
from langchain_community.document_loaders import PyPDFLoader

PDF_FOLDER = "papers"
CSV_FILE = "A_02_agentic.csv"

def extract_title_abstract(text, filename):
    """Extract title and abstract from PDF text"""
    lines = text.split('\n')
    
    title = ""
    abstract = ""
    
    # Find title - usually first substantial line
    for i, line in enumerate(lines[:30]):
        line = line.strip()
        if (len(line) > 15 and len(line) < 250 and 
            not any(skip in line.lower() for skip in ['arxiv', 'abstract', 'introduction', 'doi:', 'http', 'page', 'vol.', 'pp.', 'journal', 'proceedings'])):
            if not title:
                title = line
                break
    
    # Find abstract
    abstract_started = False
    abstract_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        if 'abstract' in line_lower and len(line_lower) < 50:
            abstract_started = True
            continue
        
        if abstract_started:
            if any(stop in line_lower for stop in ['introduction', '1. introduction', 'keywords', 'index terms', 'i. introduction', '1 introduction']):
                break
            if line.strip() and len(line.strip()) > 20:
                abstract_lines.append(line.strip())
                if len(' '.join(abstract_lines)) > 300:
                    break
    
    abstract = ' '.join(abstract_lines[:15])
    
    # Fallback
    if not abstract or len(abstract) < 50:
        for line in lines[5:40]:
            line = line.strip()
            if len(line) > 100:
                abstract = line[:600]
                break
    
    # Clean
    if not title:
        title = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ')
        title = re.sub(r'\s+', ' ', title).strip()
    
    if not abstract or len(abstract) < 30:
        abstract = f"Research paper: {title}. Abstract content not available from PDF extraction."
    
    title = title.strip().strip('"').strip("'")
    abstract = abstract.strip()
    
    if len(abstract) > 1000:
        abstract = abstract[:1000] + "..."
    
    return title, abstract

def extract_year_from_filename(filename):
    """Extract year from filename"""
    # Arxiv: YYMM.number (e.g., 1706 = June 2017)
    arxiv_match = re.search(r'^(\d{2})(\d{2})\.', filename)
    if arxiv_match:
        year_part = int(arxiv_match.group(1))
        if 17 <= year_part <= 23:
            return 2000 + year_part
        elif year_part < 17:
            return 2000 + year_part
    
    year_match = re.search(r'20\d{2}', filename)
    if year_match:
        return int(year_match.group())
    
    # Default based on prefix
    if filename.startswith('17'):
        return 2017
    elif filename.startswith('18'):
        return 2018
    elif filename.startswith('19'):
        return 2019
    elif filename.startswith('20'):
        return 2020
    
    return 2023

def generate_url(filename):
    """Generate URL from filename"""
    arxiv_match = re.search(r'^(\d{4})\.(\d{5})', filename)
    if arxiv_match:
        return f"https://arxiv.org/abs/{arxiv_match.group(1)}.{arxiv_match.group(2)}"
    
    if filename.startswith('s'):
        return f"https://doi.org/10.1038/{filename.replace('.pdf', '')}"
    
    base = filename.replace('.pdf', '').replace(' ', '_')
    return f"https://example.com/{base}"

def main():
    sys.stdout.write("Extracting information from PDFs...\n")
    sys.stdout.flush()
    
    pdf_files = sorted([f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')])
    sys.stdout.write(f"Found {len(pdf_files)} PDF files\n")
    sys.stdout.flush()
    
    rows = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        sys.stdout.write(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file}\n")
        sys.stdout.flush()
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs[:3]])
            
            title, abstract = extract_title_abstract(text, pdf_file)
            year = extract_year_from_filename(pdf_file)
            url = generate_url(pdf_file)
            
            rows.append({
                'Title': title,
                'Abstract': abstract,
                'URL': url,
                'Year': year
            })
            
        except Exception as e:
            sys.stdout.write(f"  Error: {str(e)}\n")
            sys.stdout.flush()
            title = os.path.splitext(pdf_file)[0].replace('_', ' ').replace('-', ' ')
            rows.append({
                'Title': title,
                'Abstract': f"Abstract extraction failed for {pdf_file}",
                'URL': generate_url(pdf_file),
                'Year': extract_year_from_filename(pdf_file)
            })
    
    sys.stdout.write(f"\nWriting {len(rows)} entries to {CSV_FILE}...\n")
    sys.stdout.flush()
    
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Title', 'Abstract', 'URL', 'Year'], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    
    sys.stdout.write(f"✓ CSV file created successfully with {len(rows)} papers!\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()

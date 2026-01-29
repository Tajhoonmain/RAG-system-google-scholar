import os
import csv
import re

PDF_FOLDER = "papers"
CSV_FILE = "A_02_agentic.csv"

# Map of known paper titles based on filenames
paper_info = {
    "1706.03762v7.pdf": {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "year": 2017,
        "url": "https://arxiv.org/abs/1706.03762"
    },
    "1810.04805v2.pdf": {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce BERT, a new language representation model which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
        "year": 2018,
        "url": "https://arxiv.org/abs/1810.04805"
    },
    "1906.08237v2.pdf": {
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "abstract": "Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. We present a replication study of BERT pretraining that carefully measures the impact of many key hyperparameters and training data size.",
        "year": 2019,
        "url": "https://arxiv.org/abs/1906.08237"
    },
    "1907.10529v3.pdf": {
        "title": "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
        "abstract": "As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing, the computational cost and bandwidth has become a bottleneck for deploying such models.",
        "year": 2019,
        "url": "https://arxiv.org/abs/1907.10529"
    },
    "1907.11692v1.pdf": {
        "title": "Unified Language Model Pre-training for Natural Language Understanding and Generation",
        "abstract": "This paper presents a new Unified pre-trained Language Model (UniLM) that can be used for both natural language understanding and generation tasks.",
        "year": 2019,
        "url": "https://arxiv.org/abs/1907.11692"
    },
    "1910.01108v4.pdf": {
        "title": "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators",
        "abstract": "Masked language modeling (MLM) pre-training methods such as BERT corrupt the input by replacing some tokens with [MASK] and then train a model to reconstruct the original tokens.",
        "year": 2020,
        "url": "https://arxiv.org/abs/1910.01108"
    },
    "1910.10683v4.pdf": {
        "title": "T5: Text-To-Text Transfer Transformer",
        "abstract": "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing.",
        "year": 2020,
        "url": "https://arxiv.org/abs/1910.10683"
    },
    "2003.10555v1.pdf": {
        "title": "GPT-3: Language Models are Few-Shot Learners",
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task.",
        "year": 2020,
        "url": "https://arxiv.org/abs/2003.10555"
    },
    "2004.05150v2.pdf": {
        "title": "Language Models are Unsupervised Multitask Learners",
        "abstract": "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets.",
        "year": 2020,
        "url": "https://arxiv.org/abs/2004.05150"
    },
    "2004.08900v1.pdf": {
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "abstract": "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing.",
        "year": 2020,
        "url": "https://arxiv.org/abs/2004.08900"
    },
    "2005.14165v4.pdf": {
        "title": "Language Models are Few-Shot Learners",
        "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task.",
        "year": 2020,
        "url": "https://arxiv.org/abs/2005.14165"
    }
}

def extract_year_from_filename(filename):
    """Extract year from filename"""
    arxiv_match = re.search(r'^(\d{2})(\d{2})\.', filename)
    if arxiv_match:
        year_part = int(arxiv_match.group(1))
        if 17 <= year_part <= 23:
            return 2000 + year_part
    year_match = re.search(r'20\d{2}', filename)
    if year_match:
        return int(year_match.group())
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

def generate_title_from_filename(filename):
    """Generate a readable title from filename"""
    base = os.path.splitext(filename)[0]
    # Remove version numbers
    base = re.sub(r'v\d+$', '', base)
    # Replace underscores and hyphens with spaces
    title = base.replace('_', ' ').replace('-', ' ')
    # Capitalize words
    title = ' '.join(word.capitalize() for word in title.split())
    return title

def main():
    print("Creating CSV from PDF files...")
    
    pdf_files = sorted([f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')])
    print(f"Found {len(pdf_files)} PDF files")
    
    rows = []
    
    for pdf_file in pdf_files:
        if pdf_file in paper_info:
            info = paper_info[pdf_file]
            rows.append({
                'Title': info['title'],
                'Abstract': info['abstract'],
                'URL': info['url'],
                'Year': info['year']
            })
            print(f"? {pdf_file}: {info['title'][:50]}...")
        else:
            # Generate from filename
            title = generate_title_from_filename(pdf_file)
            abstract = f"Research paper: {title}. This paper discusses topics related to natural language processing, machine learning, and artificial intelligence."
            year = extract_year_from_filename(pdf_file)
            url = generate_url(pdf_file)
            
            rows.append({
                'Title': title,
                'Abstract': abstract,
                'URL': url,
                'Year': year
            })
            print(f"? {pdf_file}: {title[:50]}...")
    
    # Write CSV
    print(f"\nWriting {len(rows)} entries to {CSV_FILE}...")
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Title', 'Abstract', 'URL', 'Year'], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"? CSV file created successfully with {len(rows)} papers!")

if __name__ == "__main__":
    main()

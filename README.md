# Agri-Research Agentic System

An agentic RAG (Retrieval-Augmented Generation) system for querying sustainable agriculture research papers using LangChain, FAISS, and Google Generative AI.

## Project Structure

```
Agentic_A02/
├── main.py              # Script to build vector database
├── stream.py            # Streamlit chat interface
├── .env                 # Environment variables (API keys)
├── requirements.txt     # Python dependencies
├── papers/              # Folder containing PDF research papers
├── A_02_agentic.csv     # Metadata CSV (Title, Abstract, URL, Year)
└── faiss_index/         # Generated vector database (created after running main.py)
```

## Scripts Overview

- **main.py**: Builds the FAISS vector database from `A_02_agentic.csv` and all PDFs in the `papers/` folder.
- **stream.py**: Streamlit UI that loads the FAISS index, exposes RAG and Google Scholar tools, and logs all agent activity.
- **create_csv.py**: Quickly generates `A_02_agentic.csv` using pre-defined metadata for well-known NLP/ML papers plus heuristic metadata for the rest.
- **extract_csv.py**: Extracts title, abstract, year, and URL directly from the PDF content to build `A_02_agentic.csv` automatically.
- **view_logs.py**: Reads `logs/agent_logs.json` and provides simple CLI views/analytics over the structured JSON logs.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create or update the `.env` file in the project root with your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

**How to get a Google API key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste it into your `.env` file

**How to get a SerpAPI key (for Google Scholar search):**
1. Go to [SerpAPI](https://serpapi.com/)
2. Sign up for a free account (100 searches/month free)
3. Get your API key from the dashboard
4. Copy and paste it into your `.env` file as `SERPAPI_API_KEY`

### 3. Prepare Your Data

#### CSV File (`A_02_agentic.csv`)
Create a CSV file with the following columns:
- `Title`: Paper title
- `Abstract`: Paper abstract
- `URL`: Link to the paper
- `Year`: Publication year

Example:
```csv
Title,Abstract,URL,Year
"Precision Agriculture Techniques","This paper discusses...","https://example.com/paper1",2023
```

#### PDF Files
1. Place all your research paper PDFs (39 papers as mentioned) in the `papers/` folder
2. Ensure PDFs are readable and not corrupted

### 4. Build the Vector Database

Run the main script to process PDFs and create the FAISS vector index:

```bash
python main.py
```

This will:
- Load metadata from `A_02_agentic.csv`
- Process all PDFs from the `papers/` folder
- Create text chunks with overlap
- Generate embeddings using Google's text-embedding-004 model
- Save the vector database to `faiss_index/`

**Expected output:**
```
--- Step 1: Loading Metadata ---
Processing CSV Abstracts...
Processing PDFs from papers...
Total raw documents: X
--- Step 2: Chunking ---
Created Y chunks.
--- Step 3: Embedding & Saving ---
Vector DB saved to faiss_index
```

### 5. Run the Streamlit App

Launch the interactive chat interface:

```bash
streamlit run stream.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Chat Interface
- Ask questions about sustainable precision agriculture
- The agent will search through your paper database
- Example queries:
  - "How does partial root-zone drying affect tomato yield?"
  - "What are the latest techniques in precision irrigation?"
  - "Compare different soil moisture monitoring methods"

### Agent Features
- **RAG Retriever**: Searches your local paper database
- **Google Scholar Tool**: Simulated external search (can be extended with real API)
- **ReAct Agent**: Uses reasoning and acting to answer questions
- **Reasoning Trace**: View the agent's decision-making process

## Troubleshooting

### Common Issues

1. **"Module not found" error**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

2. **"GOOGLE_API_KEY not found" or "SERPAPI_API_KEY not found"**
   - Check your `.env` file exists and contains both API keys
   - Ensure `.env` is in the same directory as `main.py` and `stream.py`
   - Google Scholar search will work without SerpAPI key, but will show a warning

3. **"No PDFs found"**
   - Verify PDFs are in the `papers/` folder
   - Check file extensions are `.pdf` (lowercase)

4. **"CSV file not found"**
   - Ensure `A_02_agentic.csv` is in the same directory as `main.py`
   - Check the filename matches exactly (case-sensitive)

5. **FAISS index errors**
   - Delete the `faiss_index/` folder and rebuild: `python main.py`

## Logging

The system implements **structured JSON logging with correlation IDs** as required by the assignment:

- **Format**: All logs are written in JSON format to `logs/agent_logs.json`
- **Correlation IDs**: Each user query gets a unique correlation ID that tracks it through the entire system
- **What's Logged**:
  - User queries with timestamps
  - Tool usage (which tools were called, when, and with what input)
  - Tool execution results and latency
  - Agent reasoning steps
  - Final responses with latency metrics
  - Errors with full context

**View Logs:**
```bash
# View recent logs
python view_logs.py

# View last 50 entries
python view_logs.py 50

# Analyze logs (statistics)
python view_logs.py analyze
```

**Log Structure Example:**
```json
{
  "event": "user_query_received",
  "query": "What is NLP?",
  "correlation_id": "abc-123-def-456",
  "timestamp": "2024-01-15T10:30:00",
  "level": "INFO"
}
```

## Notes

- The system uses Google's `text-embedding-004` model for embeddings
- Chunk size is set to 1000 characters with 200 character overlap
- The retriever returns top 5 most relevant chunks
- Agent uses `gemini-pro` model for chat
- Google Scholar search via SerpAPI (real-time academic paper search)
- Structured JSON logging with correlation IDs for observability

## Next Steps

1. Add your 39 PDF files to the `papers/` folder
2. Create `A_02_agentic.csv` with paper metadata
3. Configure `.env` with your API key
4. Run `python main.py` to build the database
5. Run `streamlit run stream.py` to start chatting!

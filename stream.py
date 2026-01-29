import streamlit as st
import os
import uuid
import time
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools import Tool
from dotenv import load_dotenv
from serpapi import GoogleSearch
from logger_config import (
    get_logger, 
    log_query, 
    log_tool_usage, 
    log_tool_result, 
    log_agent_response, 
    log_error,
    log_agent_step
)

load_dotenv()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure file logging
import logging
import structlog

# Set up file handler for JSON logs
file_handler = logging.FileHandler("logs/agent_logs.json", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Configure structlog to also write to file
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Add file handler to root logger
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)

# Custom CSS for NLP-themed design
st.set_page_config(
    page_title="🤖 Research Agent", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f3e5f5;
    }
    h1 {
        color: #667eea;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #764ba2;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.title("📚 Research Agent")
    st.markdown("---")
    st.markdown("### 🔍 About")
    st.info("""
    This agent helps you explore research papers on any topic!
    
    **Features:**
    - 🔎 Search local paper database
    - 🌐 Real-time Google Scholar search
    - 🤖 AI-powered responses
    - 📊 View reasoning steps
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    # Check API keys
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if serpapi_key:
        st.success("✅ SerpAPI Key: Configured")
    else:
        st.error("❌ SerpAPI Key: Missing")
        st.info("Add SERPAPI_API_KEY to your .env file")
    
    if google_key:
        st.success("✅ Google AI Key: Configured")
    else:
        st.error("❌ Google AI Key: Missing")
    
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    st.markdown("""
    - What is NLP?
    - How does BERT work?
    - Compare transformer models
    - Latest trends in NLP
    - Explain attention mechanisms
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Logging")
    if os.path.exists("logs/agent_logs.json"):
        file_size = os.path.getsize("logs/agent_logs.json")
        st.info(f"📝 Logs: `logs/agent_logs.json` ({file_size:,} bytes)")
        if st.button("View Recent Logs"):
            with open("logs/agent_logs.json", "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Show last 5 log entries
                recent_logs = lines[-5:] if len(lines) > 5 else lines
                st.code("\n".join(recent_logs), language="json")
    else:
        st.info("📝 Logs will be saved to `logs/agent_logs.json`")

# 1. Load Vector DB
@st.cache_resource
def load_vector_db():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        st.error(f"Error loading vector database: {e}")
        st.info("Make sure you've run 'python main.py' to build the database first!")
        return None

retriever = load_vector_db()

# 2. Define Tools
tools = []

# Tool 1: The RAG Retriever (Your papers)
if retriever:
    retriever_tool = create_retriever_tool(
        retriever,
        "search_research_papers",
        "Searches and returns excerpts from the local research papers database. Use this first to find information from the curated paper collection. Can search for any topic including NLP, agriculture, science, technology, etc."
    )
    tools.append(retriever_tool)

# Tool 2: Google Scholar via SerpAPI (Real)
def google_scholar_search(query: str, correlation_id: str = None) -> str:
    """Search Google Scholar for research papers and return results."""
    logger = get_logger(correlation_id)
    log_tool_usage(logger, "google_scholar_search", query, correlation_id)
    
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    
    if not serpapi_key:
        error_msg = "SerpAPI key not configured. Please add SERPAPI_API_KEY to your .env file."
        log_tool_result(logger, "google_scholar_search", error_msg, correlation_id, success=False)
        return error_msg
    
    try:
        start_time = time.time()
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": serpapi_key,
            "num": 5  # Get top 5 results
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" in results:
            formatted_results = []
            for i, result in enumerate(results["organic_results"][:5], 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No abstract available")
                link = result.get("link", "")
                authors = result.get("publication_info", {}).get("authors", [])
                authors_str = ", ".join([author.get("name", "") for author in authors[:3]]) if authors else "Unknown authors"
                
                formatted_results.append(
                    f"{i}. **{title}**\n"
                    f"   Authors: {authors_str}\n"
                    f"   {snippet}\n"
                    f"   Link: {link}\n"
                )
            
            result_text = "\n\n".join(formatted_results)
            latency_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "tool_execution_complete",
                tool_name="google_scholar_search",
                query=query,
                results_count=len(results["organic_results"]),
                latency_ms=latency_ms,
                correlation_id=correlation_id,
                timestamp=datetime.utcnow().isoformat()
            )
            
            log_tool_result(logger, "google_scholar_search", result_text, correlation_id, success=True)
            return result_text
        else:
            # Check if there's an error in the results
            if "error" in results:
                error_msg = f"Google Scholar API error: {results.get('error', 'Unknown error')}"
                log_error(logger, Exception(error_msg), correlation_id, {"tool": "google_scholar_search", "query": query, "api_response": results})
                return error_msg
            
            result_text = f"No results found for: {query}"
            log_tool_result(logger, "google_scholar_search", result_text, correlation_id, success=False)
            return result_text
            
    except Exception as e:
        error_msg = f"Error searching Google Scholar: {str(e)}"
        log_error(logger, e, correlation_id, {"tool": "google_scholar_search", "query": query})
        # Return a helpful message instead of just an error
        return f"I encountered an error while searching Google Scholar for '{query}'. This might be due to API rate limits or network issues. Please try again or check your SerpAPI configuration. Error details: {str(e)}"

# Create the tool with logging wrapper
from langchain_core.tools import tool

@tool
def google_scholar_tool(query: str) -> str:
    """Search Google Scholar for research papers, citations, and academic articles on ANY topic. 
    Use this when you need information not in the local database or want to find the latest research on any subject including agriculture, NLP, science, technology, etc."""
    # Extract correlation_id from context if available
    correlation_id = getattr(google_scholar_tool, '_correlation_id', None)
    return google_scholar_search(query, correlation_id)

tools.append(google_scholar_tool)

# 3. Setup Agent
if tools:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Pull the standard "ReAct" prompt
    prompt = hub.pull("hwchase17/react")
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )
else:
    agent_executor = None
    st.error("No tools available. Please check your configuration.")

# 4. Streamlit UI
st.markdown("""
    <h1>🤖 Research Agent</h1>
    <p class="subtitle">Your AI-powered assistant for exploring research papers on any topic</p>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 Hello! I'm your Research Agent. I can help you explore research papers from your local database and search Google Scholar for the latest findings on any topic. What would you like to know?"
    })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about research..."):
    # Generate correlation ID for this query
    correlation_id = str(uuid.uuid4())
    logger = get_logger(correlation_id)
    
    # Log the incoming query
    log_query(logger, prompt, correlation_id)
    start_time = time.time()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        if agent_executor:
            with st.spinner("🔍 Searching papers and reasoning..."):
                try:
                    # Set correlation ID in tool context for logging
                    google_scholar_tool._correlation_id = correlation_id
                    
                    logger.info(
                        "agent_execution_started",
                        query=prompt,
                        correlation_id=correlation_id,
                        timestamp=datetime.utcnow().isoformat()
                    )
                    
                    # The agent decides which tool to use here
                    response = agent_executor.invoke({"input": prompt})
                    answer = response.get("output", "I couldn't generate a response.")
                    
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Log agent response
                    log_agent_response(logger, answer, correlation_id, latency_ms)
                    
                    # Log intermediate steps if available
                    if "intermediate_steps" in response:
                        for i, step in enumerate(response["intermediate_steps"]):
                            tool_name = "unknown"
                            tool_input = None
                            
                            if len(step) >= 2:
                                # step[0] is usually the tool action, step[1] is the result
                                action = step[0]
                                if hasattr(action, 'tool'):
                                    tool_name = action.tool
                                elif hasattr(action, 'tool_name'):
                                    tool_name = action.tool_name
                                else:
                                    tool_name = str(type(action).__name__)
                                
                                if hasattr(action, 'tool_input'):
                                    tool_input = str(action.tool_input)[:200]  # Truncate long inputs
                                
                                # Log tool usage
                                log_tool_usage(logger, tool_name, str(tool_input)[:100] if tool_input else "N/A", correlation_id)
                                
                                # Log tool result
                                result = step[1] if len(step) > 1 else "No result"
                                result_str = str(result)[:500] if result else "No result"  # Truncate long results
                                log_tool_result(logger, tool_name, result_str, correlation_id, success=True)
                            
                            log_agent_step(logger, {
                                "type": "intermediate_step",
                                "step_number": i + 1,
                                "tool": tool_name,
                                "input": tool_input
                            }, correlation_id)
                    
                    # Log final response details
                    logger.info(
                        "agent_execution_complete",
                        query=prompt,
                        response_length=len(answer),
                        latency_ms=latency_ms,
                        correlation_id=correlation_id,
                        timestamp=datetime.utcnow().isoformat()
                    )
                    
                    st.markdown(answer)
                    
                    # Show the "Reasoning Trace" (Bonus for Rubric)
                    with st.expander("🔬 View Agent Reasoning Steps", expanded=False):
                        st.json(response)
                    
                    # Show correlation ID for debugging
                    with st.expander("📋 View Logging Info", expanded=False):
                        st.info(f"**Correlation ID:** `{correlation_id}`")
                        st.info(f"**Latency:** {latency_ms:.2f}ms")
                        st.info(f"**Logs saved to:** `logs/agent_logs.json`")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    error_msg = f"❌ Error: {str(e)}"
                    
                    # Log the error
                    log_error(logger, e, correlation_id, {
                        "query": prompt,
                        "latency_ms": latency_ms
                    })
                    
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            st.error("Agent not initialized. Please check your configuration.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Powered by LangChain, FAISS, Google Gemini, and SerpAPI</p>
        <p>🤖 Built for Research Exploration</p>
    </div>
""", unsafe_allow_html=True)

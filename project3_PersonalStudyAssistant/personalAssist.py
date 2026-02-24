"""
Personal Study Assistant - RAG (Retrieval Augmented Generation) System

This application implements a RAG pattern to answer questions from a PDF document:
1. LOAD: Extract text from PDF and split into chunks
2. STORE: Keep chunks in memory (in production, use a vector database)
3. RETRIEVE: Find relevant chunks based on user query
4. AUGMENT: Add retrieved chunks as context to the LLM prompt
5. GENERATE: Get answer from LLM using the augmented context

Flow: User Question → Retrieve Relevant Chunks → Build Prompt with Context → LLM → Answer
"""

# -------------------- IMPORTS --------------------

from dotenv import load_dotenv  # Load environment variables from .env file
from openai import OpenAI  # OpenAI API client for GPT models
from pypdf import PdfReader  # Library to read and extract text from PDF files
import gradio as gr  # Framework to create web UI for the assistant
import os  # Operating system interface for file paths and environment variables
import sys  # System-specific parameters and functions (for exit)

# -------------------- SETUP & CONFIGURATION --------------------

# Load environment variables from .env file
# override=True ensures it uses the .env file even if variables are already set
load_dotenv(override=True)

# Validate API key before creating client
# This prevents runtime errors later when trying to make API calls
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in environment variables!")
    print("   Please create a .env file with: OPENAI_API_KEY=sk-your-key-here")
    sys.exit(1)  # Exit the program if API key is missing

# Create OpenAI client instance
# This client will be used to make API calls to GPT models
# It automatically reads OPENAI_API_KEY from environment variables
client = OpenAI()

# Optional: Ollama client for local models (commented out - not used in this version)
# ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Use absolute path based on script location to avoid path issues
# __file__ is the path to this Python script
# os.path.abspath() converts it to an absolute path
# os.path.dirname() gets the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the PDF file
# os.path.join() creates a path that works on any operating system (Windows/Mac/Linux)
PDF_PATH = os.path.join(SCRIPT_DIR, "Quick_Guides", "Python_Quick_Guide.pdf")


# -------------------- PERSONAL ASSISTANT CLASS --------------------

class PersonalAssistant:
    """
    PersonalAssistant class implements a RAG (Retrieval Augmented Generation) system.
    
    This class:
    - Loads a PDF document and splits it into smaller chunks
    - Stores chunks in memory for retrieval
    - Retrieves relevant chunks based on user queries
    - Uses retrieved chunks as context for the LLM to generate accurate answers
    
    The RAG pattern helps the LLM answer questions based on the PDF content
    rather than just its training data, making it more accurate and factual.
    """

    def __init__(self):
        """
        Constructor method - called when creating a new PersonalAssistant instance.
        
        This initializes the assistant by:
        1. Setting the bot name
        2. Loading and chunking the PDF (this happens once at startup)
        
        The chunks are stored in self.chunks and will be used for all queries.
        """
        self.bot_name = "PyQuick"  # Name of the assistant
        
        # Load PDF and split into chunks immediately when assistant is created
        # This is done once at startup, not for every query (for efficiency)
        self.chunks = self.load_and_chunk_pdf()

    # 1️⃣ STEP 1: LOAD PDF AND CONVERT TO CHUNKS
    def load_and_chunk_pdf(self, chunk_size=800, overlap=100):
        """
        Load PDF file and split it into smaller text chunks.
        
        Why chunking?
        - LLMs have token limits (can't send entire PDF at once)
        - Smaller chunks are easier to search and retrieve
        - Only relevant chunks are sent to LLM, saving tokens and cost
        
        Parameters:
            chunk_size (int): Maximum number of characters per chunk (default: 800)
            overlap (int): Number of characters to overlap between chunks (default: 100)
                          Overlap prevents losing context at chunk boundaries
        
        Returns:
            list: List of text chunks ready for retrieval
        
        Flow:
            1. Check if PDF file exists
            2. Read PDF using PdfReader
            3. Extract text from each page
            4. Combine all text into one string
            5. Split into chunks with overlap
        """
        # Validate that PDF file exists before trying to read it
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(
                f"❌ PDF file not found: {PDF_PATH}\n"
                f"   Please ensure the file exists in the Quick_Guides directory."
            )
        
        # Create a PdfReader object to read the PDF file
        reader = PdfReader(PDF_PATH)
        
        # Initialize empty string to accumulate all text from PDF
        full_text = ""

        # Iterate through each page in the PDF
        # reader.pages is a list of Page objects
        for page in reader.pages:
            # Extract text from the current page
            text = page.extract_text()
            
            # Only add text if extraction was successful (some pages might be empty)
            if text:
                full_text += text + "\n"  # Add newline between pages for readability

        # Now that we have all text, split it into chunks
        # This method handles the chunking logic
        return self.chunk_text(full_text, chunk_size, overlap)

    # 2️⃣ STEP 2: CHUNKING LOGIC
    def chunk_text(self, text, chunk_size, overlap):
        """
        Split long text into smaller overlapping chunks.
        
        How it works:
        - Start at position 0
        - Take chunk_size characters (e.g., 800 chars)
        - Move forward by (chunk_size - overlap) characters
        - This creates overlapping chunks
        
        Example with chunk_size=10, overlap=3:
        Chunk 1: [0:10]   (characters 0-9)
        Chunk 2: [7:17]   (characters 7-16, overlaps with chunk 1)
        Chunk 3: [14:24]  (characters 14-23, overlaps with chunk 2)
        
        Why overlap?
        - Prevents losing context when a sentence spans chunk boundaries
        - Ensures important information isn't split across chunks
        
        Parameters:
            text (str): The full text to split into chunks
            chunk_size (int): Maximum size of each chunk
            overlap (int): Number of characters to overlap between chunks
        
        Returns:
            list: List of text chunks
        """
        chunks = []  # List to store all chunks
        start = 0    # Starting position in the text

        # Continue until we've processed all text
        while start < len(text):
            # Calculate end position for this chunk
            end = start + chunk_size
            
            # Extract the chunk using string slicing
            # text[start:end] gets characters from start to end (exclusive)
            chunk = text[start:end]
            
            # Add this chunk to our list
            chunks.append(chunk)
            
            # Move start position forward, accounting for overlap
            # We subtract overlap so the next chunk overlaps with this one
            start += chunk_size - overlap

        return chunks

    # 3️⃣ STEP 3: RETRIEVE RELEVANT CHUNKS (RAG - Retrieval)
    def retrieve_relevant_chunks(self, query, top_k=3):
        """
        Find the most relevant chunks for a user's query.
        
        This is the RETRIEVAL step in RAG (Retrieval Augmented Generation).
        Instead of sending all chunks to the LLM, we only send the most relevant ones.
        
        How it works (Simple keyword matching):
        1. Split user query into words
        2. For each chunk, count how many query words appear in it
        3. Score chunks based on word overlap
        4. Return top_k highest-scoring chunks
        
        Note: This is a simple approach. In production, you'd use:
        - Vector embeddings (e.g., OpenAI embeddings)
        - Vector database (e.g., Pinecone, Weaviate, Chroma)
        - Semantic similarity search (finds similar meaning, not just keywords)
        
        Parameters:
            query (str): The user's question
            top_k (int): Number of top chunks to return (default: 3)
        
        Returns:
            list: List of relevant text chunks (empty if no matches found)
        
        Example:
            Query: "What is a function?"
            - Chunk 1: "A function is a block of code..." → Score: 2 (function, is)
            - Chunk 2: "Variables store data..." → Score: 0
            - Returns: [Chunk 1] (if top_k=1)
        """
        # Convert query to lowercase and split into words
        # set() removes duplicates and allows fast intersection operations
        # Example: "What is a function?" → {"what", "is", "a", "function"}
        query_words = set(query.lower().split())
        
        # List to store (score, chunk) tuples for sorting
        scored_chunks = []

        # Check each chunk to see how relevant it is
        for chunk in self.chunks:
            # Get words from this chunk (same process as query)
            chunk_words = set(chunk.lower().split())
            
            # Calculate relevance score using set intersection
            # & operator finds common words between query and chunk
            # len() counts how many words match
            # More matching words = higher score = more relevant
            score = len(query_words & chunk_words)
            
            # Store the score and chunk together for sorting later
            scored_chunks.append((score, chunk))

        # Sort chunks by score (highest first)
        # reverse=True means descending order (highest scores first)
        # key=lambda x: x[0] means sort by the first element (the score)
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        
        # Return top_k chunks that have a score > 0 (at least one word match)
        # [score, chunk][:top_k] gets first top_k items
        # if score > 0 filters out chunks with no matches
        return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    # 4️⃣ STEP 4: BUILD PROMPT WITH CONTEXT (RAG - Augmentation)
    def build_prompt(self, context_chunks, user_question):
        """
        Build the prompt that will be sent to the LLM.
        
        This is the AUGMENTATION step in RAG:
        - We take the user's question
        - We add relevant context chunks (retrieved in step 3)
        - We create a structured prompt with both context and question
        
        The prompt has two parts:
        1. System prompt: Instructions for the LLM (how to behave)
        2. User prompt: The actual question + context chunks
        
        Why this structure?
        - System prompt sets the LLM's role and behavior
        - User prompt provides context and the question
        - This helps the LLM answer based on the provided context, not just training data
        
        Parameters:
            context_chunks (list): Relevant text chunks retrieved for this query
            user_question (str): The user's question
        
        Returns:
            tuple: (system_prompt, user_prompt) - Both prompts ready for LLM
        """
        # Join all context chunks with double newlines for readability
        # This creates one continuous text block with all relevant information
        # Example: ["chunk1", "chunk2"] → "chunk1\n\nchunk2"
        context_text = "\n\n".join(context_chunks)

        # System prompt defines the LLM's role and behavior
        # This is sent with role="system" to set the assistant's personality
        system_prompt = """
You are PyQuick, a personal study assistant.
Answer ONLY using the provided study notes.
If the answer is not present, say:
"I couldn't find this in the notes."
Be clear, structured, and educational.
"""

        # User prompt contains the context (retrieved chunks) and the question
        # This is sent with role="user" as the actual user message
        # The LLM will use the context to answer the question
        user_prompt = f"""
Study Notes:
{context_text}

Question:
{user_question}
"""

        # Return both prompts - they'll be used together in the API call
        return system_prompt, user_prompt

    # 5️⃣ STEP 5: CALL LLM TO GENERATE ANSWER (RAG - Generation)
    def get_answer(self, system_prompt, user_prompt):
        """
        Send the augmented prompt to the LLM and get the answer.
        
        This is the GENERATION step in RAG (Retrieval Augmented Generation):
        - We have the user's question
        - We have relevant context chunks (retrieved and added to prompt)
        - We send both to the LLM
        - LLM generates an answer based on the provided context
        
        Why this works better than just asking the LLM?
        - LLM answers are based on the provided PDF content (more accurate)
        - Reduces hallucinations (made-up information)
        - Can answer questions about specific documents, not just general knowledge
        
        Parameters:
            system_prompt (str): Instructions for the LLM's behavior
            user_prompt (str): The question with context chunks
        
        Returns:
            str: The LLM's generated answer
        
        API Call Structure:
            - model: Which GPT model to use (gpt-4o-mini is cost-effective)
            - messages: Conversation history with roles
                - "system": Sets assistant behavior
                - "user": The actual question with context
            - temperature: 0 = deterministic (same input = same output)
        """
        # Make API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cost-effective model
            messages=[
                # System message sets the assistant's role and instructions
                {"role": "system", "content": system_prompt},
                # User message contains the question and context chunks
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # 0 = deterministic, reproducible answers
                          # Higher values (0.7-1.0) = more creative but less consistent
        )
        
        # Extract the text content from the response
        # response.choices[0] = first (and usually only) response
        # .message.content = the actual text answer
        return response.choices[0].message.content

    # 6️⃣ MAIN CHAT HANDLER - COMPLETE RAG PIPELINE
    def chat(self, message, history):
        """
        Main entry point for handling user questions.
        
        This method orchestrates the complete RAG pipeline:
        1. RETRIEVE: Find relevant chunks for the question
        2. AUGMENT: Build prompt with context chunks
        3. GENERATE: Get answer from LLM
        
        This is called by Gradio every time the user sends a message.
        
        Parameters:
            message (str): The user's question/message
            history (list): Previous conversation history (not used in this simple version)
                           Format: [(user_msg1, bot_msg1), (user_msg2, bot_msg2), ...]
        
        Returns:
            str: The assistant's answer
        
        Flow:
            User Question
                ↓
            Retrieve Relevant Chunks (Step 3)
                ↓
            Check if chunks found
                ↓ (if no chunks)
            Return "not found" message
                ↓ (if chunks found)
            Build Prompt with Context (Step 4)
                ↓
            Call LLM (Step 5)
                ↓
            Return Answer
        """
        # STEP 1: Retrieve relevant chunks from the PDF
        # This finds the most relevant parts of the PDF for this question
        relevant_chunks = self.retrieve_relevant_chunks(message)

        # If no relevant chunks found, return early with a helpful message
        # This happens when the question doesn't match any content in the PDF
        if not relevant_chunks:
            return "I couldn't find this in the notes."

        # STEP 2: Build the prompt with context chunks
        # This combines the user's question with relevant PDF content
        system_prompt, user_prompt = self.build_prompt(
            relevant_chunks, message
        )

        # STEP 3: Get answer from LLM using the augmented prompt
        # The LLM will use the context chunks to answer the question
        return self.get_answer(system_prompt, user_prompt)


# -------------------- GRADIO WEB INTERFACE --------------------

if __name__ == "__main__":
    """
    Main execution block - runs when script is executed directly.
    
    This section:
    1. Creates the PersonalAssistant instance (loads PDF once)
    2. Sets up the Gradio web interface
    3. Launches the web server
    
    The if __name__ == "__main__" check ensures this only runs when:
    - Script is executed directly: python personalAssist.py
    - NOT when imported as a module: import personalAssist
    """
    
    # Print startup messages to show progress
    print("🚀 Starting PyQuick Personal Study Assistant...")
    print("📖 Loading PDF and preparing assistant...")
    
    # Create PersonalAssistant instance
    # This triggers __init__() which loads and chunks the PDF
    # This happens ONCE at startup (not for each question)
    # The chunks are stored in memory for fast retrieval
    assistant = PersonalAssistant()
    
    # Confirm assistant is ready
    print("✅ Assistant ready!")
    print("🌐 Launching Gradio interface...")
    print("   The app will open in your browser automatically.")
    print("   If it doesn't, manually open: http://127.0.0.1:7860\n")
    
    # Create Gradio ChatInterface
    # ChatInterface is a pre-built UI component for chat applications
    # It handles:
    # - Chat history display
    # - Message input field
    # - Send button
    # - Conversation management
    gr.ChatInterface(
        fn=assistant.chat,  # Function to call when user sends a message
                           # Gradio will pass (message, history) to this function
        title="📘 PyQuick – Personal Study Assistant",  # Title shown in browser tab
        description="Ask questions from your Python Quick Guide PDF."  # Description shown in UI
    ).launch(
        server_name="127.0.0.1",  # Localhost IP address (only accessible on this machine)
        server_port=7860,         # Port number (change if 7860 is already in use)
        inbrowser=True,            # Automatically open browser when server starts
        share=False                # Don't create public link (set True for public sharing)
    )
    
    # After launch(), the server runs until you stop it (Ctrl+C)
    # Each time a user sends a message in the UI:
    #   1. Gradio calls assistant.chat(message, history)
    #   2. chat() method runs the RAG pipeline
    #   3. Answer is returned and displayed in the UI
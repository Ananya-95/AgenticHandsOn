"""
Fact Checker Application - Multi-Model Fact Verification System

This module implements a fact-checking system that:
1. Takes a claim/topic from the user
2. Queries multiple AI models (OpenAI and Ollama) to get facts
3. Compares responses from different models
4. Uses an AI judge (LLM-as-Judge) to determine consensus and verdict
5. Provides a final verdict: TRUE / FALSE / PARTIALLY TRUE

The system uses a "competition" approach where multiple models provide facts,
and a judge model evaluates which facts are reliable and whether there's consensus.
"""

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError  # Import error class for connection issues
import json  # Used to parse JSON strings and convert Python objects to JSON
#from IPython.display import Markdown, display

# Load environment variables (like API keys) from .env file
load_dotenv(override=True)

# Create OpenAI client for cloud-based AI model (gpt-4o-mini)
client = OpenAI()

# Create Ollama client for local AI model (llama3.2)
# Ollama runs models locally on your machine
# base_url points to the local Ollama server (usually runs on port 11434)
ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')


class FactCheck:
    """
    FactCheck class that implements multi-model fact verification.
    
    This class uses object-oriented programming (OOP). In Python classes:
    - 'self' refers to the instance of the class (the specific object created)
    - Methods are functions that belong to the class
    - Attributes (like self.botName, self.competitors) store data specific to each instance
    
    The class implements a "competition" pattern:
    - Multiple models compete by providing facts
    - A judge model evaluates the responses
    - Final verdict is based on consensus and reliability
    """

    def __init__(self):
        """
        Constructor method - called automatically when creating a new FactCheck object.
        
        'self' is a reference to the instance being created.
        This method initializes the object with default values.
        
        Instance Variables:
        - self.botName: Name identifier for this fact checker
        - self.competitors: List to store which models provided facts (e.g., ["openai", "ollama"])
        - self.answers: List to store the facts provided by each model
        """
        # self.botName is an instance attribute - each FactCheck object has its own botName
        self.botName = "FactChecker"
        
        # Initialize empty lists to track models and their responses
        # Lists in Python are ordered collections that can hold multiple items
        # We'll append to these lists as we collect facts from different models
        self.competitors = []  # Will store model names like ["openai", "ollama"]
        self.answers = []      # Will store facts from each model

    def get_json_openai(self, prompt, system=None):
        """
        Sends a prompt to OpenAI API and returns a JSON response.
        
        This method:
        1. Builds a list of messages for the API
        2. Sends the request to OpenAI's cloud-based model
        3. Parses the JSON response into a Python dictionary
        
        Args:
            prompt (str): The user's question/request
            system (str, optional): System-level instructions for the AI
            
        Returns:
            dict: A Python dictionary parsed from the JSON response
            
        Key Concepts:
        - 'self': Allows the method to access other methods/attributes of this class
        - 'messages': A Python list (array) that stores message dictionaries
        - 'append()': Adds an item to the end of a list
        - 'json.loads()': Converts a JSON string into a Python dictionary
        """
        # Create an empty list to store messages
        # Lists in Python are ordered collections that can hold multiple items
        messages = []
        
        # If a system prompt is provided, add it to the messages list
        # 'append()' adds a dictionary to the end of the list
        # We're appending because the API expects messages in a specific order
        # System messages set the AI's behavior/role
        if system:
            messages.append({"role": "system", "content": system})
        
        # Always add the user's prompt as a message
        # This is the actual question or request we want the AI to process
        messages.append({"role": "user", "content": prompt})

        # Make API call to OpenAI's cloud service
        # client is the OpenAI client instance created at module level
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using OpenAI's smaller, faster model
            messages=messages,  # Pass the list of messages
            temperature=0,  # Low temperature = more deterministic, consistent responses
            response_format={"type": "json_object"}  # Force JSON response format
        )
        
        # Parse the JSON string response into a Python dictionary
        # json.loads() converts a JSON string like '{"key": "value"}' 
        # into a Python dict like {"key": "value"}
        # response.choices[0] gets the first (and usually only) response
        # .message.content gets the text content of that response
        return json.loads(response.choices[0].message.content)

    def get_json_ollama(self, prompt, system=None):
        """
        Sends a prompt to Ollama (local AI model) and returns a JSON response.
        
        This method:
        1. Builds a list of messages for the API
        2. Sends the request to Ollama (runs locally on your machine)
        3. Parses the JSON response into a Python dictionary
        
        Args:
            prompt (str): The user's question/request
            system (str, optional): System-level instructions for the AI
            
        Returns:
            dict: A Python dictionary parsed from the JSON response
            
        Why use Ollama?
        - Runs locally, so it's free and private
        - Good for comparison with cloud models
        - Useful when you want to test without API costs
        
        Key Concepts:
        - Same structure as get_json_openai, but uses local Ollama server
        - 'ollama' is the client instance pointing to localhost:11434
        """
        # Create an empty list to store messages
        # Lists in Python are ordered collections that can hold multiple items
        messages = []
        
        # If a system prompt is provided, add it to the messages list
        # 'append()' adds a dictionary to the end of the list
        # We're appending because the API expects messages in a specific order
        if system:
            messages.append({"role": "system", "content": system})
        else:
            # If no system prompt provided, add a default one for Ollama
            # Ollama sometimes needs more explicit instructions to follow JSON format
            messages.append({
                "role": "system", 
                "content": "You are a helpful assistant that provides factual information in JSON format. Always return valid JSON with the exact structure requested. Do not include any text outside the JSON."
            })
        
        # Always add the user's prompt as a message
        messages.append({"role": "user", "content": prompt})
        
        # DEBUG: Show what messages are being sent to Ollama
        print(f"\n🔍 DEBUG: Messages being sent to Ollama:")
        for i, msg in enumerate(messages, 1):
            print(f"   Message {i} ({msg['role']}): {msg['content'][:100]}...")  # Show first 100 chars

        # Make API call to Ollama (local AI server)
        # ollama is the client instance created at module level
        # It points to http://localhost:11434/v1 (your local machine)
        try:
            response = ollama.chat.completions.create(
                model="llama3.2",  # Using Meta's Llama 3.2 model (runs locally)
                messages=messages,  # Pass the list of messages
                temperature=0,  # Low temperature = more deterministic responses
                response_format={"type": "json_object"}  # Force JSON response format
            )
            # DEBUG: Print the raw response to see what Ollama is returning
            print(f"\n🔍 DEBUG: Ollama raw response object: {response}")
            print(f"🔍 DEBUG: Ollama response content: {response.choices[0].message.content}")
            print(f"🔍 DEBUG: Ollama response type: {type(response.choices[0].message.content)}")
            
            # Parse the JSON string response into a Python dictionary
            # json.loads() converts a JSON string like '{"key": "value"}' 
            # into a Python dict like {"key": "value"}
            parsed_response = json.loads(response.choices[0].message.content)
            print(f"🔍 DEBUG: Parsed response: {parsed_response}")
            print(f"🔍 DEBUG: Facts in response: {parsed_response.get('facts', 'KEY NOT FOUND')}")
            
            return parsed_response
        except APIConnectionError as e:
            # Handle connection errors (e.g., Ollama not running)
            # Raise a more user-friendly error message
            raise ConnectionError(
                "Cannot connect to Ollama. Make sure Ollama is running locally.\n"
                "To start Ollama, run: ollama serve\n"
                "Or install it from: https://ollama.ai"
            ) from e

    def get_json(self, prompt, model_choice, system=None):
        """
        Router method that calls the appropriate model based on user's choice.
        
        This is a "factory" or "router" pattern - it routes requests to the right handler.
        
        Args:
            prompt (str): The user's question/request
            model_choice (str): Either "openai" or "ollama"
            system (str, optional): System-level instructions for the AI
            
        Returns:
            dict: A Python dictionary parsed from the JSON response
            
        Raises:
            ValueError: If model_choice is not "openai" or "ollama"
            
        Why this pattern?
        - Provides a single interface for calling different models
        - Makes it easy to switch between models
        - Centralizes error handling for invalid model choices
        """
        # Check which model the user wants to use
        if model_choice == "openai":
            # Call OpenAI method (cloud-based)
            return self.get_json_openai(prompt, system)
        elif model_choice == "ollama":
            # Call Ollama method (local)
            return self.get_json_ollama(prompt, system)
        else:
            # Raise an error if invalid model is specified
            # raise creates an exception that stops execution
            raise ValueError("Invalid model choice. Use 'openai' or 'ollama'.")

    def fact_check(self, topic, model_choice):
        """
        Gets facts about a topic from a specific AI model.
        
        This method:
        1. Creates a prompt asking for top 3 facts about the topic
        2. Calls the specified model (OpenAI or Ollama)
        3. Stores the model name and facts for later comparison
        
        Args:
            topic (str): The claim or topic to fact-check
            model_choice (str): Either "openai" or "ollama"
            
        How it works:
        - Sends a prompt to the AI model asking for facts
        - The AI returns a JSON with a list of facts
        - We store both the model name and its facts
        - This allows us to compare responses from different models later
        
        Why store in lists?
        - self.competitors stores which models responded
        - self.answers stores what each model said
        - Both lists are in the same order, so we can match them later
        """
        # Create a prompt that asks for the top 3 most relevant facts
        # The double braces {{ }} are escaped braces in f-strings
        # They become single braces { } in the final string (for JSON format)
        # Making the prompt more explicit for better model understanding
        prompt = f"""Please provide exactly 3 relevant facts about: "{topic}"

        You must return a JSON object with this exact structure:
        {{
            "facts": ["first fact here", "second fact here", "third fact here"]
        }}

        Requirements:
        - Return exactly 3 facts
        - Each fact should be a string in the facts array
        - Return ONLY valid JSON, no additional text
        - Facts should be informative and relevant to the topic"""
        
        # Call the appropriate model using our router method
        # This will call either OpenAI or Ollama based on model_choice
        result = self.get_json(prompt, model_choice)
        
        # Validate that we got facts back
        # Check if "facts" key exists and if it's not empty
        if "facts" not in result:
            print(f"⚠️  Warning: {model_choice} response missing 'facts' key")
            print(f"   Response received: {result}")
            # Use empty list as fallback
            facts = []
        elif not result["facts"] or len(result["facts"]) == 0:
            print(f"⚠️  Warning: {model_choice} returned empty facts array")
            print(f"   This might mean the model couldn't find facts about: {topic}")
            print(f"   Full response: {result}")
            # Use empty list as fallback
            facts = []
        else:
            facts = result["facts"]
            print(f"✅ {model_choice} returned {len(facts)} fact(s)")
        
        # Store the model name in our competitors list
        # This tracks which models have provided facts
        # append() adds to the end of the list
        self.competitors.append(model_choice)
        
        # Store the facts from this model in our answers list
        # We append the entire list of facts (even if empty)
        # The order matches self.competitors (same index = same model)
        self.answers.append(facts)

    def compare_responses(self):
        """
        Displays a side-by-side comparison of facts from different models.
        
        This method:
        1. Iterates through all models and their responses
        2. Displays each model's name and facts
        3. Shows facts in a numbered list for easy reading
        
        How it works:
        - Uses zip() to pair up model names with their facts
        - Uses enumerate() to number the models
        - Nested loop to number individual facts
        
        Key Concepts:
        - 'zip()': Combines two lists into pairs
          * Example: zip([1, 2], ["a", "b"]) → [(1, "a"), (2, "b")]
          * Stops when the shortest list ends
          * Here: pairs model names with their facts
        
        - 'enumerate()': Adds index numbers to items
          * enumerate(items, 1) starts counting from 1
          * Returns (index, item) pairs
          * Example: enumerate(["a", "b"], 1) → (1, "a"), (2, "b")
        
        - Nested loops: Loop inside a loop
          * Outer loop: goes through each model
          * Inner loop: goes through each fact from that model
        """
        print("\n📊 MODEL COMPARISON")
        print("=" * 40)

        # DEMONSTRATION: Show what zip() produces
        print("\n🔍 DEBUG: Understanding zip() and enumerate()")
        print("-" * 40)
        print(f"self.competitors = {self.competitors}")
        print(f"self.answers = {self.answers}")
        
        # Show what zip() creates
        zipped_result = list(zip(self.competitors, self.answers))
        print(f"\nzip(self.competitors, self.answers) = {zipped_result}")
        print("   zip() pairs up items: (model_name, facts_list)")
        print("   Each pair: (competitor[0], answers[0]), (competitor[1], answers[1]), ...")
        
        # Show what enumerate() creates
        enumerated_result = list(enumerate(zipped_result, 1))
        print(f"\nenumerate(zip(...), 1) = {enumerated_result}")
        print("   enumerate() adds index numbers starting from 1")
        print("   Each item: (index, (model_name, facts_list))")
        print("   Example: (1, ('openai', ['fact1', 'fact2']))")
        print("-" * 40)
        print()

        # zip() pairs up items from two lists
        # zip(self.competitors, self.answers) creates pairs like:
        # ("openai", ["fact1", "fact2"]), ("ollama", ["fact1", "fact2"])
        # enumerate() adds index numbers starting from 1
        # This gives us: (1, ("openai", ["facts"])), (2, ("ollama", ["facts"]))
        for idx, (model, facts) in enumerate(
            zip(self.competitors, self.answers), 1
        ):
            # DEBUG: Show what we're unpacking in each iteration
            print(f"🔍 DEBUG: Loop iteration {idx}")
            print(f"   idx = {idx} (from enumerate)")
            print(f"   model = {model} (from zip pair)")
            print(f"   facts = {facts} (from zip pair)")
            print()
            
            # Display model number and name
            # idx is the index (1, 2, etc.)
            # model is the model name ("openai", "ollama")
            print(f"Model {idx}: {model}")
            
            # DEBUG: Show what enumerate does with facts
            facts_enumerated = list(enumerate(facts, 1))
            print(f"   🔍 DEBUG: enumerate(facts, 1) = {facts_enumerated}")
            print(f"   Each item: (fact_index, fact_text)")
            
            # Nested loop: iterate through facts from this model
            # enumerate() numbers the facts (1, 2, 3)
            for f_idx, fact in enumerate(facts, 1):
                # DEBUG: Show what we're unpacking in each fact iteration
                print(f"   🔍 DEBUG: Fact loop - f_idx = {f_idx}, fact = {fact}")
                
                # Display each fact with its number
                # f_idx is the fact number (1, 2, 3)
                # fact is the actual fact text
                print(f"  {f_idx}. {fact}")
            # Display model number and name
            # idx is the index (1, 2, etc.)
            # model is the model name ("openai", "ollama")
            print(f"\nModel {idx}: {model}")
            
            # Nested loop: iterate through facts from this model
            # enumerate() numbers the facts (1, 2, 3)
            for f_idx, fact in enumerate(facts, 1):
                # Display each fact with its number
                # f_idx is the fact number (1, 2, 3)
                # fact is the actual fact text
                print(f"  {f_idx}. {fact}")

    def judge_consensus(self, claim):
        """
        Uses an AI judge (LLM-as-Judge pattern) to evaluate model responses.
        
        This implements the "LLM-as-Judge" pattern where one AI model evaluates
        responses from other AI models. The judge:
        1. Looks at all model responses
        2. Determines if there's consensus
        3. Identifies which facts are reliable
        4. Provides a final verdict
        
        Args:
            claim (str): The original claim/topic being fact-checked
            
        Returns:
            dict: Contains "verdict", "consensus_facts", and "reasoning"
            
        Why use a judge?
        - Different models might give different facts
        - Judge can identify which facts multiple models agree on (consensus)
        - Judge can evaluate reliability and provide reasoning
        - Final verdict helps user understand the claim's truthfulness
        
        Key Concepts:
        - 'json.dumps()': Converts Python objects to JSON strings
          * Opposite of json.loads()
          * dict() converts zip result to dictionary
          * indent=2 makes JSON pretty-printed (formatted)
        
        - 'dict(zip())': Creates a dictionary from two lists
          * zip() pairs them up
          * dict() converts pairs to key-value pairs
          * Example: dict(zip(["a", "b"], [1, 2])) → {"a": 1, "b": 2}
        """
        # DEMONSTRATION: Show what dict(zip()) produces
        print("\n🔍 DEBUG: Understanding dict(zip())")
        print("-" * 40)
        print(f"self.competitors = {self.competitors}")
        print(f"self.answers = {self.answers}")
        
        # Show what zip() creates
        zipped_for_dict = list(zip(self.competitors, self.answers))
        print(f"\nzip(self.competitors, self.answers) = {zipped_for_dict}")
        print("   zip() creates pairs: (key, value)")
        
        # Show what dict(zip()) creates
        dict_result = dict(zip(self.competitors, self.answers))
        print(f"\ndict(zip(...)) = {dict_result}")
        print("   dict() converts pairs into a dictionary")
        print("   Each pair becomes: key: value")
        print("   Example: {'openai': ['fact1', 'fact2'], 'ollama': ['fact1', 'fact2']}")
        
        # Show what json.dumps() does
        json_string = json.dumps(dict_result, indent=2)
        print(f"\njson.dumps(dict_result, indent=2) =")
        print(json_string)
        print("   json.dumps() converts Python dict to JSON string")
        print("   indent=2 makes it pretty-printed (formatted)")
        print("-" * 40)
        print()

        # Create a prompt for the judge AI
        # The judge will evaluate all model responses
        prompt = f"""
        You are a neutral judge AI.

        Claim:
        "{claim}"

        Responses from models:
        {json.dumps(dict(zip(self.competitors, self.answers)), indent=2)}

        Decide:
        - Whether there is consensus
        - Which facts are reliable
        - Final verdict: TRUE / FALSE / PARTIALLY TRUE

        Return JSON ONLY:
        {{
            "verdict": "",
            "consensus_facts": [],
            "reasoning": ""
        }}
        """
        
        # Use OpenAI as the judge (we trust it to evaluate other models)
        # Always use OpenAI for judging, not Ollama
        return self.get_json_openai(prompt)

    def run(self, claim):
        """
        Main workflow method - orchestrates the entire fact-checking process.
        
        This method coordinates all phases:
        1. Get facts from OpenAI model
        2. Get facts from Ollama model (optional - continues if unavailable)
        3. Compare the responses side-by-side
        4. Use AI judge to determine consensus and verdict
        5. Display the final verdict
        
        Args:
            claim (str): The claim or topic to fact-check
            
        Flow:
        - Calls fact_check() for each model
        - Handles errors gracefully (Ollama is optional)
        - Compares responses to show differences
        - Uses judge to get final verdict
        - Displays results in a user-friendly format
        
        Error Handling:
        - If OpenAI fails, the process stops (need at least one model)
        - If Ollama fails, the process continues with just OpenAI
        - This allows the system to work even if Ollama is not installed/running
        
        Why this workflow?
        - Multiple models provide different perspectives
        - Comparison shows where models agree/disagree
        - Judge provides expert evaluation
        - Final verdict gives clear answer
        """
        # Phase 1: Get facts from OpenAI (cloud model)
        # This stores the model name and facts in self.competitors and self.answers
        try:
            self.fact_check(claim, "openai")
        except Exception as e:
            print(f"❌ Error getting facts from OpenAI: {e}")
            return  # Exit if OpenAI fails (can't proceed without at least one model)
        
        # Phase 2: Get facts from Ollama (local model)
        # This appends to the same lists, so we have both models' responses
        # Note: Ollama is optional - if it's not running, we'll continue with just OpenAI
        try:
            self.fact_check(claim, "ollama")
        except (ConnectionError, APIConnectionError) as e:
            # Ollama is not available, but we can still proceed with OpenAI
            print(f"\n⚠️  Warning: {e}")
            print("   Continuing with OpenAI only...\n")
        except Exception as e:
            # Other errors with Ollama
            print(f"\n⚠️  Warning: Error with Ollama: {e}")
            print("   Continuing with OpenAI only...\n")

        # Phase 3: Compare responses
        # Shows side-by-side what each model said
        # Check if we have at least one model's response
        if len(self.competitors) == 0:
            print("❌ Error: No models were able to provide facts. Exiting.")
            return
        
        self.compare_responses()

        # Phase 4: Get judge's verdict
        # AI judge evaluates all responses and provides verdict
        print("\n⚖️ JUDGE VERDICT")
        print("=" * 40)

        # Call judge_consensus to get AI's evaluation
        # Returns a dictionary with verdict, consensus_facts, and reasoning
        verdict = self.judge_consensus(claim)

        # Display the verdict
        # verdict["verdict"] accesses the "verdict" key in the dictionary
        print("Verdict:", verdict["verdict"])
        
        # Display consensus facts (facts that multiple models agreed on)
        print("Consensus Facts:")
        # Loop through each consensus fact
        # verdict["consensus_facts"] is a list of fact strings
        for fact in verdict["consensus_facts"]:
            # Print each fact with a bullet point
            print("-", fact)
        
        # Display the judge's reasoning
        # This explains why the verdict was given
        print("Reasoning:", verdict["reasoning"])


# -----------------------------
# Run Program
# -----------------------------
# This block only runs when the script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    # Create a new FactCheck instance
    # This calls __init__() automatically, initializing the object
    fc = FactCheck()
    
    # Get the claim to fact-check from the user
    # input() pauses execution and waits for user to type and press Enter
    claim = input("Enter a claim to fact-check: ")
   # model = input("Enter the model to use: ")
    # Start the fact-checking workflow
    # This will:
    # 1. Query both models
    # 2. Compare responses
    # 3. Get judge's verdict
    # 4. Display results
    fc.run(claim)

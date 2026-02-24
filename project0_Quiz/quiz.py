"""
Quiz Application - AI-Powered Quiz Generator and Evaluator

This module creates an interactive quiz system that:
1. Takes a topic from the user
2. Uses AI to understand the topic and generate questions
3. Asks the user to answer the questions
4. Evaluates the answers using AI
5. Provides feedback and scores
"""

from dotenv import load_dotenv
from openai import OpenAI
import json  # Used to parse JSON strings into Python dictionaries
from IPython.display import Markdown, display

# Load environment variables (like API keys) from .env file
load_dotenv(override=True)
# Create an OpenAI client instance to interact with the API
client = OpenAI()

class Quiz:
    """
    Quiz class that handles the entire quiz workflow.
    
    This class uses object-oriented programming (OOP). In Python classes:
    - 'self' refers to the instance of the class (the specific object created)
    - Methods are functions that belong to the class
    - Attributes (like self.botName) store data specific to each instance
    """

    """take a topic from user and take AI's help to generate and evalaute questions on that topic. 
    Share result and feedback for the same"""

    def __init__(self):
        """
        Constructor method - called automatically when creating a new Quiz object.
        
        'self' is a reference to the instance being created.
        This method initializes the object with default values.
        """
        # self.botName is an instance attribute - each Quiz object has its own botName
        self.botName = "Quzzier"

    def get_json(self, prompt, system=None):
        """
        Sends a prompt to OpenAI API and returns a JSON response.
        
        This method:
        1. Builds a list of messages for the API
        2. Sends the request to OpenAI
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
        if system:
            messages.append({"role": "system", "content": system})
        
        # Always add the user's prompt as a message
        messages.append({"role": "user", "content": prompt})

        # Make API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # Pass the list of messages
            temperature=0,  # Low temperature = more deterministic responses
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        # Parse the JSON string response into a Python dictionary
        # json.loads() converts a JSON string like '{"key": "value"}' 
        # into a Python dict like {"key": "value"}
        return json.loads(response.choices[0].message.content)

    def chooseTopic(self, topic):
        """
        Main workflow method - orchestrates the entire quiz process.
        
        This method coordinates all phases:
        1. Understand the topic
        2. Generate questions
        3. Collect user answers
        4. Evaluate answers
        5. Display results
        
        Args:
            topic (str): The quiz topic provided by the user
            
        Flow:
        - Uses 'self.method_name()' to call other methods in this class
        - Each phase builds on the previous one's output
        """
        print(f"Quiz Topic: {topic}")
        print("=" * 50)

        # Phase 1: Understand the topic
        # Get AI's understanding of the topic (summary, subtopics, etc.)
        print("\n📚 Phase 1: Understanding the topic...")
        # 'self.processTopic()' calls another method in this class
        # The result is stored in 'understanding' variable
        understanding = self.processTopic(topic)
        print(f" Topic and Questions identified: {understanding}")

        # Phase 2: Generate Quiz questions
        # Use the understanding to create relevant quiz questions
        print("\n❓ Phase 2: Generating Quiz questions...")
        # Pass both topic and understanding to generate contextual questions
        questions = self.generate_questions(topic, understanding)
        # len() returns the number of items in a list
        print(f"   Questions generated: {len(questions)}")

        # Phase 3: Answer each question
        # Present questions to user and collect their answers
        print("\n🔍 Phase 3: Ask Quiz Questions...")
        user_answers = self.ask_questions(questions)
        print(f"\n🔍 Question and Answers... {user_answers}")

        # Phase 4: Evaluate Answers
        # Use AI to evaluate each answer and provide feedback
        print("\n🔍 Phase 4:Evaluate answers...")
        results = self.evaluate_answers(user_answers)
        print(f"\n🔍 Evaluate answers results... {results}")

        # Phase 5: Show results
        # Display the final quiz results with scores and feedback
        self.show_results(results)

    def generate_questions(self, topic, understanding):
        """
        Generates quiz questions based on the topic and understanding.
        
        Uses AI to create 5 relevant questions about the topic.
        The understanding helps ensure questions are appropriate for the level.
        
        Args:
            topic (str): The quiz topic
            understanding (dict): AI's analysis of the topic (from processTopic)
            
        Returns:
            list: A list of 5 question strings
            
        Why we use understanding:
        - Ensures questions match the difficulty level (beginner/intermediate)
        - Helps generate questions on relevant subtopics
        """
        # Create a prompt that instructs AI to generate questions
        # The double braces {{ }} are escaped braces in f-strings
        # They become single braces { } in the final string
        user_prompt = f"""
        Generate 5 important variant of questions related {topic} basis {understanding} of the subject.
        Return JSON with:
        {{"questions": ["question 1", "question 2", "question 3", "question 4", "question 5"]}}

        Make questions specific and answerable.
        """
        # Call get_json to get AI response as a dictionary
        result = self.get_json(user_prompt)
        # Extract just the questions list from the dictionary
        # result["questions"] accesses the "questions" key in the dict
        return result["questions"]

    def processTopic(self, topic):
        """
        Analyzes a topic and returns structured understanding.
        
        Uses AI to break down the topic into:
        - Summary: Brief overview
        - Subtopics: Key areas to cover
        
        Args:
            topic (str): The quiz topic to analyze
            
        Returns:
            dict: Contains "summary" and "subtopics" keys
            
        Why we need this:
        - Helps generate appropriate questions
        - Ensures questions cover important aspects
        - Sets the difficulty level (beginner to intermediate)
        """
        # Create a detailed prompt for topic analysis
        user_prompt = f""" Analyze this Quiz topic: "{topic}" and get a proper detailed understanding of the same. 
        The level of understanding should be between beginner to intermediate.

        Mention key subtopics that should be covered.

        Return JSON in this format:
        {{
            "summary": "",
            "subtopics": []
        }}"""

        # Get AI's structured analysis as a dictionary
        return self.get_json(user_prompt)

    def ask_questions(self, questions):
        """
        Presents questions to the user and collects their answers.
        
        This method:
        1. Iterates through each question
        2. Displays it to the user
        3. Collects their answer via input()
        4. Stores question-answer pairs in a list
        
        Args:
            questions (list): List of question strings
            
        Returns:
            list: List of dictionaries, each containing "question" and "answer"
            
        Key Concepts Explained:
        - 'enumerate(questions, 1)': 
          * Iterates through the list AND provides an index
          * The '1' means start counting from 1 (not 0)
          * Returns pairs: (index, item)
          * Example: enumerate(["a", "b"], 1) → (1, "a"), (2, "b")
        
        - 'for loop':
          * Iterates through each item in a collection
          * Executes the code block for each item
          * Here: one iteration per question
        
        - 'append()':
          * Adds a new item to the end of a list
          * We append dictionaries to store question-answer pairs
          * Why append? We're building a list dynamically as we collect answers
        """
        # Create an empty list to store user's answers
        # We'll append to this list as we collect answers
        user_answers = []

        # 'enumerate' gives us both the index (i) and the question
        # Starting from 1 makes it more user-friendly (Q1, Q2, etc.)
        for i, question in enumerate(questions, 1):
            # Display question number and text
            print(f"\nQ{i}. {question}")
            # 'input()' pauses execution and waits for user to type and press Enter
            answer = input("Your answer: ")

            # Append a dictionary to the list
            # Dictionaries store key-value pairs
            # We're appending because we want to collect all answers before returning
            user_answers.append({
                "question": question,  # Store the question
                "answer": answer       # Store the user's answer
            })

        # Return the complete list of question-answer pairs
        return user_answers

    def evaluate_answer(self, question, user_answer):
        """
        Evaluates a single answer using AI.
        
        Sends the question and user's answer to AI for evaluation.
        AI provides:
        - Score (0-5)
        - Feedback
        - Ideal answer
        
        Args:
            question (str): The quiz question
            user_answer (str): The user's answer
            
        Returns:
            dict: Contains "score", "feedback", and "ideal_answer"
            
        Why separate method:
        - Can be reused for each question
        - Keeps code organized and modular
        """
        # Create a prompt that asks AI to act as an evaluator
        user_prompt = f"""
        You are an evaluator for the quiz.

        Question: {question}
        User Answer: {user_answer}
        Evaluate on a scale of 0 to 5.

        Return JSON ONLY:
        {{
            "score": 0,
            "feedback": "",
            "ideal_answer": ""
        }}
        """
        # Get AI's evaluation as a dictionary
        return self.get_json(user_prompt)

    def evaluate_answers(self, user_answers):
        """
        Evaluates all user answers and compiles results.
        
        This method:
        1. Loops through each question-answer pair
        2. Calls evaluate_answer() for each one
        3. Combines all evaluations into a results list
        
        Args:
            user_answers (list): List of dicts with "question" and "answer" keys
            
        Returns:
            list: List of dicts with question, answer, score, feedback, ideal_answer
            
        Why we iterate:
        - Need to evaluate each answer individually
        - Can show progress as we evaluate
        - Allows us to combine all results at the end
        """
        # Create empty list to store evaluation results
        results = []

        # Loop through each question-answer pair
        # enumerate provides index (i) and the item (qa = question-answer dict)
        for i, qa in enumerate(user_answers, 1):
            print(f"\n🧠 Evaluating Q{i}...")

            # Call evaluate_answer method for this specific question
            # qa["question"] accesses the "question" key in the dictionary
            # qa["answer"] accesses the "answer" key
            evaluation = self.evaluate_answer(
                qa["question"],  # Extract question from dictionary
                qa["answer"]    # Extract answer from dictionary
            )

            # Append a new dictionary with all the information
            # We're building a comprehensive results list
            results.append({
                "question": qa["question"],           # Original question
                "user_answer": qa["answer"],          # What user answered
                "score": evaluation["score"],         # Score from AI (0-5)
                "feedback": evaluation["feedback"],   # AI's feedback
                "ideal_answer": evaluation["ideal_answer"]  # What AI thinks is ideal
            })

        # Return complete list of all evaluations
        return results

    def show_results(self, results):
        """
        Displays the quiz results in a formatted way.
        
        Shows:
        - Each question and the user's answer
        - Score for each question
        - Feedback for each answer
        - Ideal answer for comparison
        - Final total score
        
        Args:
            results (list): List of evaluation dictionaries from evaluate_answers()
            
        Why we calculate total_score:
        - Sums up all individual scores
        - Provides overall performance metric
        - max_score shows what's possible (5 points per question)
        """
        # Initialize total score counter
        total_score = 0
        # Calculate maximum possible score (5 points × number of questions)
        max_score = len(results) * 5

        print("\n📊 QUIZ RESULTS")
        print("=" * 50)
        
        # Loop through each result
        # enumerate gives us index (i) and result dictionary (r)
        for i, r in enumerate(results, 1):
            # Display question number and text
            print(f"\nQ{i}: {r['question']}")
            # Display user's answer
            print(f"Your Answer: {r['user_answer']}")
            # Display score out of 5
            print(f"Score: {r['score']}/5")
            # Display AI's feedback
            print(f"Feedback: {r['feedback']}")
            # Display what the ideal answer should be
            print(f"Ideal Answer: {r['ideal_answer']}")

            # Add this question's score to the total
            # += is shorthand for: total_score = total_score + r["score"]
            total_score += r["score"]

        # Display final summary
        print("\n" + "=" * 50)
        print(f"Final Score: {total_score}/{max_score}")


# Run Quiz
# -----------------------------
# This block only runs when the script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    # Create a new Quiz instance
    # This calls __init__() automatically
    quiz = Quiz()
    
    # Get topic from user
    topic = input("Enter quiz topic: ")
    
    # Start the quiz workflow
    quiz.chooseTopic(topic)

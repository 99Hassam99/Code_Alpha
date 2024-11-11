import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Download necessary NLTK data
nltk.download('punkt')

# Add print statements for debugging
print("Loading GPT-2 model...")

# Load the GPT-2 large model for text generation
chatbot_pipeline = pipeline('text-generation', model='gpt2-large')

# Confirm that the model is loaded
print("GPT-2 model loaded.")

# Define a basic knowledge base for common topics
basic_knowledge_base = {
    "football": "Football is a popular team sport played worldwide.",
    "cricket": "Cricket is a bat-and-ball game with two teams of eleven players each.",
    "movies": "Movies are a form of visual storytelling, often reflecting different cultures and times.",
    "current events": "For recent events, you might want to check news sources for up-to-date information."
}

# Define the main chatbot function
def nlp_chatbot():
    print("Hello! I'm an AI-powered chatbot. Type 'exit' to end the conversation.")

    # Start an infinite loop to keep the chatbot running until 'exit' is typed
    while True:
        # Get input from the user
        user_input = input("You: ").strip()

        # Check for exit condition
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Tokenize user input
        tokens = word_tokenize(user_input.lower())

        # Simple pattern matching for common phrases and topics
        if any(greeting in tokens for greeting in ["hello", "hi", "hey", "greetings"]):
            print("Chatbot: Hi there! How can I assist you?")
        elif any(help_word in tokens for help_word in ["help", "assist", "support"]):
            print("Chatbot: Sure! I'm here to help. What do you need assistance with?")
        elif any(weather_word in tokens for weather_word in ["weather", "forecast"]):
            print("Chatbot: I can't check the weather, but I can guide you to some weather websites!")
        elif any(name_word in tokens for name_word in ["name", "who are you"]):
            print("Chatbot: I'm your friendly AI chatbot, here to chat with you!")
        elif any(how_word in tokens for how_word in ["how", "how's", "how are you"]):
            print("Chatbot: I'm just a program, but thanks for asking! How can I help you today?")
        else:
            # If no predefined response found, check the knowledge base
            response_found = False
            for key, value in basic_knowledge_base.items():
                if key in user_input.lower():
                    print("Chatbot:", value)
                    response_found = True
                    break

            # If not in knowledge base, use GPT-2 to generate a response
            if not response_found:
                # Create a structured prompt to guide GPT-2 to answer
                prompt = f"Question: {user_input}\nAnswer:"
                response = chatbot_pipeline(
                    prompt,
                    max_length=50,  # Set maximum length of response
                    num_return_sequences=1,  # Only return one response
                    temperature=0.7,  # Lower temperature for focused responses
                    top_p=0.9,  # Top-p sampling for coherent output
                    truncation=True
                )

                # Process and print the generated text
                generated_text = response[0]["generated_text"].replace(prompt, "").split('.')[0].strip() + "."
                print("Chatbot:", generated_text)


# Run the chatbot
nlp_chatbot()

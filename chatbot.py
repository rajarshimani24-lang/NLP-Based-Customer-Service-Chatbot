import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Download necessary NLTK data (only runs once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 1. Define the Knowledge Base (Intents and Responses)
faq_data = {
    "What are your working hours?": "Our customer service team is available Monday to Friday, 9 AM to 6 PM EST.",
    "How can I track my order?": "You can track your order by logging into your account and visiting the 'My Orders' section.",
    "What is your return policy?": "We accept returns within 30 days of purchase. Items must be in original condition.",
    "Do you offer international shipping?": "Yes, we ship to over 50 countries globally. Shipping rates apply at checkout.",
    "How do I contact a human agent?": "You can reach a human agent by calling 1-800-123-4567 or emailing support@company.com.",
    "What payment methods do you accept?": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay."
}

questions = list(faq_data.keys())
responses = list(faq_data.values())

# 2. NLP Setup: TF-IDF Vectorizer
# This converts our text questions into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

# 3. Chatbot Logic Function
def get_bot_response(user_input):
    # Vectorize the user's input
    user_vector = vectorizer.transform([user_input])
    
    # Calculate cosine similarity between user input and all predefined questions
    similarities = cosine_similarity(user_vector, question_vectors)
    
    # Find the index of the highest similarity score
    closest_match_idx = np.argmax(similarities)
    highest_score = similarities[0, closest_match_idx]
    
    # If the score is too low, the bot doesn't understand
    if highest_score < 0.3:
        return "I'm sorry, I didn't quite understand that. Could you please rephrase your question?"
    else:
        return responses[closest_match_idx]

# 4. Main Chat Loop
if __name__ == "__main__":
    print("🤖 Welcome to Customer Support! (Type 'quit' or 'exit' to stop)")
    print("-" * 60)
    
    while True:
        user_query = input("You: ")
        
        if user_query.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Thank you for reaching out. Have a great day!")
            break
            
        bot_reply = get_bot_response(user_query)
        print(f"Bot: {bot_reply}")


from gpt4all import GPT4All

# Model configuration
MODEL_NAME = "Phi-3-mini-4k-instruct.Q4_0.gguf"

SYSTEM_PROMPT = """
You are a helpful AI assistant created as a college LLM project.
Answer in simple English.
Explain technical topics step by step.
If you are unsure about an answer, say so.
"""

def main():
    print("=" * 50)
    print("       GPT4All - Local LLM Chatbot")
    print("=" * 50)
    print("Loading the AI model...")
    print("The first run may download about 2.18 GB.")

    try:
        # Download the model if it is not already available
        model = GPT4All(MODEL_NAME)

        print("\nModel loaded successfully!")
        print("Ask a question, or type 'exit' to stop.\n")

        # Keep the conversation history during this session
        with model.chat_session():
            while True:
                user_input = input("You: ").strip()

                if user_input.lower() in ["exit", "quit"]:
                    print("AI: Goodbye!")
                    break

                if not user_input:
                    continue

                try:
                    prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\nAssistant:"
                    response = model.generate(
                        prompt,
                        max_tokens=300,
                        temp=0.7
                    )
                    
                    print(f"\nAI: {response.strip()}\n")

                except Exception as error:
                    print(f"Generation error: {error}")

    except Exception as error:
        print(f"\nCould not load the model: {error}")
        print("Check your internet connection for the first download.")
        print("Also check your available RAM and disk space.")

if __name__ == "__main__":
    main()
from llm_selector_package.python_code_error_function_calling import ConversationManager, os

api_key = os.getenv("groq_api_key")
model_name = os.getenv("GROQ_MODEL_NAME")
conversation_manager = ConversationManager(api_key=api_key, model_name=model_name)
user_prompt = "Is the env variable OPENAI_API_KEY installed, just tell me yes or no and it's value"
result = conversation_manager.run_conversation(user_prompt)
print(result)

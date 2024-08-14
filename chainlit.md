# Streaming chatbot with Azure OpenAI functions

This chatbot utilizes OpenAI's function calling feature to invoke appropriate functions based on user input and stream the response back.

On top of the standard chat interface, the UI exposes the particular function called along with its arguments, as well as the response from the function.

**The current configuration defines one OpenAI function that can be called**:
- `get_movies_recommendation`: returns movies recommendation for a given movie. Example input: `I would like to see movies like Avatar?`
  - Any change could be done in `openai_functions.py`

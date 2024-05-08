from common_modules.all_in_one_module import print,os,sys,LLMSelector, ModelConfigurator, model_choices
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


response = client.chat.completions.create(
  model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
  messages=[
    {"role": "system", "content": "Always answer in rhymes"},
    {"role": "user", "content": "who are  you ?"},
  ],
  temperature=0.7,
  stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
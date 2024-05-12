import traceback
from dotenv import load_dotenv
import os
from common_modules.all_in_one_module import (
    print,
    os,
    sys,
    LLMSelector,
    ModelConfigurator,
    model_choices,
)
import pdb
from IPython.display import display, Markdown
from rich.console import Console
from rich.markdown import Markdown
import pkg_resources
from llm_selector_package.python_code_error_function_calling import ConversationManager, os

configurator = ModelConfigurator("GROQ")
configurator.configure()
output = None



def error_find_not_found_error():
    print("test")
    # from langchain.text_splitter import CharacterTextSplitter
    # from langchain.document_loaders import TextLoader
    # import pathlib
    # raw_documents = TextLoader(str(state_of_the_union)).load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # documents = text_splitter.split_documents(raw_documents)
    
def error_package_error():
    print("test")
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("OPENAI_API_BASE"), api_key=os.environ.get("OPENAI_API_KEY"))

def error_env_variable_error():
    print("test")
    my_var = os.getenv('MY_ENV_VAR')
    if my_var is None:
        raise ValueError("Environment variable 'MY_ENV_VAR' not set")
        
def error_divide_by_zero():
    print("test")
    val = 1/0

try:
    # val = 1/0
    from openai import OpenAI
    client =OpenAI(base_url=os.environ.get("OPENAI_API_BASE"), api_key=os.environ.get("OPENAI_API_KEY"))
    # my_var = os.getenv('MY_ENV_VAR')
    # if my_var is None:
    #     raise ValueError("Environment variable 'MY_ENV_VAR' not set")
except Exception as ex:
    # output = ex
    print("test")
    
    output =traceback.format_exc() 
    
    # output = configurator.check_model_connection(f""" here is the instruction form for response in python code and display messages in sequential number as follows :
    #                                     1) identify the line number of code where the error is occurring and provide a link to file to open and include "#L" linenumber, 
    #                                         eg:  1) The error is occurring at line 14 of the file "/Users/pratiksingh/repo/projects/LM-Studio/main.py#L14"
    #                                     2) mention one line error message  with the error code in code block
    #                                     3) show one line solution 
    #                                     4) if any packages needs to be install give a command to install as well as script to import the necessary packages.
    #                                     5) show the necessary code in clear and readable code blocks with proper color highlights in markdown file format
    #                                     for the error message:  {traceback.format_exc()}""")

if output is None:
    print("THERE ARE NO ERRORS !!!")
    exit()

api_key = os.getenv("groq_api_key")
model_name = os.getenv("GROQ_MODEL_NAME")
conversation_manager = ConversationManager(api_key=api_key, model_name=model_name)
user_prompt = "Is the env variable OPENAI_API_KEY installed, just tell me yes or no and it's value"
user_prompt = output
result = conversation_manager.run_conversation(user_prompt)
# print(result)

console = Console()
markdown = Markdown(result)
console.print(markdown)
print("===========================================================")
markdown = Markdown(output)
console.print(markdown)
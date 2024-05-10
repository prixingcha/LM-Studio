
import traceback
from dotenv import load_dotenv
import os
from common_modules.all_in_one_module import print,os,sys,LLMSelector, ModelConfigurator, model_choices
import pdb
from IPython.display import display, Markdown

# configurator = ModelConfigurator("mistral")
configurator = ModelConfigurator("GROQ")
configurator.configure()

from rich.console import Console
from rich.markdown import Markdown

# Your Markdown content
markdown_text = """
## Heading Level 2

- Bullet point 1
- Bullet point 2

**Bold text** and _italic text_.

`Code snippet`
"""



try:
    client = OpenAI(base_url=os.environ.get("OPENAI_API_BASE"), api_key=os.environ.get("OPENAI_API_KEY"))
    # output =1/0
    # print(output)
except Exception as ex:
    output = configurator.check_model_connection(f""" here is the instruction form for response in python code and display messages in sequential number as follows :
                                        1) identify the line number of code where the error is occurring and provide a link to file to open and include "#L" linenumber, 
                                            eg:  1) The error is occurring at line 14 of the file "/Users/pratiksingh/repo/projects/LM-Studio/main.py#L14"
                                        2) mention one line error message
                                        3) show one line solution 
                                        4) if any packages needs to be install give a command to install as well as script to import the necessary packages.
                                        5) show the necessary code in clear and readable code blocks with proper color highlights in markdown file format
                                        for the error message:  {traceback.format_exc()}""")

    console = Console()
    markdown = Markdown(output)
    console.print(markdown)
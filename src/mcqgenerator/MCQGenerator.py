import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# importing necessary packages packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from openai import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# MODEL = "llama2:13b"
# MODEL = "gemma:instruct"
MODEL = "gemma:7b"
llm = ChatOllama(model=MODEL)
# model = OllamaFunctions(model=MODEL)
# from openai import OpenAI

# client = OpenAI(
#     base_url='http://localhost:11434/v1',
#     api_key='ollama'
# )

# response = client.chat.completions.create(
#   model="gemma:instruct",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The LA Dodgers won in 2020."},
#     {"role": "user", "content": "Where was it played?"}
#   ]
# )
# print(response.choices[0].message.content)

# # Load environment variables from the .env file
# load_dotenv()

# Access the environment variables just like you would with os.environ
# key = os.getenv("OPENAI_API_KEY")

# print("Value of MY_VARIABLE:", key)

# llm1 = ChatOpenAI(openai_api_key=key,model_name="gpt-3.5-turbo", temperature=0.3)
# llm = OllamaFunctions(model=MODEL, temperature=0)
# llm2 = OpenAI(
#     base_url='http://localhost:11434/v1',
#     api_key=MODEL
# )

template = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to
create a quiz of {number} multiple choice questions for {subject} in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below. Ensure to make {number} MCQs in given JSON format below
and do not add any extra text apart from pure JSON format.
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "grade", "tone", "response_json"],
    template=template)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

template2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.
You need to evaluate the complexity of teh question and give a complete analysis of the quiz if the students
will be able to understand the questions and answer them. Only use at max 50 words for complexity analysis.
if the quiz is not at par with the cognitive and analytical abilities of the students,
update tech quiz questions which needs to be changed  and change the tone such that it perfectly fits the
student abilities.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=template2)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

# This is an Overall Chain where we run the two chains in Sequence
generate_evaluate_chain = SequentialChain(chains=[quiz_chain, review_chain],
                                          input_variables=["text", "number", "subject", "tone", "response_json"],
                                          output_variables=["quiz", "review"], verbose=True,)


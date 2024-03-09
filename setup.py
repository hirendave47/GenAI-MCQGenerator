from setuptools import find_packages, setup

setup(
    name='mcqgenerator',
    version='1.0.0',
    author='Hiren Dave',
    author_email='hirendave47@gmail.com',
    url='https://github.com/HirenDave/mcqgenerator',
    description='Generate MCQs for a given question',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages(),
)

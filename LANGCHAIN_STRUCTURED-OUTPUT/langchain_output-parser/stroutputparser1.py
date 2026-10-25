from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ------ With stringOutputParser ------ #

model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()  # If we use this, it directly gives content attribute section of the result(LLM response) object. Which we need in order to form a chain, so that 1st llm result response directly can go for input of the second prompt2. Since chain is an important component and that is why we often use this StrOutputParser nothing else. It just removes metadata from llm response and keep only content attribute

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result) # Not result.content, because at the end parse removed the metadata from the response


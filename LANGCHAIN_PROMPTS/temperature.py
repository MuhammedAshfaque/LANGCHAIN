from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
# same input key liye different output depends on temperature. That's the basic funda of temperature. If you want to have different outputs for the same input, you can increase the temperature. The higher the temperature, the more random the output will be.
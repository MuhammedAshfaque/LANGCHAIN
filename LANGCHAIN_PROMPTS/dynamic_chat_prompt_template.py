from langchain_core.prompts import ChatPromptTemplate

# We do not use packages like AIMessage, SystemMessage, HumanMessage in order to form dynamic prompt with chat history. Instead, we create a format of chat like this.
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)
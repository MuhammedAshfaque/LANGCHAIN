from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# In order to format our input with complete chat history. We use MessagePlaceHolder package to inject chat_history in between.

chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []
# Retrieve the chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print(prompt)
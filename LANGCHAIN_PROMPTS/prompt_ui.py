from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatOpenAI()

st.header('Reasearch Tool')

# user_input = st.text_input("Enter you query here")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# template = PromptTemplate(
#     template="Summarize the research paper '{paper_input}' in a {style_input} style and provide a {length_input} explanation.",
#     input_variables=["paper_input", "style_input", "length_input"],
#     validate_template=True  # We can validate the prompt whether all the dynamic input has come or not, it will through error. That is why we use PromptTemplate library instead of f string in python
# )

template = load_prompt('template.json')

# prompt = template.invoke({
#     'paper_input': paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    # result = model.invoke(user_input)
    
    # result = model.invoke(prompt)
    st.write(result.content)
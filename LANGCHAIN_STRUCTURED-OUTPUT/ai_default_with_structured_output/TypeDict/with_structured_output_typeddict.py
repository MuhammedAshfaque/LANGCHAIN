from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):
    summary: str
    sentiment: str
    
    ### If we write only summary, sentiment. LLM might not understad things better even though system attaches extra prompts. So in order to guide more, we use Annotation, our LLM will get surity what actually i have to generate.
                            
    # summary: Annotated[str, "A brief summary of the review"]
    # sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]

    # we can also ask for list to LLM from the prompt review.
    # key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    # pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    # cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    # name: Annotated[Optional[str], "Write the name of the reviewer"]
    # sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]   # --> literal means ans will come in either options available  

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

print(result)
print(result['summary'])
print(result['sentiment'])
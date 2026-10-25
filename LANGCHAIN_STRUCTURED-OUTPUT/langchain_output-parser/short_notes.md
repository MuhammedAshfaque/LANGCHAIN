#### Output Parsers: In LangChain help to convert raw LLM responses into structured formats like JSON, CSV, Pydantic models, and more. They ensure consistency, validation, and ease of use in applications. Output parser not only can work with open source who doesn't provide structured output by default. It can also work with those closed source AI model who provides structured output..
### There are so many types of output parser. We will learn some importants :
1. String Output Parser : Takes LLM output and simply return it as string.
2. Json Output Parser : We use it when we want LLM response(output) in json format. It doesn't enforce a particular schema. You can't ask llm to response in a particular schema. This is the drawback here.
3. Structured Output Parser : It is an output parser in langchain which helps to extract structured JSON data from LLM responses based on predefined field schemas. Which we can't do using normal JSON output parser. But we can't do data validation here. It only can enforce LLM to give response in a specific format.
4. Pydantic Output Parser : 
    
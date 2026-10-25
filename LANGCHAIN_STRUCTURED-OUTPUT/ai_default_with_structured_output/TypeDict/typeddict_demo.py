from typing import TypedDict

# Structured output means we force LLM to give structured output.
# TypedDict helps to ask LLM giving output into a proper format, In order to specify a particular format we use TypeDict
class Person(TypedDict) :
    name: str
    age: int

new_person: Person = {'name': "Ashfaque", 'age': 22}

print(new_person)
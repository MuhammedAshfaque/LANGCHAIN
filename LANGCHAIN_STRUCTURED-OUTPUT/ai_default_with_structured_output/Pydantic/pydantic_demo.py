from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# When we want output from LLM into a specific format, which we define through TypeDict. But there is an issue, we can't validate whether LLM following our guideline or not. In order to validate the output, we use Pydantic.

class Student(BaseModel):
    name: str = 'nitish'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student') # Field acts just like Annotated package in TypeDict. Provides additional prompt to the LLM.


new_student = {'age':'32', 'email':'abc@gmail.com'}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()
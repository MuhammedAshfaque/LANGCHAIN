# Runnable
## I Why Runnables Exist in LangChain

### 1. The starting point: many separate components

Building an LLM application, such as a PDF-based Q&A (RAG) app, requires several components:

- **Document loader** to load the PDF
- **Text splitter** to break it into chunks
- **Embeddings and vector store** to index the chunks
- **Retriever** for semantic search
- **LLM** to generate the answer

Each component is useful, but wiring them together by hand is complex.

### 2. The repeated-work problem

The LangChain team noticed that developers were repeating the same sequence of steps in almost every LLM application.

### 3. First solution: Chains

The idea was to combine commonly repeated steps into a single class or function. You call it once, and the steps run behind the scenes.

> Example: `RetrievalQA` and `LLMChain` bundle retrieval, prompting and generation.

The team then built more chains for different purposes.

### 4. The new problem: chain explosion

- The codebase became **messy and bloated**.
- Developers found it **hard to learn and remember** so many chains.

### 5. Root cause

So many chains were needed because **each component had a different interface and method signature**. A loader used `load()`, a retriever used `get_relevant_documents()`, an LLM used `predict()`, and so on. Components could not be connected directly, so a custom chain was needed to glue each combination together.

### 6. Final solution: Runnables

LangChain introduced a **common interface** that all components follow, called a **Runnable**.

- Every component exposes the same standard methods: `invoke()`, `batch()` and `stream()` (plus async versions).
- Because the interface is uniform, components can be **connected like building blocks**, for example with the `|` pipe operator (LCEL, the LangChain Expression Language).
- There is **no need for a separate chain for each task**. You compose Runnables instead.

```python
chain = prompt | llm | output_parser
chain.invoke({"question": "What is this PDF about?"})
```

### 7. Summary

| Stage | Approach | Problem / Benefit |
|---|---|---|
| Before | Components with different interfaces | Hard to connect, lots of glue code |
| Chains | Prebuilt classes bundling common tasks | Too many chains, messy codebase |
| Runnables | One common interface for all components | Easy composition, less code, easier to learn |

**In one line:** Runnables exist to standardize the interface of all components so they can be composed freely, which removes the need for a separate chain for every task.

# II. What Is a Runnable?

## 1. Definition

A **Runnable** is a unit of work in LangChain. It takes an input, processes it, and returns an output.

Prompts, LLMs, retrievers, output parsers and even plain Python functions (wrapped as `RunnableLambda`) are all Runnables.

```
input  -->  [ Runnable ]  -->  output
```

## 2. The common interface

Every Runnable follows the same set of standard methods.

| Method | What it does | Input / Output |
|---|---|---|
| `invoke()` | Runs the Runnable on **one** input | 1 input -> 1 output |
| `batch()` | Runs the Runnable on **multiple** inputs at once (in parallel) | list of inputs -> list of outputs |
| `stream()` | Returns the output **piece by piece** as it is produced, instead of waiting for the full result (for example, an LLM streaming tokens) | 1 input -> stream of chunks |

Each method also has an async version: `ainvoke()`, `abatch()` and `astream()`.

## 3. Composition: why the common interface matters

Because every Runnable has the same interface, the output of one can be fed directly into the next. This lets you **connect Runnables like building blocks** and build complex workflows from simple parts.

- The `|` pipe operator joins Runnables into a sequence (LCEL, the LangChain Expression Language).
- The combined result is itself a Runnable, so it also supports `invoke()`, `batch()` and `stream()`.
- The output format of one step must match what the next step expects as input.

```python
from langchain_core.runnables import RunnableLambda

add_one = RunnableLambda(lambda x: x + 1)
double = RunnableLambda(lambda x: x * 2)

add_one.invoke(3)           # 4
add_one.batch([1, 2, 3])    # [2, 3, 4]

chain = add_one | double    # a new Runnable
chain.invoke(3)             # (3 + 1) * 2 = 8
```

## 4. Summary

- A Runnable is a unit of work: **input -> process -> output**.
- All Runnables share `invoke()`, `batch()` and `stream()` (plus async versions).
- The shared interface lets you **chain Runnables together** to build complex tasks.

**In one line:** A Runnable is a standard, composable unit of work, and the common interface is what lets you connect many of them into one workflow.

# III. Types of Runnables

Runnables fall into two broad categories:

1. **Task-specific Runnables**: core LangChain components (prompts, LLMs, parsers, retrievers) that were converted to follow the Runnable interface.
2. **Runnable primitives**: building blocks that exist only to **connect and control** other Runnables.

```
Runnables
|
|-- Task-specific        -> ChatOpenAI, PromptTemplate, StrOutputParser, Retriever ...
|
|-- Primitives           -> RunnableSequence, RunnableParallel, RunnablePassthrough,
                            RunnableLambda, RunnableBranch
```

---

## 1. Task-specific Runnables

These are the components you already know. Each does one specific job, and each now has `invoke()`, `batch()` and `stream()`.

| Component | Job | Typical input -> output |
|---|---|---|
| Chat model / LLM (`ChatOpenAI`) | Generates text | prompt value or messages -> message |
| `PromptTemplate` | Fills a template with variables | dict -> prompt value |
| Output parser (`StrOutputParser`) | Converts model output into a usable format | message -> string / JSON / object |
| Retriever | Finds relevant documents | query string -> list of documents |
| Document loader, text splitter | Load and split data (partly, depending on version) | source -> documents |

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template("Explain {topic} in one line.")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

prompt.invoke({"topic": "RAG"})   # works on its own
llm.invoke("Hello")               # works on its own
```

---

## 2. Runnable primitives

Primitives do not do an AI task themselves. They decide **how other Runnables are connected**: in order, side by side, with a pass-through, or with a condition.

### 2.1 RunnableSequence: run steps one after another

The output of each step becomes the input of the next. This is the most common primitive, and the `|` operator creates it.

```python
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence(prompt, llm, parser)
# identical to:
chain = prompt | llm | parser

chain.invoke({"topic": "RAG"})
```

```
input -> [prompt] -> [llm] -> [parser] -> output
```

### 2.2 RunnableParallel: run steps side by side

All branches receive the **same input** and run at the same time. The result is a **dictionary** with one key per branch.

```python
from langchain_core.runnables import RunnableParallel

tweet_chain    = tweet_prompt | llm | parser
linkedin_chain = linkedin_prompt | llm | parser

parallel = RunnableParallel({
    "tweet": tweet_chain,
    "linkedin": linkedin_chain,
})

parallel.invoke({"topic": "AI"})
# {"tweet": "...", "linkedin": "..."}
```

```
              +--> [tweet chain] ----+
input --------|                      |--> {"tweet": ..., "linkedin": ...}
              +--> [linkedin chain] -+
```

### 2.3 RunnablePassthrough: return the input unchanged

It outputs exactly what it receives. It is useful when a later step needs the **original input** alongside a processed result.

```python
from langchain_core.runnables import RunnablePassthrough

RunnablePassthrough().invoke("hello")   # "hello"
```

Common use, keeping the original value next to a derived one:

```python
parallel = RunnableParallel({
    "original": RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split())),
})

parallel.invoke("LangChain makes LLM apps easier")
# {"original": "LangChain makes LLM apps easier", "word_count": 5}
```

In RAG it is typically used to forward the user's question while the retriever fetches context: `{"context": retriever, "question": RunnablePassthrough()}`.

### 2.4 RunnableLambda: turn any Python function into a Runnable

This lets your own custom logic (cleaning text, counting words, formatting) sit inside a chain.

```python
from langchain_core.runnables import RunnableLambda

def word_count(text: str) -> int:
    return len(text.split())

counter = RunnableLambda(word_count)

counter.invoke("Runnables are composable")   # 3
chain = prompt | llm | parser | counter       # works inside a chain too
```

### 2.5 RunnableBranch: choose a path with a condition

You give it `(condition, runnable)` pairs and a **default** Runnable as the last argument. It runs the first branch whose condition is true; if none match, it runs the default.

```python
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

branch = RunnableBranch(
    (lambda x: len(x.split()) > 300, summarize_chain),   # if long, summarize
    RunnablePassthrough(),                               # otherwise, keep as is
)

branch.invoke(long_report)
```

```
input --> condition 1 true? --> branch 1
          else condition 2 true? --> branch 2
          else --> default
```

---

## 3. Shortcuts worth knowing

Inside a `|` chain, LangChain converts some plain Python objects automatically:

| You write | LangChain treats it as |
|---|---|
| a `dict` of Runnables | `RunnableParallel` |
| a plain function | `RunnableLambda` |

```python
chain = prompt | llm | parser | {"answer": RunnablePassthrough(), "length": len}
```

---

## 4. Other Runnables you may meet

| Runnable / helper | Purpose |
|---|---|
| `RunnablePassthrough.assign(...)` | Pass the input through and **add new keys** to it |
| `.with_retry()` / `.with_fallbacks()` | Retry on failure, or switch to a backup Runnable |
| `.bind()` / `.with_config()` | Attach fixed arguments or settings (for example `stop` words, tags) |
| `RunnableWithMessageHistory` | Add chat memory to a Runnable |
| `RunnableGenerator` | Wrap a generator function for streaming transformations |

---

## 5. Summary

| Type | Role | Output shape |
|---|---|---|
| Task-specific | Does one AI task (prompt, LLM, parse, retrieve) | Depends on the component |
| `RunnableSequence` | Runs steps **in order** | Output of the last step |
| `RunnableParallel` | Runs steps **at the same time** on the same input | Dictionary |
| `RunnablePassthrough` | Returns the input **unchanged** | Same as input |
| `RunnableLambda` | Wraps a **Python function** | Whatever the function returns |
| `RunnableBranch` | Picks a path by **condition** | Output of the chosen branch |

**In one line:** Task-specific Runnables do the work, and primitives decide how that work is connected: in sequence, in parallel, passed through, wrapped from a function, or chosen by condition.
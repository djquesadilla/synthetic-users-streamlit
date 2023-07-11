import streamlit as st
from langchain.llms.openai import OpenAI
from llama_index import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.indices.postprocessor import (SentenceEmbeddingOptimizer,
                                               SimilarityPostprocessor)

from dotenv import load_dotenv

load_dotenv('.env')

### LOAD DATA ###
synthetic_user_file = open("./data/summary-example/synthetic-user.txt", "r")
synthetic_user = synthetic_user_file.read()
synthetic_user_file.close()

problem_file = open("./data/summary-example/problems.txt", "r")
problem = problem_file.read()
problem_file.close()

solution_file = open("./data/summary-example/solution.txt", "r")
solution = solution_file.read()
solution_file.close()

original_user_interviews = []
for i in range(1, 10):
    user_interview_file = open("./data/summary-example/user-interview-" + str(i) + ".txt", "r")
    user_interview = user_interview_file.read()
    user_interview_file.close()
    original_user_interviews.append(user_interview)

prompt_file = open("./data/summary-example/prompt.txt", "r")
original_prompt = prompt_file.read()
prompt_file.close()

# PROMPT DEFINED
prompt_embeddings_file = open("./data/summary-example/prompt-embeddings.txt", "r")
prompt_embeddings = prompt_embeddings_file.read()
prompt_embeddings_file.close()

prompt_embeddings = prompt_embeddings.format(problem=problem, solution=solution)

# Define a simple Streamlit app
st.title("Ask Synthetic Users to Report")
query = st.selectbox(options=[
    "refine",
    "compact",
    "tree_summarize",
    "simple_summarize",
    "generation",
    "no_text",
    "accumulate",
    "compact",
    "compact_accumulate"
    ], label="Select your Summarization strategy", index=0)

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        response_synthesizer = get_response_synthesizer(
            response_mode=query,
        )
        # assemble query engine
        documents = [Document(text=user_interview, metadata={"type": "user_interview"}) for user_interview in original_user_interviews]
        index = VectorStoreIndex.from_documents(documents=documents)
        
        query_engine = index.as_query_engine(
        similarity_top_k=3,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7),
                SentenceEmbeddingOptimizer(percentile_cutoff=0.5)
            ]
        )
        
        st.success(query_engine.query(prompt_embeddings))
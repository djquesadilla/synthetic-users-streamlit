import streamlit as st
from langchain.llms.openai import OpenAI
from llama_index import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.indices.postprocessor import (SentenceEmbeddingOptimizer,
                                               SimilarityPostprocessor)

from dotenv import load_dotenv

from llama_parameters import summarization_strategies

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

# Define a simple Streamlit app layout
st.title("Synthetic Users Report Testing Playground")
json_file = st.file_uploader('File uploader')
summarization_strategy = st.selectbox(options=summarization_strategies, label="Select summarization strategy", index=0)
prompt = st.text_area(
    label="Write the prompt",
    value="Please summarize the user interviews for the scenario with the following problem and solution:\n- {problems}\n- {solution}",
    height=200,
    max_chars=1000
)

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not summarization_strategy.strip():
        st.error(f"Please provide the summarization strategy.")
    elif not prompt.strip():
        st.error(f"Please provide the prompt.")
    elif json_file is None:
        st.error(f"Please provide the JSON file.")
    else:
        # response_synthesizer = get_response_synthesizer(
        #     response_mode=summarization_strategy,
        # )
        # # assemble query engine
        # documents = [Document(text=user_interview, metadata={"type": "user_interview"}) for user_interview in original_user_interviews]
        # index = VectorStoreIndex.from_documents(documents=documents)
        
        # query_engine = index.as_query_engine(
        # similarity_top_k=3,
        #     response_synthesizer=response_synthesizer,
        #     node_postprocessors=[
        #         SimilarityPostprocessor(similarity_cutoff=0.7),
        #         SentenceEmbeddingOptimizer(percentile_cutoff=0.5)
        #     ]
        # )
        
        # st.success(query_engine.query(prompt_embeddings))
        problems = ",".join(["one problem", "two problems"])
        solution = "one solution"
        st.success(prompt.format(problems=problems, solution=solution))
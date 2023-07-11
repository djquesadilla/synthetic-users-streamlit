import streamlit as st
from langchain.llms.openai import OpenAI
from llama_index import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.indices.postprocessor import (SentenceEmbeddingOptimizer,
                                               SimilarityPostprocessor)

from dotenv import load_dotenv

from llama_parameters import summarization_strategies
from data_processing import extract_json_data_to_index, index_user_interviews

load_dotenv('.env')

# Define a simple Streamlit app layout
st.title("Synthetic Users Report Testing Playground")
json_file = st.file_uploader('File uploader')
summarization_strategy = st.selectbox(options=summarization_strategies, label="Select summarization strategy", index=0)
prompt = st.text_area(
    label="Write the prompt",
    value="Please summarize the user interviews for the scenario with the following problem and solution:\n- {problems}\n- {solution}",
    height=200,
    max_chars=4000
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
        data = extract_json_data_to_index(json_file)
        st.text("Generating prompt embeddings for " + str(len(data["User Interviews"])) + " user interviews...")
        
        prompt = prompt.format(problems=data["Problems"], solution=data["Solution"])
        st.text("Prompt:\n" + prompt)
        st.text("Prompting with summarization strategy " + summarization_strategy + " ...")
        
        index = index_user_interviews(data)
        
        response_synthesizer = get_response_synthesizer(
            response_mode=summarization_strategy,
        )

        # assemble query engine
        query_engine = index.as_query_engine(
        similarity_top_k=3,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.7),
                SentenceEmbeddingOptimizer(percentile_cutoff=0.5)
            ]
        )
        
        st.success(query_engine.query(prompt))
        
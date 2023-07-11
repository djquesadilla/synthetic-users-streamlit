import pandas as pd
import streamlit as st
from langchain.llms.openai import OpenAI


def extract_json_data_to_index(json_file):
    if json_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_json(json_file)
        st.write("Data preview:")
        st.write(dataframe)

        # Extract the specific data
        result_dict = {
                "Synthetic user": "",
                "Problems": "",
                "Solution": "",
                "User Interviews": []
            }
        for i in range(len(dataframe)):
            data = dataframe.iloc[i]
            result_dict["Synthetic user"] = data["Synthetic user"]
            result_dict["Problems"] = data["Problems"]
            result_dict["Solution"] = data["Solution"]
            result_dict["User Interviews"].append("\n".join([data[f"Question {j}"] for j in range(1, 11)]))
        
        return result_dict
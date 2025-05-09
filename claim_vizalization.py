import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Patent Claim Vizualization", layout="wide")

st.title("Patent Claim Vizualization")

# Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file (.xlsx) with 'sentence' and 'class' columns", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        if 'sentence' not in df.columns or 'pred_class' not in df.columns:
            st.error("Error: Your file must have 'sentence' and 'pred_class' columns.")
        else:
            st.success("File uploaded successfully!")

            # Assign unique colors to each class
            color_map = {'FUN': 'red', 'STR': 'blue', 'MIX': 'green', 'OTH': 'grey'}

            st.subheader("Sentences")

            # Display each sentence with background color based on class
            for _, row in df.iterrows():
                sentence = row['sentence']
                class_label = row['pred_class']
                color = color_map[class_label]

                st.markdown(
                    f"<div style='background-color:{color}; padding:10px; border-radius:8px; margin-bottom:5px;'>"
                    f"{sentence}</div>",
                    unsafe_allow_html=True
                )

    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")

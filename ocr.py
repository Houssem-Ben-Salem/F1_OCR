import streamlit as st
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import numpy as np
import re

# Ensure the value starts with a '+'
def ensure_plus_prefix(val):
    return val if val.startswith('+') else '+' + val

def process_image(image_bytes):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # use_angle_cls for detecting text orientation

    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = ocr.ocr(image, cls=True)
    result = result[0]  # Assuming the first result is the one we're interested in
    high_confidence_lines = [text.strip() for _, (text, confidence) in result if confidence > 0.8]

    plain_text = "\n".join(high_confidence_lines)
    return plain_text, image

def create_csv(plain_text):
    lines = plain_text.strip().split("\n")
    competition_name = lines[0].replace(" ", "_").replace("FORMULA1_", "")
    points_pattern = re.compile(r"^\+\d+$|^\d+\+$")

    driver_blocks = []
    for i, line in enumerate(lines):
        if points_pattern.search(line) and i >= 3:
            block = lines[i-3:i+1]
            driver_blocks.append(block)

    data = [{
        "Driver": block[0],
        "Car": block[1],
        "Time": ensure_plus_prefix(block[2]),
        "Points": ensure_plus_prefix(block[3])
    } for block in driver_blocks]

    df = pd.DataFrame(data)
    return df, competition_name

# Streamlit App
st.title('Formula One OCR Data Extraction')

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    plain_text, image = process_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Generate and display the data
    df, competition_name = create_csv(plain_text)
    csv_filename = f"{competition_name}.csv"
    
    # Add a 'Rank' column that starts from 1
    df.index = np.arange(1, len(df) + 1)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Rank'}, inplace=True)

    st.write("Extracted Data Table:")
    st.dataframe(df)
    st.download_button(
            label="Download data as CSV",
            data=df.to_csv().encode('utf-8'),
            file_name=csv_filename,
            mime='text/csv',
        )

st.markdown("""
This Streamlit app uses PaddleOCR to process uploaded Formula One images, extract text data,
and display it in a structured table format.
""")
import streamlit as st
import fitz  # PyMuPDF
import openai
from fpdf import FPDF
import zipfile
import os
import re
import pdfkit
import markdown
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import tempfile
from io import BytesIO
import base64


load_dotenv()  # Load environment variables from .env file
openai.api_key = st.secrets['OPENAI_API_KEY']

def extract_name_from_response(response_content):    
    prompt = f"""Extract the field applicants name from the following text:\n\n{response_content}.
                Return name without title. For example, return 'Maxwell Singer' not 'Name: Maxwell Singer'"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", "content":  "You are an assistant who extracts data from text.",
                "role": "user", "content": prompt},
        ],
        temperature=0.0
        )
    name = response.choices[0].message.content
    return name

def sanitize_filename(name):
    return re.sub(r'[^\w\s-]', '', name).replace(' ', '_')

def extract_text_from_pdfs(pdf_files):
    extracted_texts = []
    for pdf in pdf_files:
        pdf_data = pdf.read()
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        extracted_texts.append(text)
    return extracted_texts

def query_gpt4(text, fields, template=None):
    try:
        with open('medical_school_rankings.txt', 'r') as file:
            medical_rankings = file.read()
        with open('college_rankings.txt', 'r') as file:
            college_rankings = file.read()
    except FileNotFoundError as e:
        st.error(f"Rankings file not found: {e}")
        return ""
    
    fields_to_extract = ", ".join(fields) if fields else "Name"
    
    user_content = f"""Extract the following information from this text: Name, {fields_to_extract}.
    (always extract name first even if not specified).\n\nText: {text}.
    When printing the information please be as simple as possible.
    For example for name the output should be 'Name: [name]'
    Bold all titles of extracted information such as 'Name'
    Provide an extra line space between each extracted section or unit (such as publication).
    If publications are requested, please format them as AMA common citation style arranged by date (most recent first) and unnumbered with a line space between
    each of them. Do not block all publications in a single paragraph, make sure there is a line space notated in between each seperate
    publication.
    For anything that is countable such as publications or presentations after the title of that section please display the number of
    units within that section (such as number of presentations).

    If personal statement is requested, please maintain paragraph formatting of original personal statement.
    Do not block personal statements in a single chunk of text without maintaining paragraph breaks.

    If asked about medical school rankings, use the list from the provided file titled "medical_school_rankings".
    Allow non-perfect close matches from the list. For example, if the medical school is "Pritzker" but the "medical_school_rankings"
    lists "20. University of Chicago (Pritzker)" consider those the same, and the medical school ranking 20.
    If it is not on the list, say "Unranked".

    If asked about college or university rankings, use the list from the provided file titled "college_rankings". If it is not on the list,
    say "Unranked". Allow non-perfect matches from the list as described above for medical schools.

    Please do the markdown formatting for this content.
    """
    
    if template:
        user_content += f"\n\nUse the following format as a template for all responses:\n{template}"

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", "content":  "You are an assistant who extracts specified information from text.",
                "role": "user", "content": user_content},
        ],
        temperature=0.0
        )
    
    message_content = response.choices[0].message.content
    return message_content

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Function to use GPT-4o to extract specific fields
def extract_fields_with_gpt(response_content, fields):
    field_values = {}
    for field in fields:
        prompt = f"""Extract the value for each '{field}' from the following text:\n\n{response_content}.
                For any field that has a numberical value after it in parenthesis, collect that information.
                For example, if field is "Published Papers (middle author, actually published)" and the
                source says "Published Papers (middle author, actually published) (6)" the response should be "6".
                Do not provide any sort of formatting in your response, just the value.
                For example for "Published Papers (middle author, actually published) (6)", the response should be "6" not "(6):**"
                Please do not return the field title in your response."""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", "content":  "You are an assistant who extracts data from text.",
                        "role": "user", "content": prompt},
                ],
                temperature=0.0
                )
            field_value = response.choices[0].message.content
            field_values[field] = field_value if field_value else "N/A"
        except Exception as e:
            field_values[field] = "N/A"
            print(f"Error extracting field '{field}': {e}")
    return field_values

# Function to create a summary PDF with statistics
def generate_summary_with_gpt(df):
    summary_prompt = f"""Please provide a well-organized and clear statistical summary of the following dataset in markdown format:\n\n{df.to_string()}\n\n
                         For numberical variables provide range and average.
                         For categorical variables that are yes or no (or any sort of binary) provide numbers of positive and number of negative.
                         Do not provide summary statisitcs for categorical variables that are non-binary.
                         For any part of this summary do not mention individuals.
                         Do not provide any notable patterns, insights, or conclusion. Just statistical data listed above.
                         Ensure the summary is formatted in a reader-friendly way, using bullet points, numbered lists, and clear headers as appropriate.
                         """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", "content":  "You are an assistant who creates statistical summary reports of residency applicants from data.",
                "role": "user", "content": summary_prompt},
        ],
        temperature=0.0
        )

    return response.choices[0].message.content

# Function to generate and save figures for numeric fields
def generate_figures(df):
    figure_files = []
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            figure_file = f"{sanitize_filename(column)}_distribution.png"
            plt.figure()
            df[column].plot(kind='hist', title=f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.savefig(figure_file)
            plt.close()
            figure_files.append(figure_file)
    return figure_files

# Function to create a summary PDF with statistics and figures
def create_summary_pdf_with_gpt(df, file_name):
    summary_text = generate_summary_with_gpt(df)

    # Convert the Markdown summary text to HTML
    html_content = markdown.markdown(summary_text)

    # Generate figures and add their HTML tags to the content
    figure_files = generate_figures(df)
    print(figure_files)
    figures_html = ""
    for figure_file in figure_files:
        figures_html += f'<div style="page-break-after: always;"><img src="{figure_file}" style="width: 100%; margin: 20px 0;" /></div>'
    print(figures_html)

    # Add inline CSS for better styling
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{
                font-size: 14px;
                font-family: 'DejaVu Sans', sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                font-size: 24px;
                text-align: center;
                margin-bottom: 20px;
            }}
            h2 {{
                font-size: 20px;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            p, li {{
                font-size: 14px;
                margin-bottom: 10px;
            }}
            ul {{
                list-style-type: disc;
                margin-left: 20px;
            }}
            ol {{
                list-style-type: decimal;
                margin-left: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .figure {{
                page-break-after: always;
                margin-top: 20px;
            }}
            .figure img {{
                width: 100%;
                border: 1px solid #ddd;
                padding: 10px;
                background-color: #f9f9f9;
            }}
        </style>
    </head>
    <body>
        <h1>Summary Statistics Generated</h1>
        {html_content}
        <div class="figure">
            {figures_html}
        </div>
    </body>
    </html>
    """

    # Define PDF options for pdfkit
    pdf_options = {
        'page-size': 'A4',
        'encoding': 'UTF-8',
        'margin-top': '10mm',
        'margin-right': '10mm',
        'margin-bottom': '10mm',
        'margin-left': '10mm'
    }

    # Convert the styled HTML to PDF using pdfkit
    try:
        pdfkit.from_string(styled_html, file_name, options=pdf_options)
    except Exception as e:
        print(f"Error creating summary PDF: {e}")

    # Clean up the generated figure files after including them in the PDF
    for figure_file in figure_files:
        if os.path.exists(figure_file):
            os.remove(figure_file)

def create_pdf(content, buffer):
    # Create an HTML template for the PDF content
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Generated PDF</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <pre>{}</pre>
    </body>
    </html>
    """.format(content)

    # Convert the HTML content to a base64-encoded data URL
    pdf_data = base64.b64encode(html_template.encode('utf-8')).decode('utf-8')
    pdf_data_url = f"data:application/pdf;base64,{pdf_data}"

    # Write the data URL to the provided buffer
    buffer.write(pdf_data_url.encode('utf-8'))

def create_zip(data_frame_file, summary_pdf_file, extracted_infos):
    zip_buffer = BytesIO()  # Use a BytesIO buffer to create the zip in memory

    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        # Create and add PDF files to the ZIP
        for extracted_info, response_content in extracted_infos:
            sanitized_name = sanitize_filename(extracted_info)
            pdf_buffer = BytesIO()  # Create a BytesIO buffer for each PDF
            
            # Generate the PDF data URL and write it to the buffer
            create_pdf(response_content, pdf_buffer)
            
            # Add the PDF buffer to the ZIP
            zipf.writestr(f"{sanitized_name}.pdf", pdf_buffer.getvalue())
            print(f"Added {sanitized_name}.pdf to zip")

        # Add the DataFrame CSV file
        if os.path.exists(data_frame_file):
            zipf.write(data_frame_file)
            print(f"Added {data_frame_file} to zip")
        
        # Add the summary PDF file
        if os.path.exists(summary_pdf_file):
            with open(summary_pdf_file, 'rb') as file:
                summary_pdf_data = file.read()
            zipf.writestr("summary_statistics.pdf", summary_pdf_data)
            print(f"Added summary_statistics.pdf to zip")

    zip_buffer.seek(0)  # Rewind the buffer to the beginning
    return zip_buffer.getvalue(), "Applications.zip"


def add_text_inputs():
    num_inputs = st.session_state.get('num_inputs', 6)
    for i in range(num_inputs):
        st.text_area(f"Field {i + 1}", key=f"input_{i}", height=200)

if 'num_inputs' not in st.session_state:
    st.session_state['num_inputs'] = 6

st.title("Residency Program Application Processor")

uploaded_files = st.file_uploader("Upload PDF applications", type="pdf", accept_multiple_files=True)

st.markdown(
    """
    <style>
    .stFileUploader {
        max-height: 600px;
        overflow-y: auto;
    }
    div[aria-busy="false"] > div:nth-of-type(2) {
        max-height: 400px;
        overflow-y: auto;
    }
    .stFileUploader {
        border: 2px solid #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Enter the information you want to extract in the fields below (such as GPA or Personal Statement):")

add_text_inputs()

if st.button("Add more fields"):
    st.session_state['num_inputs'] += 1
    st.experimental_rerun()

info_to_extract = [st.session_state[f"input_{i}"] for i in range(st.session_state['num_inputs']) if st.session_state[f"input_{i}"]]
info_to_extract = [info for info in info_to_extract if info]

# Simplify field names up to the first ")"
simplified_fields = [field.split(")")[0] + ")" if ")" in field else field for field in info_to_extract]

# Main PDF creation part
if st.button("Process Applications"):
    if uploaded_files and info_to_extract:
        with st.spinner("Processing applications..."):  # Add the spinner here

            # Combine the info_to_extract into a list of fields
            fields_to_extract = info_to_extract

            # Extract text from uploaded PDFs
            texts = extract_text_from_pdfs(uploaded_files)
            st.write("Text extraction complete.")

            # Use the first response as a template for the rest
            extracted_infos = []
            first_response = None
            data_list = []  # List to store data for the DataFrame

            for idx, text in enumerate(texts):
                if idx == 0:
                    first_response = query_gpt4(text, fields_to_extract)
                    response_content = first_response
                else:
                    response_content = query_gpt4(text, fields_to_extract, template=first_response)

                if response_content:
                    extracted_info = extract_name_from_response(response_content)
                    extracted_infos.append((extracted_info, response_content))

                    # Use extract_fields_with_gpt to get field values
                    extracted_fields = extract_fields_with_gpt(response_content, simplified_fields)
                    extracted_fields['Name'] = extracted_info  # Add the applicant's name
                    data_list.append(extracted_fields)

            # Create the DataFrame with simplified column names
            df = pd.DataFrame(data_list, columns=["Name"] + simplified_fields)

            # Save the DataFrame to a CSV file
            data_frame_file = "applicant_data.csv"
            df.to_csv(data_frame_file, index=False)

            st.write("DataFrame creation complete.")

            # Create a summary PDF with statistics
            summary_pdf_file = "summary_statistics.pdf"
            create_summary_pdf_with_gpt(df, summary_pdf_file)

            st.write("Summary statistics PDF creation complete.")

            # Create a zip file containing all the new PDFs, the DataFrame, and summary statistics
            zip_data, zip_filename = create_zip(data_frame_file, summary_pdf_file, extracted_infos)
            st.write("Zipping complete!")

            # Remove the temporary files after creating the ZIP
            if os.path.exists(data_frame_file):
                os.remove(data_frame_file)
            if os.path.exists(summary_pdf_file):
                os.remove(summary_pdf_file)

            # Provide a download link for the zip file
            st.download_button(
                label="Download ZIP",
                data=zip_data,
                file_name=zip_filename,
                mime="application/zip"
            )

    else:
        st.error("Please upload PDF files and specify the information to extract.")
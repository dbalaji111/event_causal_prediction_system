import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path, output_path):
    """
    Extracts text from each page of the PDF and saves it to a text file.
    
    Parameters:
        pdf_path (str): The path to the PDF file.
        output_path (str): The path to save the output text file.
    """
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    text = ""
    
    # Loop through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        page_text = page.get_text("text")  # Extract text from the page
        text += f"--- Page {page_num + 1} ---\n{page_text}\n"
    
    # Save text to file
    with open(output_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)

def process_pdf_folder(pdf_folder, output_folder):
    """
    Processes each PDF file in a folder, extracting text and saving it as a .txt file.
    
    Parameters:
        pdf_folder (str): Path to the folder containing PDF files.
        output_folder (str): Path to the folder where .txt files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all PDF files in the folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            txt_filename = f"{os.path.splitext(filename)[0]}.txt"
            output_path = os.path.join(output_folder, txt_filename)
            
            # Extract text and save as .txt file
            extract_text_from_pdf(pdf_path, output_path)
            print(f"Extracted text from {filename} to {txt_filename}")

# Example usage
pdf_folder = r"C:\Users\balaj\code_files\Documents\Brahmanda\context_aware_risk_methodology\Gold_WSJ_Data"
output_folder = r"C:\Users\balaj\code_files\Documents\Brahmanda\context_aware_risk_methodology\event_causal_prediction_system\scripts\pdf_to_csv_output"
process_pdf_folder(pdf_folder, output_folder)

import os
import json
import pandas as pd
import onnxruntime as rt
from transformers import AutoTokenizer
import shutil
from PyPDF2 import PdfReader
import csv

def load_encoded_category_types(json_path):
    with open(json_path, 'r') as json_file:
        return json.load(json_file)

def load_model_and_tokenizer(model_path):
    inf_session = rt.InferenceSession(model_path)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    return inf_session, tokenizer

encoded_category_types = load_encoded_category_types("category_types_encoded2.json")
model, tokenizer = load_model_and_tokenizer("resume-classifier-quantized2.onnx")

output_dir = "categorized_resumes"
csv_path = "categorized_resumes.csv"

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from PDF '{pdf_path}': {e}")
        return ""

def categorize_resumes(input_dir, output_dir, tokenizer, encoded_category_types):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categorized_resumes = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            resume_path = os.path.join(input_dir, filename)
            
            try:
                resume_text = extract_text_from_pdf(resume_path)  # Extract text from PDF

                # Tokenize the resume text with a specified maximum sequence length
                input_ids = tokenizer(resume_text, return_tensors='pt', max_length=500, truncation=True, padding='max_length')['input_ids']

                probs = model.run(None, {'input_ids': input_ids.cpu().numpy()})[0][0]

                pred_class_labels = []
                for idx, prob in enumerate(probs):
                    if prob >= 0.01:
                        pred_class_labels.append(list(encoded_category_types.keys())[idx])

                if pred_class_labels:
                    for class_label in pred_class_labels:
                        category_dir = os.path.join(output_dir, class_label)
                        if not os.path.exists(category_dir):
                            os.makedirs(category_dir)

                        new_resume_filename = os.path.basename(filename)
                        new_resume_filename = new_resume_filename.replace(' ', '_') 
                        new_resume_path = os.path.join(category_dir, new_resume_filename)

                        try:
                            shutil.copyfile(resume_path, new_resume_path)  
                            categorized_resumes.append({'filename': new_resume_filename, 'category': class_label})
                        except Exception as e:
                            print(f"Error copying file '{new_resume_path}': {e}")
                            continue  

            except Exception as e:
                print(f"Error processing PDF file '{resume_path}': {e}")
                continue  

    return categorized_resumes   

def write_to_csv(data, csv_path):
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
def read_from_csv(csv_path):
    file_category = {}
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row['filename']
            category = row['category']
            file_category[filename] = category
    return file_category
    
def categorize(input_dir):
    categorized_resumes = categorize_resumes(input_dir, output_dir, tokenizer, encoded_category_types)
    write_to_csv(categorized_resumes, csv_path)
    dict = read_from_csv(csv_path)
    return dict

print(categorize("C:/Users/debje/Downloads/Resume-Classifier-IC-main/sample_input_directory"))
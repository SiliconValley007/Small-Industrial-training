import spacy
import PyPDF2
import warnings

warnings.filterwarnings('ignore')

model = spacy.load("nlp_model")

def _extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
def parseResume(resume_path):
    resume_text = _extract_text_from_pdf(resume_path)
    doc = model(resume_text)

    response_text = []
    for ent in doc.ents:
        response_text.append(f"{ent.label_.upper()}- {ent.text}")
    return ', '.join(response_text)

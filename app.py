from flask import Flask, render_template, request

# import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# import csv
import os
# import requests
# import json
# from categorize_resume import categorize
from resume_parser import parseResume
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

app = Flask(__name__)
# CHECK_RESUME_URL = "http://localhost:8001/check-resume"

# Load spaCy NER model
# nlp = spacy.load("en_core_web_sm")

similarity_scores = []
filenames = []


# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text


# Extract entities using spaCy NER
def extract_entities(text):
    emails = re.findall(r"\S+@\S+", text)
    names = re.findall(r"^([A-Z][a-z]+)\s+([A-Z][a-z]+)", text)
    if names:
        names = [" ".join(names[0])]
    return emails, names

def plot_and_save_cosine_similarity_dist(similarity_scores):
    # Plotting the histogram of similarity scores
    plt.figure(figsize=(8, 6))
    plt.hist(similarity_scores, bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution between Job Description and Resumes')
    plt.grid(True)
    plt.savefig("/templates/plots/cosine_similarity_dist.png")
    plt.close()
    
def plot_bar_plot(filenames, similarity_scores, n):
    # Plotting the bar chart for top N resumes vs. similarity score
    plt.figure(figsize=(10, 6))
    plt.barh(filenames[:n], similarity_scores[:n], color='orange')
    plt.xlabel('Similarity Score')
    plt.ylabel('Resumes')
    plt.title(f'Top {n} Resumes vs. Similarity Scores')
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.savefig("bar_plot.png")
    plt.close()
    
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=450, background_color='white').generate(text)
    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis labels
    plt.title('Word Cloud for Job Description')
    plt.savefig("word_cloud.png")
    plt.close()
    
def generate_similarity_matrix(cosine_matrix):
    # Display similarity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.xlabel('Resumes')
    plt.ylabel('Job Descriptions')
    plt.title('Cosine Similarity Matrix between Job Descriptions and Resumes')
    plt.savefig("similarity_matrix.png")
    plt.close() 


@app.route("/", methods=["GET", "POST"])
def index():
    global results
    results = []
    if request.method == "POST":
        similarity_scores.clear()
        filenames.clear()
        # job_description = request.form['job_description']
        job_description_files = request.files.getlist("job_description_files")
        resume_files = request.files.getlist("resume_files")

        # Create a directory for uploads if it doesn't exist
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        if not os.path.exists("uploads/JDs"):
            os.makedirs("uploads/JDs")

        if not os.path.exists("uploads/Resumes"):
            os.makedirs("uploads/Resumes")

        # # save jd
        # jd_path = os.path.join("uploads/JDs/", job_description_file.filename)
        # job_description_file.save(jd_path)

        # # Process saved jd
        # job_description = extract_text_from_pdf(jd_path)
        # # generate_word_cloud(job_description)
        
        processed_jds = []
        for jd in job_description_files:
            #Save uploaded JD
            filename = jd.filename
            jd_path = os.path.join("uploads/Resumes/", filename)
            jd.save(jd_path)
            # Process the saved file
            jd_text = extract_text_from_pdf(jd_path)
            processed_jds.append((jd_text, filename))

        # Process uploaded resumes
        processed_resumes = []
        for resume_file in resume_files:
            # Save the uploaded file
            resume_path = os.path.join("uploads/Resumes/", resume_file.filename)
            resume_file.save(resume_path)

            # params = {"file": open(resume_path, "rb"), "desc": job_description}
            # headers = {
            #     "Accept": "multipart/form-data",
            #     # "Content-Type": "application/pdf",
            # }
            # response = requests.post(CHECK_RESUME_URL, files=params, headers=headers)
            # response = response.json()
            # # Parse the JSON response
            # # parsed_response = json.loads(response.json())
            # print(response)

            # # Extract the values for "unMatchedKeywords" and "matchPercentage"
            # unmatched_keywords = response['unMatchedKeywords']
            # match_percentage = response['matchPercentage']

            # Process the saved file
            resume_text = extract_text_from_pdf(resume_path)
            emails, names = extract_entities(resume_text)
            processed_resumes.append((names, emails, resume_text, resume_file.filename))
            
        # dict = categorize("./uploads/Resumes/")
        # print(dict)
        
        final_result = []
        for jd_text, jd_filename in processed_jds:
            # TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer()
            job_desc_vector = tfidf_vectorizer.fit_transform([jd_text])

            # Rank resumes based on similarity
            ranked_resumes = []
            for names, emails, resume_text, filename in processed_resumes:
                resume_vector = tfidf_vectorizer.transform([resume_text])
                similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
                resume_path = os.path.join("uploads/Resumes/", filename)
                parsed_data = parseResume(resume_path)
                ranked_resumes.append((names, emails, similarity, filename, parsed_data))

            # Sort resumes by similarity score
            ranked_resumes.sort(key=lambda x: x[2], reverse=True)
            
            formatted_resumes = []
            for resume in ranked_resumes:
                formatted_resumes.append((resume[0], resume[1], "{:.2f}%".format(resume[2]), resume[3], resume[4]))
                similarity_scores.append(resume[2])
                filenames.append(resume[3])
                
            # plot_and_save_cosine_similarity_dist(similarity_scores=similarity_scores)
            # plot_bar_plot(filenames=filenames, similarity_scores=similarity_scores, n=3)

            final_result.append((jd_filename, formatted_resumes))
        results = final_result

    return render_template('index.html', results=results)


from flask import send_file


@app.route("/download_csv")
def download_csv():
    # Generate the CSV content
    csv_content = "Rank,Name,Email,Similarity\n"
    for rank, (names, emails, similarity) in enumerate(results, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        csv_content += f"{rank},{name},{email},{similarity}\n"

    # Create a temporary file to store the CSV content
    csv_filename = "ranked_resumes.csv"
    with open(csv_filename, "w") as csv_file:
        csv_file.write(csv_content)

    # Send the file for download
    csv_full_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), csv_filename
    )
    return send_file(
        csv_full_path, as_attachment=True, download_name="ranked_resumes.csv"
    )


if __name__ == "__main__":
    app.run(debug=True)

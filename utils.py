import spacy
import re

def _remove_stop_words(string):
    # Load the spaCy English language model
    nlp = spacy.load('en_core_web_lg')
    
    # Tokenize the string into individual words
    doc = nlp(string)
    
    # Filter out stop words
    filtered_words = [token.text for token in doc if not token.is_stop]
    
    # Join the filtered words back into a string
    new_string = ' '.join(filtered_words)
    
    return new_string

#Data Preprocessing
def cleanResume(resumeText):
    resumeText = _remove_stop_words(resumeText)
    resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText) 
    resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
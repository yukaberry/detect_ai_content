# Core Libraries: 
## Text and NLP

	1.	NLTK (nltk):
	•	Purpose: The Natural Language Toolkit (NLTK) is used for various text-processing tasks, including tokenization, removing stop words, and sentence splitting.
	•	Example Use: When tokenizing text to calculate word counts, NLTK’s word_tokenize and sent_tokenize functions help break down the text.
 
	2.	Stopwords (nltk.corpus.stopwords):
	•	Purpose: Stopwords are common words (like “the,” “and,” etc.) that are often filtered out in text analysis. We use NLTK’s corpus to easily access lists of stopwords.
	•	Example Use: To calculate lexical diversity, stopwords are removed before further analysis.
 
	3.	TextBlob (TextBlob):
	•	Purpose: TextBlob is a simpler NLP library that offers sentiment analysis, part-of-speech tagging, and noun phrase extraction.
	•	Example Use: Used to extract sentiment polarity and subjectivity, helping classify text as positive, neutral, or negative.
 
	4.	SpaCy (spacy):
	•	Purpose: SpaCy is a powerful NLP library known for its speed and efficiency in text processing tasks, including part-of-speech tagging, named entity recognition, and dependency parsing.
 	•	Purpose: TextStat calculates readability scores, such as the Flesch Reading Ease and SMOG index, to determine how difficult or easy a text is to read.
	•	Example Use: Calculating the readability of each text segment to assess its complexity level, which may be a distinguishing feature.
 
  5. TextStat (textstat):
	•	Purpose: TextStat calculates readability scores, such as the Flesch Reading Ease and SMOG index, to determine how difficult or easy a text is to read.
	•	Example Use: Calculating the readability of each text segment to assess its complexity level, which may be a distinguishing feature.

	6.	Transformers (transformers.pipeline):
	•	Purpose: Transformers by Hugging Face is used for pre-trained language models. The pipeline function provides easy access to tasks like summarization, translation, and more.
	•	Example Use: We could use the summarization pipeline to generate condensed versions of text, aiding in text comparison.

## Word Embeddings and Semantic Analysis

	7.	Sentence Transformers (SentenceTransformer, util):
	•	Purpose: Provides pre-trained models that can generate sentence embeddings, which are vector representations capturing the semantic meaning of text.
	•	Example Use: Embeddings are generated for each text segment, allowing for similarity comparisons and clustering of similar texts.

## Topic Modeling Libraries

	8.	Scikit-Learn (LatentDirichletAllocation, CountVectorizer):
	•	Purpose: Scikit-Learn offers various machine learning algorithms, including the Latent Dirichlet Allocation (LDA) for topic modeling and CountVectorizer for word counts.
	•	Example Use: LDA is used to find underlying topics in the text, and CountVectorizer helps by converting text data into a bag-of-words format for topic modeling.

# Functions

## Text Based 
**Explanation of Each Feature:**

1. **Word Count**: Calculates the total number of words in each text by tokenizing the text with word_tokenize(). This gives an insight into the length and verbosity of the text.

2. **Sentence Count**: Uses sent_tokenize() to count the number of sentences in each text. This helps understand the sentence structure and complexity.

3. **Average Word Length**: Computes the average length of words in each text. This provides information on the complexity of vocabulary used, longer 

4. **Stopwords Count**: Counts the number of stopwords in each text using the NLTK English stopwords list. Stopwords are words like “the”, “and", that dotn have much significant meaning alone.

**What Are Stopwords?**

**Stopwords** are common words in a language that often don’t add significant meaning or value to text analysis tasks, like “the,” “is,” “in,” and “and” in English. These words appear frequently but don’t usually contribute much to understanding the main content or meaning of a text, they are often removed in Natural Language Processing (NLP) tasks to improve processing efficiency and focus on more meaningful words.

**Purpose of Stopword Count**

Counting stopwords in a text can provide insights into the structure and formality of the text. For example:

•	A high stopword count might indicate that the text is more conversational or written in a natural, spoken style.
•	A low stopword count might suggest a more formal or technical text, as these typically avoid filler words..

**Example :**
```
text = "This is an example sentence showing stopword usage."
stopwords_count = sum([1 for word in word_tokenize(text) if word.lower() in stopwords.words('english')])
```
•Tokenization: ["This", "is", "an", "example", "sentence", "showing", "stopword", "usage"]
•Stopwords: ["this", "is", "an"]
•Stopword Count: 3

## Lexical_diversity_readability and scores ( NLTK--> tokenization, Textstast--> calculate scores: built -in fucntions )
 **Lexical Diversity (unique words / total words)**
- Values closer to 1 indicate higher diversity, meaning more varied vocabulary, while lower values suggest repetition in word use.
  
 **Flesch Reading Ease = 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)**
- Higher score indicates easier readability.
- 90-100: Very easy to read (understandable by an average 11-year-old)
- 60-70: Easily understood by 13- to 15-year-old students
- 0-30: Very difficult to read (best understood by university graduates)
 
 **Flesch-Kincaid Grade Level = 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59**
- Estimates the U.S. school grade level required to understand the text.
- Higher grade level. For example, a score of 8.0 means the text is appropriate for an 8th-grade student.
  
 **SMOG Index: SMOG Index = 1.043 * sqrt(number of polysyllabic words * (30 / number of sentences)) + 3.1291**
- Estimates the years of education required to comprehend a piece of text.

 ## Part Of Speech (POS) 
 - make a dict of the number od nouns, verbs, adj...
 **Example**
   - “The quick brown fox jumps over the lazy dog,” 
   - pos_NOUN is the count of nouns (e.g., “fox”, “dog”) --> 2
   - pos_VERB is the count of verbs (e.g., “jumps”)--> 1
   - pos_ADJ is the count of adjectives (e.g., “quick”, “lazy”)--> 2
   - pos_DET is the count of determiners (e.g., “the”) --> 1
   - pos_ADV is the count of adverbs (if any were present)--> 1....
  
## Sentiment and Emotion Analysis ##



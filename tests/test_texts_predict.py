
import unittest
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import preprocess, load_model
from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import compute_number_of_text_corrections, compute_number_of_text_corrections_using_TextBlob, compute_number_of_text_corrections_using_pyspellchecker, compute_number_of_text_corrections_using_nltk_words

class TestTextPrediction(unittest.TestCase):
    def test_prediction_using_TfidfVectorizer_MultinomialNB(self):
        text = """
        Also they feel more comfortable at home. Some school have decreased bullying and high and middle school because some students get bullied. Some Schools offter distance learning as an option for students to attend classes from home by way of online or video conferencing. Students can ncreased to learn at home. Also is more hard to students understand by online. students get distract at home. Some schools in United States ofter classes from home because is good option to students . Also students can stay more relaxing from home. Students don't want to go more at school and they want to get classes at home. Students get fall in environment learning. But students can get relaxing at home.

Students can get distract at home because they have easy to use phones. If students sleep in class they want to sleep at home too. They feel more bored at home because they need stay at home more time. Also students don't do anything at home because if they stay at home is esay to get more distract. Students can get fall in environment learning and is hard they learn at home. Students have more time to do homework. Also they don't want to learn by online. Also many students have hard time in class to understand that teacher explain in more hard they learn at home.

Some schools affter classes from home because they think is good option to students. If they get classes at home they don't learning. Many students work and they would tired and they don't want to learn.

Students don't pay attention in class they too don't pay attention at home because is more hard. But students get more distract stay near the family and they don't want to pay attention.

Conclude if students get classes from home by online they don't want to pay attention because is more easy they get distract. Students feel more pressure at home. Also they want to play or use phone and not is good option to students get class at home they get distration and decrease to learning. Also students get frustration in class because they don't understand but if they get classes in online they don't learning too because is more hard they learning from home.. Also they want to listening to music or play.

Students fall in environment learning because they learn at class when the teacher explain if students attend classes from home by online is hard they learn. Also student sleep more and stay with more energy to receive the class by online. Also they feel safe at home with their family"
"""
        import detect_ai_content
        import os
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = f = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/genai_text_detection_using_TfidfVectorizer_MultinomialNB.pickle'
        loaded_model = pickle.load(open(model_path, 'rb'))
        y_pred = loaded_model.predict(X=[text])
        print(y_pred)

    def test_prediction_using_LinearRegression(self):
        text = """
        Also they feel more comfortable at home. Some school have decreased bullying and high and middle school because some students get bullied. Some Schools offter distance learning as an option for students to attend classes from home by way of online or video conferencing. Students can ncreased to learn at home. Also is more hard to students understand by online. students get distract at home. Some schools in United States ofter classes from home because is good option to students . Also students can stay more relaxing from home. Students don't want to go more at school and they want to get classes at home. Students get fall in environment learning. But students can get relaxing at home.

Students can get distract at home because they have easy to use phones. If students sleep in class they want to sleep at home too. They feel more bored at home because they need stay at home more time. Also students don't do anything at home because if they stay at home is esay to get more distract. Students can get fall in environment learning and is hard they learn at home. Students have more time to do homework. Also they don't want to learn by online. Also many students have hard time in class to understand that teacher explain in more hard they learn at home.

Some schools affter classes from home because they think is good option to students. If they get classes at home they don't learning. Many students work and they would tired and they don't want to learn.

Students don't pay attention in class they too don't pay attention at home because is more hard. But students get more distract stay near the family and they don't want to pay attention.

Conclude if students get classes from home by online they don't want to pay attention because is more easy they get distract. Students feel more pressure at home. Also they want to play or use phone and not is good option to students get class at home they get distration and decrease to learning. Also students get frustration in class because they don't understand but if they get classes in online they don't learning too because is more hard they learning from home.. Also they want to listening to music or play.

Students fall in environment learning because they learn at class when the teacher explain if students attend classes from home by online is hard they learn. Also student sleep more and stay with more energy to receive the class by online. Also they feel safe at home with their family"
"""
        import detect_ai_content
        import os
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = f = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/genai_text_detection_using_ml_features.pickle'

        df = pd.DataFrame(data={'text': [text]})
        df = preprocess(df)

        loaded_model = pickle.load(open(model_path, 'rb'))
        y_pred = loaded_model.predict(X=df)
        print(y_pred)

if __name__ == '__main__':
    unittest.main()

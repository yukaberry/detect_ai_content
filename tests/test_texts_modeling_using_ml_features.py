
import unittest
import pandas as pd

from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import compute_number_of_text_corrections, compute_number_of_text_corrections_using_TextBlob, compute_number_of_text_corrections_using_pyspellchecker, compute_number_of_text_corrections_using_nltk_words
from detect_ai_content.ml_logic.preprocess import preprocess

def get_human_texts():
    texts = [
        "Also they feel more comfortable at home.",
        "Some school have decreased bullying and high and middle school because some students get bullied.",
        "Some Schools offter distance learning as an option for students to attend classes from home by way of online or video conferencing.",
        "Students can ncreased to learn at home. ",
        "Also is more hard to students understand by online. students get distract at home."
    ]
    return texts

def get_ai_texts():
    texts = [
        "Therefore, when it comes to allowing students the option to attend classes from home, there are intricacies that need to be taken into consideration in order to ensure that the best decision is made."
        "Ultimately, this decision will depend on the individual student and their ability to take advantage of the opportunities available to them..",
        "However, in the end, the effect that home-based classes will have on learning is largely dependent on the situation of the student.",
        "On the other hand, there could be a lack of social interaction with classmates, a lack of guidance from instructors, and potential technical issues as well.",
        "For example, those who are already motivated to learn and are self-disciplined may reap the full benefits of studying in the comfort of their own home.",
        "Conversely, for those who require more interaction and guidance that comes with physical classrooms, a home-based learning style may not be as effective.",
        "On the one hand, it eliminates the need for physical attendance to classrooms, reduces the psychology of peer pressure, and eliminates potential health risks resulting from attending crowded places."
        ]
    return texts

class TestTextModeling(unittest.TestCase):
    def test_detect_text_errors(self):
        text = """
        Also they feel more comfortable at home. Some school have decreased bullying and high and middle school because some students get bullied. Some Schools offter distance learning as an option for students to attend classes from home by way of online or video conferencing. Students can ncreased to learn at home. Also is more hard to students understand by online. students get distract at home. Some schools in United States ofter classes from home because is good option to students . Also students can stay more relaxing from home. Students don't want to go more at school and they want to get classes at home. Students get fall in environment learning. But students can get relaxing at home.

Students can get distract at home because they have easy to use phones. If students sleep in class they want to sleep at home too. They feel more bored at home because they need stay at home more time. Also students don't do anything at home because if they stay at home is esay to get more distract. Students can get fall in environment learning and is hard they learn at home. Students have more time to do homework. Also they don't want to learn by online. Also many students have hard time in class to understand that teacher explain in more hard they learn at home.

Some schools affter classes from home because they think is good option to students. If they get classes at home they don't learning. Many students work and they would tired and they don't want to learn.

Students don't pay attention in class they too don't pay attention at home because is more hard. But students get more distract stay near the family and they don't want to pay attention.

Conclude if students get classes from home by online they don't want to pay attention because is more easy they get distract. Students feel more pressure at home. Also they want to play or use phone and not is good option to students get class at home they get distration and decrease to learning. Also students get frustration in class because they don't understand but if they get classes in online they don't learning too because is more hard they learning from home.. Also they want to listening to music or play.

Students fall in environment learning because they learn at class when the teacher explain if students attend classes from home by online is hard they learn. Also student sleep more and stay with more energy to receive the class by online. Also they feel safe at home with their family"
"""
        errors = compute_number_of_text_corrections_using_pyspellchecker(text)
        self.assertEqual(errors, 17)

    def test_detect_text_errors_using_nltk(self):
        text = """
        Also they feel more comfortable at home. Some school have decreased bullying and high and middle school because some students get bullied. Some Schools offter distance learning as an option for students to attend classes from home by way of online or video conferencing. Students can ncreased to learn at home. Also is more hard to students understand by online. students get distract at home. Some schools in United States ofter classes from home because is good option to students . Also students can stay more relaxing from home. Students don't want to go more at school and they want to get classes at home. Students get fall in environment learning. But students can get relaxing at home.

Students can get distract at home because they have easy to use phones. If students sleep in class they want to sleep at home too. They feel more bored at home because they need stay at home more time. Also students don't do anything at home because if they stay at home is esay to get more distract. Students can get fall in environment learning and is hard they learn at home. Students have more time to do homework. Also they don't want to learn by online. Also many students have hard time in class to understand that teacher explain in more hard they learn at home.

Some schools affter classes from home because they think is good option to students. If they get classes at home they don't learning. Many students work and they would tired and they don't want to learn.

Students don't pay attention in class they too don't pay attention at home because is more hard. But students get more distract stay near the family and they don't want to pay attention.

Conclude if students get classes from home by online they don't want to pay attention because is more easy they get distract. Students feel more pressure at home. Also they want to play or use phone and not is good option to students get class at home they get distration and decrease to learning. Also students get frustration in class because they don't understand but if they get classes in online they don't learning too because is more hard they learning from home.. Also they want to listening to music or play.

Students fall in environment learning because they learn at class when the teacher explain if students attend classes from home by online is hard they learn. Also student sleep more and stay with more energy to receive the class by online. Also they feel safe at home with their family"
"""
        errors = compute_number_of_text_corrections_using_nltk_words(text)
        self.assertEqual(errors, 7)

    def test_preprocessing(self):
        texts = get_human_texts()
        df = pd.DataFrame(data={"text": texts})
        df['generated'] = 0
        X = df[['text']]
        X_preprocessed = preprocess(X)
        self.assertEqual("text" in X_preprocessed, False)

if __name__ == '__main__':
    unittest.main()

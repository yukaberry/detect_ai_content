
import unittest
import pandas as pd
from sklearn.model_selection import train_test_split

from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import preprocess,enrich, train_LogisticRegression_model, evaluate_model

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
    def test_preprocessing(self):
        texts = get_human_texts()
        df = pd.DataFrame(data={"text": texts})
        df['generated'] = 0
        X = df[['text']]
        X_preprocessed = preprocess(X)
        self.assertEqual("text" in X_preprocessed, False)

    def test_training(self):
        texts = get_human_texts()
        human_df = pd.DataFrame(data={"text": texts})
        human_df['generated'] = 0

        texts = get_ai_texts()
        ai_df = pd.DataFrame(data={"text": texts})
        ai_df['generated'] = 1

        df = pd.concat(objs=[ai_df, human_df])
        X = df[['text']]
        y = df['generated']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_train_preprocessed = preprocess(X_train)
        model = train_LogisticRegression_model(X_train_preprocessed, y_train)

        X_test_preprocessed = preprocess(X_test)
        results = evaluate_model(model=model, X_test_processed=X_test_preprocessed, y_test=y_test)
        print(results)

    def test_training_using_rich_texts(self):
        path = "./raw_data/huggingface.co_human_ai_generated_text/model_training_dataset_enriched_7Mo.csv"
        df = pd.read_csv(path)
        print(df.shape)

        X = df
        y = df['generated']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print(X_train.shape)
        X_train_preprocessed = preprocess(X_train, auto_enrich=False)

        print(X_train_preprocessed.shape)
        model = train_LogisticRegression_model(X_train_preprocessed, y_train)

        X_test_preprocessed = preprocess(X_test, auto_enrich=False)
        print(X_test_preprocessed.shape)

        results = evaluate_model(model=model, X_test_processed=X_test_preprocessed, y_test=y_test)
        print(results)

if __name__ == '__main__':
    unittest.main()

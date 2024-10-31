
import unittest
import pandas as pd
from sklearn.model_selection import train_test_split

from detect_ai_content.ml_logic.for_texts.using_ml_features.using_sentences_decomposition import to_sentences, preprocess, train_LogisticRegression_model, evaluate_model

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

class TestTextSeModelingUsingSentences(unittest.TestCase):
    def test_preprocessing(self):
        texts = ['Hello, je suis jerome. Et voilà!']
        df = pd.DataFrame(data={"text": texts})
        df['generated'] = 0

        df_sentences = to_sentences(df)
        self.assertEqual("text" in df_sentences, True)
        self.assertEqual(df_sentences.shape[0], 2)
        self.assertEqual(df_sentences.shape[1], 2)

        df_sentences = to_sentences(df, include_generated=False)
        self.assertEqual("text" in df_sentences, True)
        self.assertEqual(df_sentences.shape[0], 2)
        self.assertEqual(df_sentences.shape[1], 1)

        df_sentences_preprocessed = preprocess(df_sentences)
        self.assertEqual("text" in df_sentences_preprocessed, False)
        self.assertEqual(df_sentences_preprocessed.shape[0], 2)
        self.assertEqual(df_sentences_preprocessed.shape[1], 5)

    def test_train_and_evaluate(self):
        texts = get_human_texts()
        human_df = pd.DataFrame(data={"text": texts})
        human_df['generated'] = 0

        texts = get_ai_texts()
        ai_df = pd.DataFrame(data={"text": texts})
        ai_df['generated'] = 1

        df = pd.concat(objs=[ai_df, human_df])
        df = to_sentences(df)

        df_train, df_test = train_test_split(df, test_size=0.2)

        X_train_preprocessed = preprocess(df_train)
        y_train = df_train['generated']
        model = train_LogisticRegression_model(X_train_preprocessed, y_train)

        X_test_preprocessed = preprocess(df_test)
        y_test = df_test['generated']

        results = evaluate_model(model=model, X_test_processed=X_test_preprocessed, y_test=y_test)
        print(results)

    def test_sentences_enriched_dataset(self):
        df = pd.read_csv('./tests/data/model_training_dataset_sequences_enriched.csv')
        df_train, df_test = train_test_split(df, test_size=0.2)

        X_train_preprocessed = preprocess(df_train, execute_enrich=False)
        print(df_train.shape)
        print(X_train_preprocessed.shape)
        y_train = df_train['generated']
        model = train_LogisticRegression_model(X_train_preprocessed, y_train)

        X_test_preprocessed = preprocess(df_test, execute_enrich=False)
        print(df_test.shape)
        print(X_test_preprocessed.shape)
        y_test = df_test['generated']
        results = evaluate_model(model=model, X_test_processed=X_test_preprocessed, y_test=y_test)
        print(results)


if __name__ == '__main__':
    unittest.main()

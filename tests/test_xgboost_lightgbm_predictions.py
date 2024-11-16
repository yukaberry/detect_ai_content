import unittest
import pandas as pd
from detect_ai_content.ml_logic.for_texts.xgboost_external import XGBoostExternal

def get_human_texts():
    return [
        "Also they feel more comfortable at home.",
        "Some school have decreased bullying and high and middle school because some students get bullied.",
        "Some Schools offer distance learning as an option for students to attend classes from home by way of online or video conferencing.",
        "Students can increased to learn at home.",
        "Also is more hard to students understand by online. students get distract at home."
    ]

def get_ai_texts():
    return [
        "Therefore, when it comes to allowing students the option to attend classes from home, there are intricacies that need to be taken into consideration in order to ensure that the best decisions are made.",
        "Ultimately, this decision will depend on the individual student and their ability to take advantage of the opportunities available to them.",
        "However, in the end, the effect that home-based classes will have on learning is largely dependent on the situation of the student.",
        "On the other hand, there could be a lack of social interaction with classmates, a lack of guidance from instructors, and potential technical issues as well.",
        "For example, those who are already motivated to learn and are self-disciplined may reap the full benefits of studying in the comfort of their own home."
    ]

class TestExternalXgBoost(unittest.TestCase):

    def setUp(self):
        """Initialize the XgBoost instance before each test."""
        self.xgboost_external = XGBoostExternal()

    def test_model_loading(self):
        """Test if the XgBoost model loads correctly."""
        self.assertIsNotNone(self.xgboost_external.model, "Failed to load the XgBoost model.")

    def test_predict_human_texts(self):
        """Test the predict function with human-written texts."""
        human_texts = get_human_texts()
        for text in human_texts:
            prediction, message = self.xgboost_external.predict(text)
            self.assertEqual(prediction, 0, f"Expected 'Human generated', got {message}")

    def test_predict_ai_texts(self):
        """Test the predict function with AI-generated texts."""
        ai_texts = get_ai_texts()
        for text in ai_texts:
            prediction, message = self.xgboost_external.predict(text)
            self.assertEqual(prediction, 1, f"Expected 'AI generated', got {message}")

# Run the tests
if __name__ == '__main__':
    unittest.main()

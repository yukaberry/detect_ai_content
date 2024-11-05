
import unittest

from detect_ai_content.ml_logic.for_images.TrueNetImageUsinCustomCNN import *

class TestImageTrueNetImageUsinCustomCNN(unittest.TestCase):
    def test_load_model(self):
        model = TrueNetImageUsinCustomCNN()
        self.assertEqual(model is not None, True)

    def test_predict_using_model(self):
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        image_path = os.path.join(f'{module_dir_path}/..', 'tests','data', 'real_0192.jpg')

        model = TrueNetImageUsinCustomCNN()
        y_pred = model.predict(image_path)
        self.assertEqual(y_pred == 0 or y_pred == 1, True, msg="Image from the kaggle-cifake-real-and-ai-generated-synthetic dataset")

if __name__ == '__main__':
    unittest.main()

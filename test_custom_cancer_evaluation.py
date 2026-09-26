import unittest
from custom_cancer_evaluation import metric_image_recall

class TestMetricImageRecall(unittest.TestCase):
    def test_non_image_category(self):
        self.assertIsNone(metric_image_recall("Some answer", "text"))
        self.assertIsNone(metric_image_recall("[IMAGE: test.png]", "general"))

    def test_proper_tag(self):
        self.assertEqual(metric_image_recall("Here is the image [IMAGE: figure1.png]", "image"), 1.0)
        self.assertEqual(metric_image_recall("[IMAGE: chart.jpeg] shows...", "image"), 1.0)
        self.assertEqual(metric_image_recall("Multiple [image: a.png] tags [IMAGE: b.jpg]", "image"), 1.0)

    def test_visual_keyword(self):
        self.assertEqual(metric_image_recall("The figure shows the results.", "image"), 0.5)
        self.assertEqual(metric_image_recall("As depicted in the diagram below.", "image"), 0.5)
        self.assertEqual(metric_image_recall("See the kaplan survival curve.", "image"), 0.5)

    def test_no_image_reference(self):
        self.assertEqual(metric_image_recall("The study concluded that treatment is effective.", "image"), 0.0)
        self.assertEqual(metric_image_recall("No relevant visual information found.", "image"), 0.0)

if __name__ == '__main__':
    unittest.main()

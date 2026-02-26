#!/usr/bin/python3.11
"""
Tests for the EmotionDetection package.

This test suite validates the integration with IBM Watson's
NLP service and ensures correct averaging of emotion scores.
"""

import unittest
from EmotionDetection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """
    Unit tests for the EmotionDetection module.

    Tests the dominant emotion detection logic by querying the
    IBM Watson NLP service.
    """

    def test_emotion_detection_joy(self):
        """
        Test if 'joy' is correctly identified.

        Input: "I am glad this happened"
        Expected: dominant_emotion == "joy"
        """
        test_text = "I am glad this happened"
        self.assertEqual(emotion_detector(test_text)["dominant_emotion"], "joy")

    def test_emotion_detection_anger(self):
        """
        Test if 'anger' is correctly identified.

        Input: "I am really mad about this"
        Expected: dominant_emotion == "anger"
        """
        test_text = "I am really mad about this"
        self.assertEqual(emotion_detector(test_text)["dominant_emotion"], "anger")

    def test_emotion_detection_disgust(self):
        """
        Test if 'disgust' is correctly identified.

        Input: "I feel disgusted just hearing about this"
        Expected: dominant_emotion == "disgust"
        """
        test_text = "I feel disgusted just hearing about this"
        self.assertEqual(emotion_detector(test_text)["dominant_emotion"], "disgust")

    def test_emotion_detection_sadness(self):
        """
        Test if 'sadness' is correctly identified.

        Input: "I am so sad about this"
        Expected: dominant_emotion == "sadness"
        """
        test_text = "I am so sad about this"
        self.assertEqual(emotion_detector(test_text)["dominant_emotion"], "sadness")

    def test_emotion_detection_fear(self):
        """
        Test if 'fear' is correctly identified.

        Input: "I am really afraid that this will happen"
        Expected: dominant_emotion == "fear"
        """
        test_text = "I am really afraid that this will happen"
        self.assertEqual(emotion_detector(test_text)["dominant_emotion"], "fear")


unittest.main()

"""Emotion Detection via IBM watson"""

import requests
import json
from collections import defaultdict


class EmotionDetectionError(Exception):
    """Base Class for Errors in this Module."""

def emotion_detector(text_to_analyse):
    """
    Queries IBM's Watson NLP API for emotion detection of a given text string

    Parameters
    ----------
    text_to_analyse : str
        The text that should be analysed

    Raises
    ------
    EmotionDetectionError
        EmotionDetection API is not reachable or fails

    Returns
    -------
    emotion_scores : dict
        contains the name of each emotion as key and the average score as value,
        plus the name of the emotion with highest score as value for key "dominant_emotion"
    """
    url = (
        "https://sn-watson-emotion.labs.skills.network"
        "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyse}}

    try:
        response = requests.post(url, json=myobj, headers=headers, timeout=30)

        if response.status_code == 400:
            # watson API couldn't process input string

            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None
            }

        response.raise_for_status()

        # format results to be returned
        response_json = json.loads(response.text)
        if "emotionPredictions" not in response_json:
            raise EmotionDetectionTechnicalError(
                "No emotionPredictions found in API response"
            )

        emotion_sums = defaultdict(lambda: {"count": 0, "score_sum": 0.0})
        for emotion in response_json["emotionPredictions"]:
            if "emotion" not in emotion:
                continue

            # accumulate scores and counts of all emotions
            for emo_name, emo_value in emotion["emotion"].items():
                emotion_sums[emo_name]["score_sum"] += emo_value
                emotion_sums[emo_name]["count"] += 1

        if not emotion_sums:
            raise EmotionDetectionInferenceError(
                "No emotions could be extracted from the API response."
            )

        # avg emotion scores
        emotion_avgs = {}
        for emo, data in emotion_sums.items():
            emotion_avgs[emo] = data["score_sum"] / data["count"]

        # find dominant emotion
        dominant_emotion = max(emotion_avgs, key=emotion_avgs.get)

        return emotion_avgs | {"dominant_emotion": dominant_emotion}

    except requests.exceptions.RequestException as e:
        raise EmotionDetectionError(
            f"Error using API failure: {type(e).__name__}"
        ) from e

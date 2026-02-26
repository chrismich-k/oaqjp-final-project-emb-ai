"""Emotion Detection via IBM watson"""

import requests
import json
from collections import defaultdict


class EmotionDetectionError(Exception):
    """Base Class for Errors in this Module."""


class EmotionDetectionInferenceError(EmotionDetectionError):
    """Input is not suitable to be processed by EmotionDetection (empty or not recognisable)."""


class EmotionDetectionTechnicalError(EmotionDetectionError):
    """EmotionDetection API is not reachable or fails."""


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
    EmotionDetectionError
        Input not suitable to be processed by EmotionDetection (empty or not recognisable)

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

    # check input
    if not text_to_analyse or not text_to_analyse.strip():
        raise EmotionDetectionInferenceError("input is empty or blank.")

    try:
        response = requests.post(url, json=myobj, headers=headers, timeout=30)

        if response.status_code == 400:
            # watson API couldn't process input string
            raise EmotionDetectionInferenceError(
                f"Bad HTTP status code from watson API: {response.status_code}"
            )

        response.raise_for_status()

        # format results to be returned
        response_json = json.loads(response.text)
        if not "emotionPredictions" in response_json:
            raise EmotionDetectionTechnicalError(
                "No emotionPredictions found in API response"
            )

        emotion_sums = {}
        for emotion in response_json["emotionPredictions"]:
            if not "emotion" in emotion:
                continue

            # accumulate scores and counts of all emotions
            for emo_name in emotion["emotion"].keys():
                if emo_name not in emotion_sums:
                    emotion_sums[emo_name] = {
                        "count": 1,
                        "score_sum": emotion["emotion"][emo_name],
                    }
                else:
                    emotion_sums[emo_name]["score_sum"] += emotion["emotion"][emo_name]
                    emotion_sums[emo_name]["count"] += 1

        if not emotion_sums:
            raise EmotionDetectionInferenceError(
                "No emotions could be extracted from the API response."
            )

        # avg emotion scores
        emotion_avgs = defaultdict(lambda: {"count": 0, "score_sum": 0.0})
        for emo, data in emotion_sums.items():
            emotion_avgs[emo] = data["score_sum"] / data["count"]

        # find dominant emotion
        dominant_emotion = max(emotion_avgs, key=emotion_avgs.get)

        return emotion_avgs | {"dominant_emotion": dominant_emotion}

    except requests.exceptions.RequestException as e:
        raise EmotionDetectionTechnicalError(
            f"Technical API failure: {type(e).__name__}"
        ) from e

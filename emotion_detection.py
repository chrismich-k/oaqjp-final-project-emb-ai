"""Emotion Detection via IBM watson"""

import requests


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
    response_text : str
        the text attribute of the API response object
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
        return response.text

    except requests.exceptions.RequestException as e:
        raise EmotionDetectionTechnicalError(
            f"Technical API failure: {type(e).__name__}"
        ) from e

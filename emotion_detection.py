"""Emotion Detection via IBM watson"""
import requests

class EmotionDetectionError(Exception):
    """Base Class for Errors in this Module."""

class EmotionDetectionRequestError(EmotionDetectionError):
    """Base Class for Errors in this Module."""

class EmotionDetectionFailureError(EmotionDetectionError):
    """Error class for bad response status codes."""

def emotion_detector(text_to_analyse):
    """
    Queries IBM's Watson NLP API for emotion detection of a given text string

    Parameters
    ----------
    text_to_analyse : str
        The text that should be analysed
    
    Raises
    ------
    EmotionDetectionRequestError
        HTTP request to watson API went wrong
    EmotionDetectionFailureError
        watson API couldn't process the text

    Returns
    -------
    response_text : str
        the text attribute of the API response object
    """
    url = ('https://sn-watson-sentiment-bert.labs.skills.network'
           '/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict')
    headers = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}
    myobj = { "raw_document": { "text": text_to_analyse } }
    try:
        response = requests.post(url, json = myobj, headers=headers, timeout=30)

        if response.status_code == 500:
            # watson API couldn't process input string
            raise EmotionDetectionFailureError(
                f"Bad HTTP status code from watson API: {response.status_code}"
            )

        if response.status_code != 200:
            # raise any other bad http status
            response.raise_for_status()

        return response.text

    except requests.Timeout as e:
        raise EmotionDetectionRequestError("Request timed out") from e
    except requests.ConnectionError as e:
        raise EmotionDetectionRequestError("Error connectiong to watson API") from e
    except requests.TooManyRedirects as e:
        raise EmotionDetectionRequestError("Too many redirects for watson API") from e
    except requests.HTTPError as e:
        raise EmotionDetectionRequestError("HTTP error for watson API") from e
    except Exception as e:
        raise EmotionDetectionRequestError("Unexpected Error when querying for watson API") from e

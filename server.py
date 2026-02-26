#!/usr/bin/python3.11
from flask import Flask, request, render_template
from EmotionDetection import emotion_detector

app = Flask(__name__)

@app.errorhandler(404)
def api_not_found(error):
    """
    general 404 error handler
    """
    
    return({"message": "API path not found"}, 404)

# Define a route for the root URL ("/")
@app.route("/")
def render_index_page():
    """"
    This function initiates the rendering of the main application
    page over the Flask channel
    """

    return render_template("index.html")

@app.route("/emotionDetector")
def api_emotion_detector():
    """
    This code receives the text from the HTML interface and runs emotion detection
    on it using emotion_detection() function.
    The output returned shows scores for each detected emotion and the dominant emotion.
    """
    param_text = request.args.get('textToAnalyze')
    if param_text:
        response = emotion_detector(param_text)
        out_str = (
            f"For the given statement, the system response is "
            f"'anger': {response['anger']}, "
            f"'disgust': {response['disgust']}, "
            f"'fear': {response['fear']}, "
            f"'joy': {response['joy']} and "
            f"'sadness': {response['sadness']}. "
            f"the dominant emotion is {response['dominant_emotion']}."
        )
        return out_str
    else:
        return None

if __name__ == "__main__":
    # start the flask app and deploy it on localhost:5000
    app.run(host="0.0.0.0", port=5000)

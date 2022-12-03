from flask import Flask, request, jsonify, render_template
from answer_question import answer_question
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


app = Flask(__name__)

# Default model name
DEFAULT_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Load the default model and tokenizer
app.last_used_model_name = DEFAULT_MODEL_NAME
app.model = AutoModelForQuestionAnswering.from_pretrained(app.last_used_model_name)
app.tokenizer = AutoTokenizer.from_pretrained(app.last_used_model_name)


@app.route('/', methods=['GET', 'POST'])
def get_answer():
    if request.method == 'POST':
        # Parse input
        data = request.form
        question = data['question']
        reference = data['reference']
        model_name = data.get('model_name', DEFAULT_MODEL_NAME)

        # To avoid loading the model and tokenizer every time, we only do it if the model name has changed
        if model_name != app.last_used_model_name:
            # Load the new model and tokenizer
            app.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            app.tokenizer = AutoTokenizer.from_pretrained(model_name)
            app.last_used_model_name = model_name

        # Get the answer to the question
        answer = answer_question(question, reference, app.model, app.tokenizer)
        answer = answer if answer else 'I do not know the answer to that question 😢'

        # Return the predicted answer as a JSON object
        return jsonify({'answer': answer.capitalize()})
    else:
        # Return the HTML page with the form
        return render_template('index.html')


if __name__ == '__main__':
    app.run()

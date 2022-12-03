import pytest
from app import app, DEFAULT_MODEL_NAME


def test_default_model():
    assert app.last_used_model_name == DEFAULT_MODEL_NAME
    assert app.tokenizer is not None
    assert app.model is not None


def test_html_page():
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert 'AskBERT' in response.data.decode('utf-8')


def test_form_submission():
    # Submit the form with a question and reference text
    response = app.test_client().post('/', data={
        'question': 'What is the capital of France?',
        'reference': 'Paris is the capital of France.',
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['answer'] == 'Paris'


def test_empty_input():
    # Submit the form with an empty question
    response = app.test_client().post('/', data={
        'question': '',
        'reference': 'France is a country in Europe.',
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['answer'] == 'I do not know the answer to that question 😢'

    # Submit the form with an empty reference text
    response = app.test_client().post('/', data={
        'question': 'What is the capital of France?',
        'reference': '',
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['answer'] == 'I do not know the answer to that question 😢'


def test_model_name_change():
    # Submit the form with a different model name
    response = app.test_client().post('/', data={
        'question': 'What is the capital of France?',
        'reference': 'France is a country in Europe.',
        'model_name': 'bert-base-cased',
    })
    assert response.status_code == 200
    assert app.last_used_model_name == 'bert-base-cased'
    assert app.model is not None
    assert app.tokenizer is not None


if __name__ == '__main__':
    pytest.main()

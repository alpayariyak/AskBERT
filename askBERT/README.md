# AskBERT

AskBERT is a simple web app that uses a BERT model of your choice to answer questions based on a reference text. 

## Installation

To use AskBERT, you need to have Python 3 and pip installed on your system. Then, run the following command to install the required packages:

```
pip install -r requirements.txt
```

## Usage

Run the following command:
```
python app.py
```
Open your browser and go to http://localhost:5000/

### Specifying Model
In the app, you are able to specify the model to use for prediction. The default is a BERT model that was pre-trained on the Stanford Question Answering Dataset(SQuAD). If you would like to test a different one out, you can either choose from existing ones on the Hugging Face model hub, or upload your own model. It is heavily recommended you use a BERT model that has been fine-tuned on the SQuAD dataset to avoid possible issues.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
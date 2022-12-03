import torch


def answer_question(question, reference, model, tokenizer):
    """
    Returns answer to given question by reference
    :param question: Input question
    :param reference: Input text to look for answer in
    :param model: Model to use for prediction
    :param tokenizer: Tokenizer to use for the model
    :return: answer
    """

    # Tokenize the question and reference and assign IDs
    token_IDs = tokenizer.encode_plus(question, reference, max_length=512, truncation=True, return_tensors='pt')

    # Extract the tensor containing the token IDs from the dictionary
    input_tokens = token_IDs["input_ids"]
    token_type_ids = token_IDs["token_type_ids"]
    attention_mask = token_IDs["attention_mask"]
    # Make the model predict the start and end tokens of the answer
    model_output = model(input_tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
    start_scores, end_scores = model_output.start_logits, model_output.end_logits

    # Find the combination of start and end tokens that has the highest score
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = tokenizer.decode(input_tokens.squeeze()[answer_start:answer_end + 1].tolist())  # +1 to include last token
    special_tokens = ['[SEP]', '[CLS]', '[PAD]', '[UNK]']
    answer = ' '.join([word for word in answer.split() if word not in special_tokens])
    return answer
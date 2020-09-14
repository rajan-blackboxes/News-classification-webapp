"""
Contains accuracy metrics functions
"""
import torch
from tqdm.notebook import tqdm


def predict(list_of_sequences, model, dataset, device='cpu'):
    """
    Args:
        list_of_sequences:
        model:
        dataset:
        device:

    Returns:
        predicted

    """
    preds = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(list_of_sequences))):
            test_vector = torch.LongTensor([dataset.pad_index(
                dataset.convert_sequence(list_of_sequences[i], dataset.word2index, dataset.tokenizer), dataset.max_seq_len,
                dataset.word2index)[:dataset.max_seq_len]]).to(device)
            output = model(test_vector)
            pred = torch.argmax(output, dim=-1)
            preds.append(pred)
    return list(map(int, preds))







def classify(news, model, dataset, device='cpu'):
    """
    Classifies text: news
    Args:
        news: given news article
        model: trained model
        dataset: a vocabulary class defined (containing word2index, tokenizer, max_len_q objects)
        device: which device to use

    Returns:
        dictionary of classes and its probabilities
    """
    model.eval()
    classes = dataset.labels_list
    classes_names = {k: v for k, v in zip(sorted(classes), ['World', 'Sports', 'Business', 'Sci/Tech'])}
    with torch.no_grad():
        test_vector = torch.LongTensor([dataset.pad_index(
            dataset.convert_sequence(news, dataset.word2index, dataset.tokenizer), dataset.max_seq_len,
            dataset.word2index)[:dataset.max_seq_len]]).to(device)
        output = model(test_vector)
    probs = torch.nn.Softmax(dim=0)(output) * 100
    percentage_on_classes = {classes_names[idx]: int(percent) for idx, percent in enumerate(probs)}
    return percentage_on_classes


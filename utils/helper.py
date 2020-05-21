# External Libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy

nlp = spacy.load('en_core_web_sm')


def prep_each_document(document: str):
    unsorted_sent_words = [[token.text for token in sent] for sent in nlp(document).sents]
    unsorted_sent_lens = torch.tensor([len(sent_words) for sent_words in unsorted_sent_words])

    _, sorted_sent_ixs = unsorted_sent_lens.sort(dim=0, descending=True)
    _, unsorted_sent_ixs = sorted_sent_ixs.sort(dim=0, descending=False)

    sorted_sent_words = []
    unsorted_sent_packet = torch.zeros(len(unsorted_sent_lens), max(unsorted_sent_lens), dtype=torch.long)
    for sorted_sent_ix in sorted_sent_ixs:
        sorted_sent_words.append(unsorted_sent_words[sorted_sent_ix])

    sorted_sent_lens = torch.tensor([len(sent_words) for sent_words in sorted_sent_words])

    return sorted_sent_words, sorted_sent_lens, unsorted_sent_ixs


def _map_attn(sentence_attns, sentence_word_wattns, sentence_word_tokens):
    highlighted_text = []
    for sent_attn, sent_word_attns, sent_word_tokens in zip(sentence_attns, sentence_word_wattns, sentence_word_tokens):
        highlighted_text.append('<br><p style="background:rgba(249, 122, 122, {0});">'.format(sent_attn.item()))
        for sent_word_attn, sent_word_token in zip(sent_word_attns, sent_word_tokens):
            highlighted_text.append(
                '<span style="background-color:rgba(135,206,250,{0});">{1} </span>'.format(sent_word_attn.item(),
                                                                                           sent_word_token))
        highlighted_text.append('</p>')
    highlighted_text = ' '.join(highlighted_text)
    return highlighted_text


def predict(text, model):
    softmax = nn.Softmax(dim=0)
    text = ' '.join(str(text).split())
    text_packet = prep_each_document(text)
    pred_packet = model([text_packet])
    pred_unnormalized, sattns, wattns = pred_packet[0]
    sorted_sent_packet, sorted_sent_lens, unsorted_sent_ix = text_packet

    unsorted_sent_pack = [sorted_sent_packet[ix] for ix in unsorted_sent_ix]
    unsorted_sent_lens = sorted_sent_lens[unsorted_sent_ix]

    predicted_class = softmax(pred_unnormalized).argmax().item()
    assert [len(each_wattn) for each_wattn in wattns] == [len(each_unsorted_sent) for each_unsorted_sent in
                                                          unsorted_sent_pack], "Mismatch in Words & Word Attentions"
    highlighted_text = _map_attn(sattns, wattns, unsorted_sent_pack)

    return predicted_class, highlighted_text


def plot_confusion(true_labels, pred_labels, category_labels=None):
    confusion = torch.zeros(len(category_labels), len(category_labels))
    for t, p in zip(true_labels, pred_labels):
        confusion[t][p] += 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    ax.set_xticklabels([''] + category_labels, rotation=90)
    ax.set_yticklabels([''] + category_labels)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    return fig


def model_train(model, model_criterion, model_optimizer, dataloader):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        model_optimizer.zero_grad()
        batch_texts, batch_labels = batch
        batch_input_packets = [prep_each_document(batch_text) for batch_text in batch_texts]
        batch_output_packets = model(batch_input_packets)
        batch_output_unnormalized = torch.stack(
            [output_unnormalized for output_unnormalized, _, _ in batch_output_packets])
        loss = model_criterion(input=batch_output_unnormalized,
                               target=batch_labels)
        loss.backward()
        model_optimizer.step()
        epoch_loss += loss.item() / len(dataloader)
    return model, model_criterion, model_optimizer, epoch_loss


def model_evaluate(model, model_criterion, dataloader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_texts, batch_labels = batch
            batch_input_packets = [prep_each_document(batch_text) for batch_text in batch_texts]
            batch_output_packets = model(batch_input_packets)
            batch_output_unnormalized = torch.stack(
                [output_unnormalized for output_unnormalized, _, _ in batch_output_packets])
            loss = model_criterion(input=batch_output_unnormalized,
                                   target=batch_labels)
            epoch_loss += loss.item() / len(dataloader)
    return epoch_loss


#
def model_test(model, dataloader):
    softmax = nn.Softmax(dim=1)
    model.eval()
    epoch_loss = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch_texts, batch_labels = batch
            batch_input_packets = [prep_each_document(batch_text) for batch_text in batch_texts]
            batch_output_packets = model(batch_input_packets)
            batch_output_unnormalized = torch.stack(
                [output_unnormalized for output_unnormalized, _, _ in batch_output_packets])
            batch_scores_normalized = softmax(batch_output_unnormalized)

            true_labels.append(batch_labels)
            pred_labels.append(torch.argmax(batch_scores_normalized, dim=1))
    true_labels = torch.cat(true_labels)
    pred_labels = torch.cat(pred_labels)
    return true_labels, pred_labels

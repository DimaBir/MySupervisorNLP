import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import shap
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style

from utils import sentence_feature_extractor, get_num_of_wordiness


def prepare_data(sentence):
    sentence = "[CLS] " + sentence + " [SEP]"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_sent = tokenizer.tokenize(sentence)

    # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
    # In the original paper, the authors used a length of 512.
    MAX_LEN = 128

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_id = tokenizer.convert_tokens_to_ids(tokenized_sent)

    # Pad our input tokens
    input_id = pad_sequences([input_id], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_id:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    sentence_input = torch.tensor(input_id)
    sentence_mask = torch.tensor(attention_masks)
    sentence_label = torch.tensor(np.empty(1, dtype=int))

    validation_data = TensorDataset(sentence_input, sentence_mask, sentence_label)
    validation_sampler = SequentialSampler(validation_data)
    return DataLoader(validation_data, sampler=validation_sampler, batch_size=1)


def evaluate_model(dataloader, model, device):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Evaluate data for one epoch
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids.type(torch.LongTensor), token_type_ids=None, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        return np.argmax(logits, axis=1).flatten()


def predict_class(sentence, model, device):
    result = None
    sentence_dataloader = prepare_data(sentence=sentence)
    pred_label = evaluate_model(dataloader=sentence_dataloader, model=model, device=device)

    threshold = 0.2
    factor = 1.9
    delta = 0.02
    res = get_num_of_wordiness(sentence) / len(sentence.split())

    if pred_label == 1 and res >= threshold:
        result = "Wordy"
    elif pred_label == 1 and res < threshold:
        result = "Clear" if res < threshold / factor + delta else "Wordy"
    elif pred_label == 0 and res < threshold:
        result = "Clear"
    elif pred_label == 0 and res >= threshold:
        result = "Wordy" if threshold * factor - delta < res and res != 0.4 else "Clear"
        # TODO: Deal with "A few inches of snow is necessary to go sledding."

    return result


def init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_name = 'best_0.9_checkpoint.pth'
    path = F"{model_save_name}"

    # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model, device


if __name__ == '__main__':
    not_wordy_0 = ("Among pathological hallmarks of AD are the senile plaques, which are formed by the copper induced aggregation of the amyloid beta peptides.", "Clear")
    not_wordy_1 = ("Unmixing results is highly affected from imaging geometry: camera/view zenithal and azimuth angles and light source direction.", "Clear")
    not_wordy_2 = ("The main idea of the DIP paper is using the network itself as the regularization for the corrupted image, without the need for training on thousandths of examples.", "Clear")
    not_wordy_3 = ("Imaging is taking a big part of our lives, sciences use imagery for discovering from which particles are moon or Mars made of, agricultures know to irrigate or fertilize their field according to imagery of their field..", "Clear")
    not_wordy_4 = ("Indeed, if parasites and pathogens follow the patterns predicted for other taxa, it is reasonable to expect that some diseases will adapt to changing environmental conditions and potentially increase in prevalence, whereas others will suffer negative consequences leading to range contractions and even local extinctions.", "Clear")
    not_wordy_5 = ("In my opinion torture is always wrong.", "Clear")
    not_wordy_6 = ("A few inches of snow is necessary to go sledding.", "Clear")
    not_wordy_7 = ("New students are required to attend a meeting on Friday, September 22.", "Clear")
    not_wordy_8 = ("I bought a dog for companionship.", "Clear")
    wordy_1 = ("In my research I'm following the long quest for cognitive system.", "Wordy")
    wordy_2 = ("This technique was proved to be efficient and accurate one, however, it still needneeds a primary expert analysis and not fully automatic.", "Wordy")
    wordy_3 = ("At first, it is rather surprising that it is an abstract thing.", "Wordy")
    wordy_4 = ("A few inches of snow on the ground is all that is necessary in order for a person to be able to go sledding.", "Wordy")
    wordy_5 = ("The subjects that are considered most important by students are those that have been shown to be useful to them after graduation.", "Wordy")
    wordy_6 = ("There are many students who like reading.", "Wordy")
    wordy_7 = ("As part of the Paris agreement which was  signed in 2015, the world’s nations have agreed to pursue efforts to limit global warming to 1.5oC above the pre-industrial levels, in light of the  risks of the climate crisis.", "Wordy")
    wordy_8 = ("The theory of lattices is a well developed one and has been used  to define the  real world objects known as crystals.", "Wordy")
    wordy_9 = ("I bought a dog for the purpose of providing me with companionship.", "Wordy")

    sentences = [not_wordy_0, not_wordy_1, not_wordy_2, not_wordy_3, not_wordy_4, not_wordy_5, not_wordy_6, not_wordy_7,
                 not_wordy_8, wordy_1, wordy_2, wordy_3, wordy_4, wordy_5, wordy_6, wordy_7, wordy_8, wordy_9]

    model, device = init()

    result_labels = []

    for sen in sentences:
        result_labels.append(predict_class(model=model, device=device, sentence=sen[0]))

    i = 0
    correct = 0
    for label in result_labels:
        print("\n" + sentences[i][0] + "\nPrediction: (" + label + ") -- " + "Correct" if sentences[i][1] == label else "Wrong")
        correct = correct + (1 if sentences[i][1] == label else 0)
        i = i + 1

    print("Accuracy: %.2f%%" % ((correct/i) * 100))

import sys
import os
import random

# Add project root directory to sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from slowtorch.tensor import Tensor
from slowtorch.nn import Module, RNN, Linear, CrossEntropyLoss, Softmax
from slowtorch.optim import Adam
from slowtorch.data import TensorDataset, DataLoader


# Sample miniature NLP dataset: Sentences & Sentiments (0: Negative, 1: Positive)
TRAIN_DATA = [
    ("this film is great and fun", 1),
    ("i loved this movie so good", 1),
    ("awesome acting brilliant story", 1),
    ("a fantastic joy to watch", 1),
    ("superb performance wonderful masterpiece", 1),
    ("terrible movie so boring and bad", 0),
    ("worst film i hated this waste", 0),
    ("awful story horrible acting waste", 0),
    ("poor script very boring avoid", 0),
    ("disaster of a film pure trash", 0),
]

TEST_DATA = [
    ("this movie is awesome and great", 1),
    ("horrible script terrible waste boring", 0),
]


def build_vocab(dataset):
    word2idx = {"<pad>": 0, "<unk>": 1}
    for text, _ in dataset:
        for token in text.lower().split():
            if token not in word2idx:
                word2idx[token] = len(word2idx)
    return word2idx


def text_to_one_hot_sequence(text, word2idx, max_len=6):
    tokens = text.lower().split()[:max_len]
    vocab_size = len(word2idx)
    sequence = []

    for word in tokens:
        idx = word2idx.get(word, word2idx["<unk>"])
        one_hot = [0.0] * vocab_size
        one_hot[idx] = 1.0
        sequence.append(one_hot)

    # Pad if sequence is shorter than max_len
    while len(sequence) < max_len:
        pad_vec = [0.0] * vocab_size
        pad_vec[word2idx["<pad>"]] = 1.0
        sequence.append(pad_vec)

    return sequence


class RNNSentimentClassifier(Module):
    def __init__(self, vocab_size, hidden_dim=8, num_classes=2):
        super().__init__()
        self.rnn = RNN(input_size=vocab_size, hidden_size=hidden_dim, batch_first=True)
        self.fc = Linear(in_features=hidden_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        # x is (batch_size, seq_len, vocab_size)
        # output is (batch_size, seq_len, hidden_dim)
        # h_n is (batch_size, hidden_dim) -> summary representation of sentence
        output, h_n = self.rnn(x)
        logits = self.fc(h_n)
        return logits


def main():
    random.seed(42)

    word2idx = build_vocab(TRAIN_DATA)
    vocab_size = len(word2idx)
    max_len = 6

    # Encode training set
    raw_x = []
    raw_y = []
    for text, label in TRAIN_DATA:
        raw_x.append(text_to_one_hot_sequence(text, word2idx, max_len=max_len))
        raw_y.append([float(label)])

    x_tensor = Tensor(raw_x, requires_grad=False)
    y_tensor = Tensor(raw_y, requires_grad=False)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = RNNSentimentClassifier(vocab_size=vocab_size, hidden_dim=8, num_classes=2)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.03)
    softmax = Softmax()

    epochs = 35
    total_samples = len(TRAIN_DATA)

    print("Starting RNN Sentiment Analysis Training...")
    print("Vocabulary Size: " + str(vocab_size) + " unique tokens | Max Length: " + str(max_len))
    print("Architecture: RNN(in=" + str(vocab_size) + ", hidden=8, batch_first=True) -> Linear(8, 2)")
    print("----------------------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct_preds = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            logits = model(batch_x)
            targets = [int(item[0]) for item in batch_y.data]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = len(targets)
            total_loss = total_loss + loss.data * batch_size

            for i in range(batch_size):
                pred_c = 0 if logits.data[i][0] > logits.data[i][1] else 1
                if pred_c == targets[i]:
                    correct_preds = correct_preds + 1

        avg_loss = total_loss / float(total_samples)
        acc = (correct_preds / float(total_samples)) * 100.0

        if epoch % 5 == 0 or epoch == 1:
            print("Epoch: " + str(epoch).rjust(2) + " | Loss: " + str(round(avg_loss, 4)).ljust(7) + " | Accuracy: " + str(round(acc, 2)) + "%")

    print("----------------------------------------------------------------------")
    print("Training finished! Testing on unseen evaluation sentences:")

    test_x = [text_to_one_hot_sequence(t, word2idx, max_len=max_len) for t, _ in TEST_DATA]
    test_tensor = Tensor(test_x)
    test_logits = model(test_tensor)
    test_probs = softmax(test_logits)

    labels_map = {0: "NEGATIVE", 1: "POSITIVE"}
    for idx in range(len(TEST_DATA)):
        sentence, expected = TEST_DATA[idx]
        p_neg = round(test_probs.data[idx][0], 3)
        p_pos = round(test_probs.data[idx][1], 3)
        pred_label = 1 if p_pos > p_neg else 0

        print("Sentence: \"" + sentence + "\"")
        print("  Expected: " + labels_map[expected] + " | Predicted: " + labels_map[pred_label] + " (Neg: " + str(p_neg) + ", Pos: " + str(p_pos) + ")")


if __name__ == "__main__":
    main()
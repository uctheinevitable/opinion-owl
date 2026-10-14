"""ATAE-LSTM building blocks shared by notebooks 02, 05 and 06.

Kept in one place so the training notebook and the prediction notebooks can never
disagree about the tokenizer, the padding rules or the model layers. If any of these
change, the saved weights from notebook 02 must be retrained.
"""
import re

import torch
import torch.nn as nn

PAD_ID = 0  # padding, ignored by the model
UNK_ID = 1  # word not in the training vocabulary


def tokenize(text):
    """Lowercase words, numbers and contractions: "8GB RAM isn't enough!" -> 8gb, ram, isn't, enough."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())


def encode_tokens(tokens, vocabulary, max_length):
    """Words -> ids, cut to max_length, padded with <pad>."""
    token_ids = [vocabulary.get(token, UNK_ID) for token in tokens[:max_length]]
    token_ids += [PAD_ID] * (max_length - len(token_ids))
    return token_ids


class ATAELSTM(nn.Module):
    """Aspect-aware attention BiLSTM for aspect polarity classification.

    1. Embed the sentence words and the aspect (GloVe start, fine-tuned during training).
    2. Average the embeddings of a multi-word aspect into one vector.
    3. Append that aspect vector to every word and run a bidirectional LSTM.
    4. Score each LSTM state against the aspect, so words relevant to the aspect weigh more.
    5. Classify the attention-weighted sentence summary.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, embedding_weights=None, dropout=0.30):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        # Training passes GloVe vectors here. At prediction time this is left out,
        # because load_state_dict fills in the trained values anyway.
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,  # word vector + aspect vector
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        lstm_output_dim = hidden_dim * 2  # forward + backward
        self.attention_projection = nn.Linear(lstm_output_dim + embedding_dim, lstm_output_dim)
        self.attention_score = nn.Linear(lstm_output_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, sentence_ids, aspect_ids):
        sentence_mask = sentence_ids.ne(PAD_ID)  # True for real words

        sentence_embeddings = self.embedding(sentence_ids)  # [batch, words, dim]
        aspect_embeddings = self.embedding(aspect_ids)      # [batch, aspect words, dim]

        # Average only the real (non-pad) aspect words.
        aspect_mask = aspect_ids.ne(PAD_ID).unsqueeze(-1)
        aspect_vector = (aspect_embeddings * aspect_mask).sum(dim=1) / aspect_mask.sum(dim=1).clamp(min=1)

        # Copy the aspect vector next to every word position.
        expanded_aspect = aspect_vector.unsqueeze(1).expand(-1, sentence_embeddings.size(1), -1)

        lstm_output, _ = self.lstm(torch.cat([sentence_embeddings, expanded_aspect], dim=-1))

        attention_hidden = torch.tanh(
            self.attention_projection(torch.cat([lstm_output, expanded_aspect], dim=-1))
        )
        attention_scores = self.attention_score(attention_hidden).squeeze(-1)

        # Padding must never receive attention.
        attention_scores = attention_scores.masked_fill(~sentence_mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum of the LSTM states = one summary vector per sentence.
        summary = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        logits = self.classifier(self.dropout(summary))  # raw scores; the loss applies softmax
        return logits, attention_weights


class AtaePredictor:
    """Loads the checkpoint saved by notebook 02 and scores (text, aspect) pairs.

    The checkpoint stores the vocabulary, label order and layer sizes next to the
    weights, so nothing has to be retyped to rebuild the model.
    """

    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vocab = checkpoint["vocab"]
        self.label_to_id = checkpoint["label_to_id"]
        self.config = checkpoint["config"]
        self.best_validation_macro_f1 = checkpoint.get("best_validation_macro_f1")
        self.id_to_label = {label_id: label for label, label_id in self.label_to_id.items()}

        self.model = ATAELSTM(
            vocab_size=len(self.vocab),
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_classes=len(self.label_to_id),
            dropout=self.config["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  # switches dropout off

    def predict(self, text, aspect_term):
        """Return polarity, its probability, every class probability and per-word attention."""
        sentence_ids = encode_tokens(tokenize(text), self.vocab, self.config["max_sentence_len"])
        aspect_ids = encode_tokens(tokenize(aspect_term), self.vocab, self.config["max_aspect_len"])

        with torch.no_grad():
            logits, attention_weights = self.model(
                torch.tensor([sentence_ids], dtype=torch.long).to(self.device),
                torch.tensor([aspect_ids], dtype=torch.long).to(self.device),
            )
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_id = torch.argmax(probabilities).item()

        return {
            "predicted_polarity": self.id_to_label[predicted_id],
            "confidence": round(float(probabilities[predicted_id]), 4),
            "all_class_scores": {
                self.id_to_label[i]: round(float(probabilities[i]), 4) for i in range(len(self.label_to_id))
            },
            "attention_weights": attention_weights[0].detach().cpu().numpy(),
        }

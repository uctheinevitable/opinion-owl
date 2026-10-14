"""ATAE-LSTM building blocks shared by notebook 02 and inference.py.

inference.py is used by notebooks 04 and 06 and by the app. Everything lives in one place so the
training notebook and the prediction code can never disagree about the tokenizer, the padding
rules or the model layers. If any of these change, the saved weights from notebook 02 must be
retrained (bump TOKENIZER_VERSION when tokenize() changes; old checkpoints are then refused).
"""
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PAD_ID = 0  # padding, ignored by the model
UNK_ID = 1  # word not in the training vocabulary

# Bump this whenever tokenize() changes. A checkpoint trained with another version is refused,
# because its vocabulary would no longer match the way new text is split into words.
TOKENIZER_VERSION = 2

_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u00b4": "'", "`": "'"})

# Words, numbers and contractions split the way the GloVe vectors were built:
#   isn't -> is + n't, don't -> do + n't, it's -> it + 's, I'm -> i + 'm
_TOKEN = re.compile(r"[a-z0-9]+(?=n't\b)|n't\b|'(?:s|re|ve|ll|d|m)\b|[a-z0-9]+")


def tokenize(text):
    """Lowercase words and numbers, with contractions split like GloVe does.

    "8GB RAM isn't enough!" -> 8gb, ram, is, n't, enough
    Curly apostrophes (don\u2019t) are treated like straight ones, so negations survive in
    text copied from web pages.
    """
    return _TOKEN.findall(str(text).lower().translate(_APOSTROPHES))


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

    The checkpoint stores the vocabulary, label order and layer sizes next to the weights, so
    nothing has to be retyped to rebuild the model. If notebook 02 trained several models (one per
    seed), all of them are loaded and their probabilities are averaged.
    """

    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # weights_only=True: the file holds only tensors, numbers, text and plain dicts.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.vocab = checkpoint["vocab"]
        self.label_to_id = checkpoint["label_to_id"]
        self.config = checkpoint["config"]
        self.best_validation_macro_f1 = checkpoint.get("best_validation_macro_f1")
        self.selection_metric = checkpoint.get("selection_metric")
        self.id_to_label = {label_id: label for label, label_id in self.label_to_id.items()}

        saved_version = self.config.get("tokenizer_version", 1)
        if saved_version != TOKENIZER_VERSION:
            raise ValueError(
                f"{Path(checkpoint_path).name} was trained with tokenizer version {saved_version}, "
                f"but atae.py uses version {TOKENIZER_VERSION}. Re-run notebook 02 to retrain the ATAE-LSTM."
            )

        # One state dict per trained model. Older single-model files only have "model_state_dict".
        state_dicts = checkpoint.get("model_state_dicts") or [checkpoint["model_state_dict"]]
        self.models = []
        for state_dict in state_dicts:
            model = ATAELSTM(
                vocab_size=len(self.vocab),
                embedding_dim=self.config["embedding_dim"],
                hidden_dim=self.config["hidden_dim"],
                num_classes=len(self.label_to_id),
                dropout=self.config["dropout"],
            ).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()  # switches dropout off
            self.models.append(model)

        self.model = self.models[0]   # the first model, for code that expects a single one
        self.n_models = len(self.models)

    def predict_batch(self, texts, aspect_terms, return_attention=False):
        """Score many (text, aspect) pairs at once. Returns one result dict per pair, in order."""
        if len(texts) != len(aspect_terms):
            raise ValueError("texts and aspect_terms must have the same length")
        if not texts:
            return []

        sentence_ids = torch.tensor(
            [encode_tokens(tokenize(t), self.vocab, self.config["max_sentence_len"]) for t in texts],
            dtype=torch.long,
        ).to(self.device)
        aspect_ids = torch.tensor(
            [encode_tokens(tokenize(a), self.vocab, self.config["max_aspect_len"]) for a in aspect_terms],
            dtype=torch.long,
        ).to(self.device)

        probabilities, attentions = [], []
        with torch.no_grad():
            for model in self.models:
                logits, attention_weights = model(sentence_ids, aspect_ids)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
                attentions.append(attention_weights.cpu().numpy())

        # With several models the answer is the average of their probabilities.
        probabilities = np.mean(probabilities, axis=0)
        attention = np.mean(attentions, axis=0) if return_attention else None

        results = []
        for row, probs in enumerate(probabilities):
            predicted_id = int(probs.argmax())
            result = {
                "predicted_polarity": self.id_to_label[predicted_id],
                "confidence": round(float(probs[predicted_id]), 4),
                "all_class_scores": {
                    self.id_to_label[i]: round(float(probs[i]), 4) for i in range(len(self.label_to_id))
                },
            }
            if return_attention:
                result["attention_weights"] = attention[row]
            results.append(result)
        return results

    def predict(self, text, aspect_term):
        """Return polarity, its probability, every class probability and per-word attention."""
        return self.predict_batch([text], [aspect_term], return_attention=True)[0]

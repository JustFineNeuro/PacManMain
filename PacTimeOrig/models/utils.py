import torch
import torch.nn as nn
import numpy as np


def eval_single_trial_pad_rnn(model, X_data, y_data=None, outcard=2):
    X_trial = torch.tensor(X_data)  # Your input data
    X_trial = X_trial.float()  # Converts tensor to torch.float32
    seq_length = torch.tensor([X_trial.shape[0]])

    X_trial = X_trial.unsqueeze(0)  # Shape: (1, seq_len, input_size)
    # Move data to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_trial = X_trial.to(device)
    seq_length = seq_length.to(device)
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Run the model
        if outcard == 2:
            outputs, hidden_states = model(X_trial, seq_length)

            # Remove batch dimension if needed
            outputs = outputs.squeeze(0)  # Shape: (seq_len, output_size)
            hidden_states = hidden_states.squeeze(0)  # Shape: (seq_len, hidden_size)
            latents = []
        elif outcard == 3:
            outputs, hidden_states,latents = model(X_trial, seq_length)

            # Remove batch dimension if needed
            outputs = outputs.squeeze(0)  # Shape: (seq_len, output_size)
            hidden_states = hidden_states.squeeze(0)  # Shape: (seq_len, hidden_size)
            latents = latents.squeeze(0)
    return outputs, hidden_states,latents

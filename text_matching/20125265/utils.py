import torch
import torch.nn as nn
import time
from tqdm import tqdm

def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def validate(model, dataloader, criterion, device):
    model.eval()
    #device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for batch in dataloader:
            q1 = batch.sentence1.to(device)
            q2 = batch.sentence2.to(device)
            # mask1 = (q1 == 1)
            # mask2 = (q2 == 1)
            labels = batch.label.to(device)
            logits, probs = model(q1, q2)
            #logits = model(q1, q2, mask1, mask2)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)
            #running_accuracy += correct_predictions(logits, labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy

def test(model, dataloader, device):
    model.eval()
    #device = model.device
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            q1 = batch.sentence1.to(device)
            q2 = batch.sentence2.to(device)
            mask1 = (q1 == 1)
            mask2 = (q2 == 1)
            _, probs = model(q1, q2)
            #probs = model(q1, q2, mask1, mask2)
            _, out_classes = probs.max(dim=1)
            all_labels.extend(out_classes.cpu().numpy())
    return all_labels

def train(model, dataloader, optimizer, criterion, max_gradient_norm, device):
    model.train()
    #device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    correct_preds = 0

    tqdm_iter = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_iter):
        q1 = batch.sentence1.to(device)
        q2 = batch.sentence2.to(device)
        # mask1 = (q1 == 1)
        # mask2 = (q2 == 1)
        # mask1 = mask1.to(device)
        # mask2 = mask2.to(device)
        labels = batch.label.to(device)
        optimizer.zero_grad()
        logits, probs = model(q1, q2)
        #logits = model(q1, q2, mask1, mask2)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        running_loss = loss.item()
        correct_preds += correct_predictions(probs, labels)
        #correct_preds += correct_predictions(logits, labels)
        description = "Avg Batch loss: {:.4f}".format(running_loss)
        tqdm_iter.set_description(description)
        del batch
        del q1
        del q2
        del labels
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy
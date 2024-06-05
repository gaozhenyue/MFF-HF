import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc


# function to train the model
def train_model(model, traindata, loss_func, optimizer):
    print('Training...')
    model.train()
    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(traindata):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. loss {:>5,}'.format(step, len(traindata), loss))

        if torch.cuda.is_available():
            device = torch.device("cuda")
            # push the batch to gpu
            batch = [r.to(device) for r in batch]

        cat, num, img, labels = batch
        # clear previously calculated gradients
        model.zero_grad()
        # get model predictions for the current batch
        logits = model(cat, num, img)
        loss = loss_func(logits.reshape(-1, 1), labels.float().reshape(-1, 1))
        # add on to the total loss
        total_loss = total_loss + loss
        # backward pass to calculate the gradients
        loss.requires_grad_(True)
        loss.backward()
        #         # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        #         scheduler.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = logits.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)

        torch.cuda.empty_cache()

    # compute the training loss of the epoch
    avg_loss = total_loss / len(traindata)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate_model(model, valdata, loss_func):
    print("\nEvaluating...")
    t0 = time.time()

    model.eval()  # deactivate dropout layers
    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(valdata):
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            #             elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(valdata)))

        if torch.cuda.is_available():
            device = torch.device("cuda")
            # push the batch to gpu
            batch = [t.to(device) for t in batch]

        cat, num, img, labels = batch

        # deactivate autograd
        with torch.no_grad():  # Dont store any previous computations, thus freeing GPU space

            # model predictions
            logits = model(cat, num, img)
            loss = loss_func(logits.reshape(-1, 1), labels.float().reshape(-1, 1))
            #             loss, logits = output['loss'], output['logits']

            total_loss = total_loss + loss
            preds = logits.detach().cpu().numpy()
            total_preds.append(preds)

        torch.cuda.empty_cache()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(valdata)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


def test_model(model, testdata):
    print('\nTest Set...')
    test_preds = []
    print('Total batches:', len(testdata))

    # Put the model in evaluation mode.
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Reset the total loss for this epoch.
    total_val_loss = 0

    for j, test_batch in enumerate(testdata):

        inference_status = 'Batch ' + str(j + 1)

        print(inference_status, end='\r')

        b_cat = test_batch[0].to(device)
        b_num = test_batch[1].to(device)
        b_img = test_batch[2].to(device)
        b_test_y = test_batch[3].to(device)

        logits = model(b_cat, b_num, b_img)

        # Move preds to the CPU
        val_preds = logits.detach().cpu().numpy()

        # Stack the predictions.
        if j == 0:  # first batch
            stacked_val_preds = val_preds

        else:
            stacked_val_preds = np.vstack((stacked_val_preds, val_preds))

    test_preds.append(stacked_val_preds)
    print('\nPrediction complete.')

    return test_preds


def plot_result(true_y, test_preds):

    fpr, tpr, thresholds = roc_curve(true_y, test_preds)
    auc_score = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='MFF-HF (AUC={:.3f})'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
#     plt.title('ROC curve')
    plt.legend(loc='best')

    plt.show()



# Python Libraries
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path().resolve()))

# Project Imports
from utils import constant, helper
from data import loader
from model import architecture

# External Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Datasets
training_dataset = loader.YelpLoader(path_to_csv=os.path.join(constant.DATASET_FOLDER, constant.TRAIN_FILE))
validation_dataset = loader.YelpLoader(path_to_csv=os.path.join(constant.DATASET_FOLDER, constant.VALID_FILE))
test_dataset = loader.YelpLoader(path_to_csv=os.path.join(constant.DATASET_FOLDER, constant.TEST_FILE))

# DataLoaders
training_dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("Length of Training Dataset is :{0}".format(len(training_dataloader)))
print("Length of Valid Dataset is :{0}".format(len(validation_dataloader)))
print("Length of Test Dataset is :{0}".format(len(test_dataloader)))

# Model
HAN = architecture.HanModel(
    input_dim=256,
    hidden_dim=32,
    bidirectional=True,
    layers=2,
    padding_idx=constant.PADDING_TOK_IX,
    class_size=len(training_dataset.ctgry),
    randomize_init_hidden=True
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(HAN.parameters(), lr=0.005, momentum=0.9)

# Run Test Dataset before Training
print("Predicting on Test Set")
true_test_labels, pred_test_labels = helper.model_test(HAN, test_dataloader)
print(classification_report(y_true=true_test_labels, y_pred=pred_test_labels))

start_time = time.time()

epoch_training_loss_collect = []
epoch_validation_loss_collect = []
lowest_validation_loss = 100000
best_HAN = None

for epoch in range(constant.EPOCHS):
    # Training Run
    HAN, criterion, optimizer, epoch_training_loss = helper.model_train(HAN, criterion, optimizer, training_dataloader)
    # Validation Run
    epoch_validation_loss = helper.model_evaluate(HAN, criterion, validation_dataloader)

    epoch_training_loss_collect.append(epoch_training_loss)
    epoch_validation_loss_collect.append(epoch_validation_loss)

    # Print Results
    print("EPOCH: {0}\t TRAIN_LOSS : {1}\t VALID_LOSS : {2} ".format(epoch, epoch_training_loss, epoch_validation_loss))

    if epoch_validation_loss < lowest_validation_loss:
        lowest_validation_loss = epoch_validation_loss
        best_HAN = HAN
        torch.save(best_HAN.state_dict(), 'HAN.pth')
        print("\tLowest Validation Loss! -> Model Saved!")

stop_time = time.time()
time_taken = stop_time - start_time
print("\n\nTraining Complete!\t Total Time: {0}\n\n".format(time_taken))

# Run Test Dataset after Training
print("Predicting on Test Set")
true_test_labels, pred_test_labels = helper.model_test(best_HAN, test_dataloader)
print(classification_report(y_true=true_test_labels, y_pred=pred_test_labels))

print("Writing the Loss Graph")
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_training_loss_collect, label='training')
plt.plot(epoch_validation_loss_collect, label='validation')
plt.legend(loc="upper right")
plt.savefig("loss.png")

print("Writing the Confusion Matrix")
loss_fig = helper.plot_confusion(true_test_labels, pred_test_labels, training_dataset.ctgry)
loss_fig.savefig("confusion.png")

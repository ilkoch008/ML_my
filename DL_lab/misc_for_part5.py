from keras.utils import to_categorical
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train_model(model, train_loader, val_loader, loss_fn, opt, n_epochs: int, device=device, decr_coeff = 1, decr_ep_n = 10):
    '''
    model: нейросеть для обучения,
    train_loader, val_loader: загрузчики данных
    loss_fn: целевая метрика (которую будем оптимизировать)
    opt: оптимизатор (обновляет веса нейросети)
    n_epochs: кол-во эпох, полных проходов датасета
    '''
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    counter = 0
    for epoch in range(n_epochs):
        ep_train_loss = []
        ep_val_loss = []
        ep_train_accuracy = []
        ep_val_accuracy = []
        start_time = time.time()
        counter += 1
        if counter == decr_ep_n:
            counter = 0
            for g in opt.param_groups:
                g['lr'] = g['lr'] / decr_coeff

        model.train(True) # enable dropout / batch_norm training behavior
        for X_batch, y_batch in train_loader:
            opt.zero_grad()
            # move data to target device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
            out = model(X_batch)
            loss = loss_fn(out, y_batch)
            loss.backward()
            opt.step()
            ep_train_loss.append(loss.item())
            ep_train_accuracy.append(torch_accuracy_score(y_batch, model(X_batch)))

        model.train(False) # disable dropout / use averages for batch_norm
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # move data to target device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
                out = model(X_batch)
                loss = loss_fn(out, y_batch)

                # compute predictions
                ep_val_loss.append(loss.item())
                y_pred = out.max(dim=1)[1]
                ep_val_accuracy.append(np.sum(y_batch.cpu().numpy() == y_pred.cpu().numpy().astype(float))/ len(y_batch.cpu()))
        # print the results for this epoch:
        print(f'Epoch {epoch + 1} of {n_epochs} took {time.time() - start_time:.3f}s')

        train_loss.append(np.mean(ep_train_loss))
        train_accuracy.append(np.mean(ep_train_accuracy))

        val_loss.append(np.mean(ep_val_loss))
        val_accuracy.append(np.mean(ep_val_accuracy))
        if (epoch+1) % 10 == 0:
            print(f"\t training loss: {train_loss[-1]:.6f}")
            print(f"\t training accuracy: {train_accuracy[-1]:.6f}")
            print(f"\t validation loss: {val_loss[-1]:.6f}")
            print(f"\t validation accuracy: {val_accuracy[-1]:.3f}")

    return train_loss, train_accuracy, val_loss, val_accuracy

def torch_accuracy_score(y_true, predictions):
    y_pred = predictions.max(dim=1)[1]
    return np.mean(np.array(y_true.cpu() == y_pred.cpu()))

def plot_training(tr_loss, tr_acc, val_loss, val_acc):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].set_title('Accuracy')
    axes[0].plot(tr_acc, label='Training accuracy')
    axes[0].plot(val_acc, label='Validation accuracy')
    axes[0].set_xlabel('n epoch')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title('Loss')
    axes[1].plot(tr_loss, label='Training loss')
    axes[1].plot(val_loss, label='Validation loss')
    axes[1].set_xlabel('n epoch')
    axes[1].grid(True)
    axes[1].legend()

    plt.show()

def model_test_score(model, test_loader):
    test_accuracy = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            test_accuracy.append(torch_accuracy_score(y_batch.flatten(), predictions))

    return np.mean(test_accuracy)
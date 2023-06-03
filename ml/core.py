'''Where the core operation for the machine learning part are stored.
'''
import numpy as np
import torch

from sfw.constraints import create_simplex_constraints

def train(model, criterion, optimizer, scheduler, train_loader, n_epochs, device,
         alpha_1, alpha_2):
    '''Function to train the model in place

    Parameters
    ----------
    model : nn.Module subclass
        Class implementing the model to train
    criterion : nn.loss
        Some function implementing the loss
    optimizer : torch.optim
        Optimizer for the training procedure
    scheduler : torch.optim
        Schedule the learning rate behavour
    train_loader : torch.utils.data.DataLoader
        Loader for the training data
    n_epochs : int
        Number of epochs
    device : str
        Device to send the computations
    epochs_to_prune : array
        At which epochs pruning is done
    alpha_1 : float
        Parameter setting the L1 regularization strength
    alpha_2 : float
        Parameter setting the L2 regularization strength
    '''
    mean_train_loss = []

    for epoch in range(1, n_epochs+1):
        model.train()
        print('= Starting epoch ', epoch, '/', n_epochs)

        summed_train_loss = np.array([])

        # Train
        for batch_index, (vv, t, batch_in, batch_out) in enumerate(train_loader):

            constraints = create_simplex_constraints(model)

            X = batch_in.float().to(device)
            y = batch_out.float().to(device)

            # set gradients to zero to avoid using old data
            optimizer.zero_grad()

            # apply the model
            recon_y = model.forward(t=t, x=X)
            
            # calculate the loss
            loss = criterion(recon_y, y)
            # sum to the loss per epoch
            summed_train_loss = np.append(summed_train_loss, loss.item())

            # weights regularization : Elastic net
            if len(alpha_1) != 0:
                loss += alpha_1[0]*torch.norm(model.MLP.v_x, 1)
                loss += alpha_1[0]*torch.norm(model.MLP.v_y, 1)
                loss += alpha_1[1]*torch.norm(model.MLP.omega, 1)

            if len(alpha_2) != 0:
                loss += alpha_2[0]*torch.norm(model.MLP.v_x)
                loss += alpha_2[0]*torch.norm(model.MLP.v_y)
                loss += alpha_2[1]*torch.norm(model.MLP.omega)

            # backpropagate = calculate derivatives
            loss.backward(retain_graph=True)

            # update lr
            optimizer.step(constraints)

        scheduler.step()

        print('=== Mean train loss: {:.12f}'.format(summed_train_loss.mean()))
        print('=== lr: {:.5f}'.format(scheduler.get_last_lr()[0]))
        mean_train_loss.append(summed_train_loss.mean())

    print('=== Mean train loss: {:.12f}'.format(summed_train_loss.mean()))
    print('=== lr: {:.5f}'.format(scheduler.get_last_lr()[0]))

    return mean_train_loss

def eval(model, criterion, eval_loader, device):
    '''Function to evaluate the model
    '''

    model.eval()
    summed_eval_loss = np.array([])

    for batch_index, (vv, t, batch_in, batch_out) in enumerate(eval_loader):

        X = batch_in.float().to(device)
        y = batch_out.float().to(device)

        # apply the model
        recon_y = model.forward(t=t, x=X)

        # calculate the loss
        loss = criterion(recon_y, y)

        summed_eval_loss = np.append(summed_eval_loss, loss.item())

    print('=== Test set loss:   {:.12f}'.format(summed_eval_loss.mean()))
    return {'loss': summed_eval_loss}

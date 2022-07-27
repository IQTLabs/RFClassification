## Functions for training and testing Neural Networks
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import time
from sklearn.metrics import f1_score

def runkfoldcv(model, dataset, device, k_folds, batch_size, learning_rate, num_epochs, momentum, l2reg):
    num_classes = model.num_classes
    if num_classes == 2:
        f1type = 'binary'
    else:
        f1type = 'weighted' # is this the best choice
    
    # For fold results
    results = {}
    runtimes = np.zeros(k_folds)
    f1s = np.zeros(k_folds)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=batch_size, sampler=test_subsampler)

        # Init the neural network
        model = model.to(device)
    #     network.apply(reset_weights)

        criterion = nn.CrossEntropyLoss()

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2reg, momentum = momentum)  

        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):
            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
                # Get inputs
                inputs, targets = data
#                 inputs = inputs.float()
                targets= targets.type(torch.long)

                # Move tensors to configured device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss            
                loss = criterion(outputs, targets)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 50 == 49:
                    print('    Loss after mini-batch %5d: %.5f' %
                          (i + 1, current_loss / 50))
                    current_loss = 0.0
    #         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        # Process is complete.
    #     print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')
        print('----------------')

        # Saving the model
    #     save_path = f'./model-fold-{fold}.pth'
    #     torch.save(network.state_dict(), save_path)

        # Evaluation for this fold
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            runtimes_thisfold = []
            f1s_thisfold = []
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Generate outputs
                n_instances = len(inputs)
                ys = torch.empty(n_instances)
                ys = ys.to(device)

                for i in range(n_instances):
                    instance = inputs[i]
                    instance = instance.float()
                    instance = torch.unsqueeze(instance, 0)
                    start = time.time()
                    starter.record()
                    yi = model(instance)
                    _,pred = torch.max(yi,1)
                    ender.record()

                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender) #miliseconds

                    runtimes_thisfold.append(curr_time*1e-3)
                    ys[i] = pred


                # Set total and correct
                total += targets.size(0)
                correct += (ys == targets).sum().item()
                f1i = f1_score(ys.cpu().numpy(), targets.cpu().numpy(), average=f1type)
                f1s_thisfold.append(f1i)

            mean_runtime = np.mean(np.array(runtimes_thisfold))
            mean_f1 = np.mean(np.array(f1s_thisfold))

        # Summarize and print results
        results[fold] = 100.0 * (correct / total)
        runtimes[fold] = mean_runtime
        f1s[fold] = mean_f1
        print('Accuracy for fold %d: %.2f %%' % (fold, 100.0 * correct / total))
        print('F1 for fold %d: %.2f ' % (fold, mean_f1))
        print('Runtime for fold %d: %.4f s' % (fold, mean_runtime))
        print('--------------------------------')

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    mean_acc = sum/len(results.items())
    mean_f1s = np.mean(f1s)
    mean_runtime = np.mean(runtimes)
    print(f'Average Accuracy: {mean_acc} %')
    print(f'Average F1: {mean_f1s}')
    print(f'Average Runtime: {mean_runtime} s')
    
    return model
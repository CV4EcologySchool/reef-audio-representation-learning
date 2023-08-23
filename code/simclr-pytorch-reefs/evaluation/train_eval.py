'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    2022 Benjamin Kellenberger
'''

import os
import argparse
import yaml
import glob
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD

# let's import our own classes and functions!
from util_eval import init_seed
from my_custom_dataset_eval import CTDataset_train, CTDataset_test ###??
from model_eval import CustomResNet18, CustomResNet50, SimClrPytorchResNet50, PAWSResNet50 
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import class_weight
import numpy as np
import wandb

from sklearn.metrics import f1_score



def create_dataloader(cfg, split='test_data', transform=False, train_percent=None, train_test=None):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    #dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class
    if train_test == 'train':
        dataset_instance = CTDataset_train(cfg, split=split, transform=transform, train_percent=train_percent)
    elif train_test == 'test':
        dataset_instance = CTDataset_test(cfg, split=split, transform=False, train_percent=train_percent)

    device = cfg['device']

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'], ## is there any drop-last going on? - ben
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    
    ### compute weights for class balancing
    classes_for_weighting = []
    for data, labels in dataLoader:
        classes_for_weighting.extend(list(labels))  

    class_weights=class_weight.compute_class_weight('balanced',classes = np.unique(classes_for_weighting),y = np.array(classes_for_weighting))
    class_weights = class_weights/np.sum(class_weights)
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)

    return dataLoader, class_weights
    


def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    #model_instance = CustomResNet18(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    model_instance = CustomResNet50(cfg['num_classes'])         # create an object instance of our CustomResNet18 class
    #model_instance = SimClrPytorchResNet50(cfg['num_classes'])
    #model_instance = PAWSResNet50(cfg['num_classes'])
    
    # load latest model state
#    model_states = glob.glob('model_states/*.pt')
#     if len(model_states):
#         # at least one save state found; get latest
#         model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
#         start_epoch = max(model_epochs)

#         # load state dict and apply weights to model
#         print(f'Resuming from epoch {start_epoch}')
#         state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
#         model_instance.load_state_dict(state['model'])

#     else:
#         # no save state found; start anew
#         print('Starting new model')
#         start_epoch = 0
    start_epoch = 0
    return model_instance, start_epoch

def load_pretrained_weights(cfg, model):
    custom_weights = cfg['starting_weights']

    state = torch.load(open(custom_weights, 'rb'), map_location='cpu')

    if 'state_dict' in state.keys():
        pretrained_dict = state['state_dict']
    else:
        pretrained_dict = state['model']

    ## only update the weights of layers with the same names and also don't update the last layer because of size mismatch between num classes
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ['classifier.weight', 'classifier.bias']}

    #model_dict.update(pretrained_dict)
    log = model.load_state_dict(pretrained_dict, strict=False)
    assert log.missing_keys == ['classifier.weight', 'classifier.bias']

    #### finetuning only last layers ###############
    if cfg['finetune'] == True:
        for name, param in model.named_parameters():
            if name not in ['classifier.weight', 'classifier.bias']:
                param.requires_grad = False

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # classifier.weight, classifier.bias

    ## we need to copy everything except the last layer
    #for key in state['model'].keys():
        #if not(key == 'classifier.weight' or key== 'classifier.bias'):
        ###### Tarun : LESSON LEARNT : THE LINE BELOW DOES NOT WORK FOR SOME REASON ..you must do it like mentioned in the most liked answer here https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2 ###
        #model.state_dict()[key] = state['model'][key]
    #model.load_state_dict(state['model'])
    ##import ipdb;ipdb.set_trace()
    return model


def load_pretrained_weights_PAWS(cfg, model):
    checkpoint = torch.load(cfg['starting_weights'], map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['encoder'].items()}
    new_dict = model.state_dict()
    
    for k, v in model.state_dict().items():
        k_ = k.replace('convnet.','')
        if k_ not in pretrained_dict:
            print (f'key "{k_}" could not be found in loaded state dict')
        elif pretrained_dict[k_].shape != v.shape:
            print (f'key "{k_}" is of different shape in model and loaded state dict')
            
        else:
            #pretrained_dict[k] = v
            new_dict[k] = pretrained_dict[k_]
            
    msg = model.load_state_dict(new_dict, strict=False)
    print (f'loaded pretrained model with msg: {msg}')
    print (f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} ')
    del checkpoint
    return model


def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs('model_states_i2map_simclr', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'model_states_i2map_simclr/10p_{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = 'model_states/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

            

def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer



def train(cfg, dataLoader, model, optimizer, class_weights_train):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss(class_weights_train)
    #criterion = nn.CrossEntropyLoss()
    
    # running averages
    loss_total, oa_total, f1 = 0.0, 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    all_predicted_labels = []
    all_ground_truth_labels = []

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        all_predicted_labels.extend(pred_label.cpu()) # this moves all predicted labels to a list above
        all_ground_truth_labels.extend(labels.cpu())
            
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        f1 += f1_score(labels.cpu(), pred_label.cpu())

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)
    f1 /= len(dataLoader)

    bac = balanced_accuracy_score(all_ground_truth_labels, all_predicted_labels)

    return loss_total, oa_total, bac, f1



def validate(cfg, dataLoader, model, class_weights_val):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    #criterion = nn.CrossEntropyLoss(class_weights_val)   # we still need a criterion to calculate the validation loss
    criterion = nn.CrossEntropyLoss()   # we still need a criterion to calculate the validation loss
    
    # running averages
    loss_total, oa_total, f1 = 0.0, 0.0, 0,0     # for now, we just log the loss and overall accuracy (OA)

    all_predicted_labels = []
    all_ground_truth_labels = []

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            all_predicted_labels.extend(pred_label.cpu()) # this moves all predicted labels to a list above
            all_ground_truth_labels.extend(labels.cpu())
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            f1 += f1_score(labels.cpu(), pred_label.cpu())

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)
    f1 /= len(dataLoader) 
    bac = balanced_accuracy_score(all_ground_truth_labels, all_predicted_labels)

    return loss_total, oa_total, bac, f1



def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))   ### DO NOT REMOVE, as used on my_custom_dataset.py

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'


    wandb.init(
    # set the wandb project where this run will be logged
    project="midwater i2map data",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": cfg['learning_rate'],
    "architecture": "resnet 50",
    "dataset": "15 percent labeled",
    "epochs": cfg['num_epochs'],
    "weight_decay": cfg['weight_decay'],
    "batch_size": cfg['batch_size']
    })


    # initialize data loaders for training and validation set
    dl_train, class_weights_train = create_dataloader(cfg, split='test_data', transform=False, train_percent = cfg['train_percent'], train_test = 'train')
    dl_val, class_weights_val = create_dataloader(cfg, split='test_data', transform=False, train_percent = cfg['train_percent'], train_test = 'test')


    # initialize model
    model, current_epoch = load_model(cfg)
    
    if cfg['starting_weights'] != 'None':
        starting_weights = cfg['starting_weights']
        print (f'loading custom starting weights: {starting_weights}')
        #model = load_pretrained_weights(cfg, model)
        model = load_pretrained_weights_PAWS(cfg, model)
        
    else:
        print ('starting weights are imagenet weights')

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train, bac_train, f1_train = train(cfg, dl_train, model, optim, class_weights_train)
        loss_val, oa_val, bac_val, f1_val = validate(cfg, dl_val, model, class_weights_val)

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'bac_train':bac_train,
            'oa_val': oa_val,
            'bac_val':bac_val,
            'f1_train': f1_train,
            'f1_val:': f1_val


        }
        wandb.log(stats)
        #wandb.log({'loss_train': loss_train,
        #     'loss_val': loss_val,
        #     'oa_train': oa_train,
        #     'bac_train':bac_train,
        #     'oa_val': oa_val,
        #     'bac_val':bac_val
        #     })
        
        if current_epoch % 40 ==0:
            save_model(cfg, current_epoch, model, stats)
    

    wandb.finish()
        


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
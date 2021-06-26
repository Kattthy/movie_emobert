import os
import time
import numpy as np
from opts.train_opts import TrainOptions
from data import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, is_save=False, phase='test'):
    model.eval()
    total_pred = []
    total_label = []
    
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)
    
    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    model.train()
    
    # save test results
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    return acc, uar, f1, cm

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))

if __name__ == '__main__':
    opt = TrainOptions().parse()                        # get training options
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo)) # get logger path
    if not os.path.exists(logger_path):                 # make sure logger path exists
        os.mkdir(logger_path)

    result_recorder = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result.tsv'), total_cv=1) # init result recoreder
    suffix = '_'.join([opt.model, opt.dataset_mode])    # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger
    if opt.has_test:                                    # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])  
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
    
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    best_eval_epoch = 11 #这里需要自己手动设置


    # test
    #if opt.has_test:
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    model.load_networks(best_eval_epoch)
    _ = eval(model, val_dataset, is_save=True, phase='val')
    acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
    logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
    logger.info('\n{}'.format(cm))
    result_recorder.write_result_to_tsv({
        'acc': acc,
        'uar': uar,
        'f1': f1
    }, cvNo=opt.cvNo)

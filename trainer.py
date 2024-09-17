import os
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from torch import optim
from torch.utils.data import DataLoader

from utils import *
from Dataset_v12 import MD_detection
from MD_transform import *

import matplotlib.pyplot as plt
import numpy as np

class BaseTrainer:
    def __init__(self,args,logger):
        self.logger = logger
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

        # Define Dataloader
        self.s_train_dataloader, self.s_val_dataloader = define_data_loader(args.source_root, args.batch_size)
        self.t_train_dataloader, self.t_test_dataloader = define_data_loader(args.target_root, args.batch_size)

        # Delete this line if you want to use the source dataset for validation
        #self.s_val_dataloader = self.t_val_dataloader
        
        generalization_test = MD_detection(args.generalization_test_root)
        self.generalization_dataloader = DataLoader(generalization_test,batch_size=1,shuffle=False)
        
        # Define model and loss fn
        self.model = define_model(args)
        self.model = self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    
    def train(self):
        args = self.args
        writer = define_writer(args)
        model = self.model
        loss_fn = self.loss_fn
        opt = define_opt(model,args)
        best_acc = 0

        for epoch in trange(self.args.num_epochs, desc='Training Loop'):
            train_epoch_loss, train_acc = self._train_one_epoch(self.s_train_dataloader,model,loss_fn,opt)
            writer.add_scalar("Loss/train_total", train_epoch_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            val_epoch_loss, val_acc = self._validation(self.s_val_dataloader, model, loss_fn)
            writer.add_scalar("Loss/val", val_epoch_loss, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            
            if val_acc > best_acc:
                trials = 0
                best_acc = val_acc
                save_name = "earlystop" + args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
                save_pth = os.path.join('saved_model_pth', save_name)
                torch.save(model.state_dict(),save_pth)
                self.logger.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= args.patience:
                    self.logger.info(f'Early stopping on epoch {epoch}')
                    break
            
        save_name = "last_epoch" + args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
        save_pth = os.path.join('saved_model_pth', save_name)
        torch.save(model.state_dict(),save_pth)
        self.logger.info(f'last model saved with accuracy: {val_acc:2.2%}')
        self.logger.info(f'Transfer result: {best_acc:.4f}')
    
    def _train_one_epoch(self,dataloader,model,loss_fn,opt):
        size = len(dataloader.dataset)
        train_epoch_loss = AverageMeter()
        correct = 0
        torch.manual_seed(42)
        model.train()
        source_iter = iter(dataloader)
        #for batch in dataloader:
        for i in range(len(dataloader)):
            batch = next(source_iter)
            x = batch['data']
            y = batch['label']
            x, y = x.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
            # Compute prediction error
            feature,out = model(x)
            pred = F.log_softmax(out, dim=1).argmax(dim=1)
            loss = loss_fn(out, y)
            train_epoch_loss.update(loss.item())
            correct += (pred == y.data.argmax(dim=1)).sum().item()
            # Backpropagation
            opt.zero_grad() # To reset the gradients of model parameters.
            loss.backward()
            opt.step() # To adjust the parameters by gradients collected in the backward pass
        acc = correct/size
        return train_epoch_loss.avg, acc
    
    def _validation(self,dataloader,model,loss_fn):
        size = len(dataloader.dataset)
        model.eval()
        correct = 0
        test_loss = AverageMeter()
        # By default, all tensors with requires_grad=True are tracking their computational history and support gradient computation. 
        # In evaluation round, no need to do this, hence, torch.no_grad(), it can also speed up computations
        with torch.no_grad():
            for batch in dataloader:
                x = batch['data']
                y = batch['label']
                x, y = x.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)
                out = model.predict(x)
                pred = F.log_softmax(out, dim=1).argmax(dim=1)
                loss = loss_fn(out,y)
                test_loss.update(loss.item())
                correct += (pred == y.data.argmax(dim=1)).sum().item()
        acc = correct/size
        return test_loss.avg, acc

    
    def test_target(self):
        args = self.args
        model = self.model
        test_dataloader = self.t_test_dataloader
        save_name = "earlystop" + args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
        model_pth = os.path.join('saved_model_pth', save_name)
        model.load_state_dict(torch.load(model_pth))
        model.eval()

        test_loss = AverageMeter()
        correct = 0
        len_target_dataset = len(test_dataloader.dataset)
        
        preds = []
        gt = []
        with torch.no_grad():
            for batch in test_dataloader:
                x = batch['data']
                y = batch['label']
                x, y = x.to(self.device,dtype=torch.float), y.to(self.device,dtype=torch.float)
                out = model.predict(x)
                loss = self.loss_fn(out,y)
                test_loss.update(loss.item())
                pred = F.log_softmax(out, dim=1).argmax(dim=1)
                correct += (pred == y.data.argmax(dim=1)).sum().item()

                # confusion matrix
                preds.extend(pred.detach().cpu().numpy().tolist())
                gt.extend(y.data.argmax(dim=1).detach().cpu().numpy().tolist())
        acc = correct/len_target_dataset
        cm = confusion_matrix(gt,preds, labels=[0,1])
        return acc, test_loss.avg,cm
    
    def test_dg(self):
        args = self.args
        model = self.model
        test_dataloader = self.generalization_dataloader
        save_name = "earlystop" + args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
        model_pth = os.path.join('saved_model_pth', save_name)
        model.load_state_dict(torch.load(model_pth))
        model.eval()

        test_loss = AverageMeter()
        correct = 0
        len_target_dataset = len(test_dataloader.dataset)
        
        preds = []
        gt = []
        with torch.no_grad():
            for batch in test_dataloader:
                x = batch['data']
                y = batch['label']
                x, y = x.to(self.device,dtype=torch.float), y.to(self.device,dtype=torch.float)
                out = model.predict(x)
                loss = self.loss_fn(out,y)
                test_loss.update(loss.item())
                # pred = torch.round(out.squeeze()) 
                # correct += (pred == y.squeeze()).sum().item()
                pred = F.log_softmax(out, dim=1).argmax(dim=1)
                correct += (pred == y.data.argmax(dim=1)).sum().item()

                # confusion matrix
                preds.extend(pred.detach().cpu().numpy().tolist())
                gt.extend(y.data.argmax(dim=1).detach().cpu().numpy().tolist())
        acc = correct/len_target_dataset
        cm = confusion_matrix(gt,preds, labels=[0,1])
        return acc,cm
    
    def img_visualization(self):
        args = self.args
        model = self.model
        test_dataloader = self.t_test_dataloader
        save_name = "earlystop" + args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
        #save_name = "last_epoch" + args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
        model_pth = os.path.join('saved_model_pth', save_name)
        model.load_state_dict(torch.load(model_pth))
        idx = 0
        for batch in test_dataloader:
            x = batch['data']
            y = batch['label']
            x, y = x.to(self.device,dtype=torch.float), y.to(self.device,dtype=torch.float)
            out = model.predict(x)
            pred = F.log_softmax(out, dim=1).argmax(dim=1)
            # get index of misclassified images
            misclassified_idx = (pred != y.data.argmax(dim=1)).nonzero()

            for i in range(len(x)):
                #i = i.data.item()
                title = f'GT: {y[i].argmax().item()} Pred: {pred[i].item()}'
                img = x[i].squeeze()
                save_img(img,f'all_case_exp2_1/{i}.png',title)
                np.save(f'all_case_exp2_1/{i}.npy',img.cpu().numpy())

            break
        return
def save_img(tensor,name,title):
    tensor = tensor.squeeze().detach().cpu().numpy()
    figure = plt.figure()
    ax = figure.add_subplot(111)
    data = tensor
    slice1 = data[0:62,:]
    slice2 = data[67:128,:]
    data = np.concatenate((slice1,slice2),axis=0)
    im = ax.imshow(data, cmap="jet")
    ax.set_aspect(0.75*data.shape[1] / data.shape[0])
    ax.axis('off')
    ax.set_title(title)
    plt.savefig(name, bbox_inches='tight')


class DistilationTrainer(BaseTrainer):

    def __init__(self,args, logger):
        super().__init__(args,logger)
    
    def train(self):
        args = self.args
        model = self.model
        opt = optim.Adam(model.get_parameters(initial_lr=args.lr),lr=args.lr)
        writer = define_writer(args)
        
        best_acc = 0
        for epoch in trange(self.args.num_epochs):
            train_loss_dis,train_loss_clf, train_loss_transfer, train_loss_total, acc = self._train_one_epoch(model,self.s_train_dataloader,self.t_train_dataloader,opt,epoch)
            writer.add_scalar("Loss/train_clf", train_loss_clf, epoch)
            writer.add_scalar("Loss/train_distill", train_loss_dis, epoch)
            writer.add_scalar("Loss/train_transfer", train_loss_transfer, epoch)
            writer.add_scalar("Loss/train_total",  train_loss_total, epoch)
            writer.add_scalar("Acc/train",  acc, epoch)
            # Test
            val_acc, val_loss = self._validation(model, self.s_val_dataloader)
            writer.add_scalar("Loss/val_clf",  val_loss, epoch)
            writer.add_scalar("Acc/val",  val_acc, epoch)
            if val_acc > best_acc:
                trials = 0
                best_acc = val_acc
                save_name = "earlystop" + args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
                save_pth = os.path.join('saved_model_pth', save_name)
                torch.save(model.state_dict(),save_pth)
                self.logger.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= args.patience:
                    self.logger.info(f'Early stopping on epoch {epoch}')
                    break

    

    def _train_one_epoch(self,model,source_loader,target_loader,opt,epoch):
        model.train()
        train_loss_clf = AverageMeter()
        train_loss_transfer = AverageMeter()
        train_loss_total = AverageMeter()
        train_loss_distill = AverageMeter()
        correct, total = 0, 0
        n_batch = min(len(target_loader),len(source_loader))
        iter_target = iter(target_loader)
        iter_source = iter(source_loader)
        for i in range(n_batch):
            source_batch = next(iter_source)
            data_source = source_batch['data']
            label_source = source_batch['label']

            target_batch = next(iter_target)
            data_target = target_batch['data']

            data_source, label_source = data_source.to(
                self.device,dtype=torch.float), label_source.to(self.device,dtype=torch.float)
            data_target = data_target.to(self.device,dtype=torch.float)

            out = model.predict(data_source)
            
            clf_loss, distill_loss,transfer_loss,other_loss = model(data_source, data_target, label_source)


            loss = clf_loss + self.args.transfer_loss_weight * transfer_loss + distill_loss * self.args.dis_loss_weight + other_loss*0.01
            train_loss_distill.update(distill_loss.item())

            pred = F.log_softmax(out, dim=1).argmax(dim=1)
            correct += (pred == label_source.data.argmax(dim=1)).sum().item()
            total += len(source_batch['data'])
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
        acc = correct/total
        return train_loss_distill.avg, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, acc
    
    def _validation(self,model,dataloader):
        model.eval()
        test_loss = AverageMeter()
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        len_target_dataset = len(dataloader.dataset)
        with torch.no_grad():
            for batch in dataloader:
                x = batch['data']
                y = batch['label']
                x, y = x.to(self.device,dtype=torch.float), y.to(self.device,dtype=torch.float)
                out = model.predict(x)
                loss = criterion(out,y)
                test_loss.update(loss.item())
                pred = F.log_softmax(out, dim=1).argmax(dim=1)
                correct += (pred == y.data.argmax(dim=1)).sum().item()
        acc = correct/len_target_dataset
        return acc, test_loss.avg
    
class DomainAdaptTrainer(BaseTrainer):
    def __init__(self,args,logger):
        super().__init__(args,logger)
    def train(self):
        args = self.args
        model = self.model
        opt = optim.Adam(model.get_parameters(initial_lr=args.lr),lr=args.lr)
        writer = define_writer(args)
        
        best_acc = 0
        for epoch in trange(self.args.num_epochs):
            train_loss_clf, train_loss_transfer, train_loss_total, acc = self._train_one_epoch(model,self.s_train_dataloader,self.t_train_dataloader,opt)
            writer.add_scalar("Loss/train_clf", train_loss_clf, epoch)
            writer.add_scalar("Loss/train_transfer", train_loss_transfer, epoch)
            writer.add_scalar("Loss/train_total",  train_loss_total, epoch)
            writer.add_scalar("Acc/train",  acc, epoch)
            # Test
            val_acc, val_loss = self._validation(model, self.s_val_dataloader)
            writer.add_scalar("Loss/val_clf",  val_loss, epoch)
            writer.add_scalar("Acc/val",  val_acc, epoch)
            if val_acc > best_acc:
                trials = 0
                best_acc = val_acc
                save_name = "earlystop" +args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
                save_pth = os.path.join('saved_model_pth', save_name)
                torch.save(model.state_dict(),save_pth)
                self.logger.info(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= args.patience:
                    self.logger.info(f'Early stopping on epoch {epoch}')
                    break
            
        save_name = "last_epoch" + args.model + '_' +f'{args.transfer_loss}' +'_dis_'+f'{args.dis_loss}'+ ".pth"
        save_pth = os.path.join('saved_model_pth', save_name)
        torch.save(model.state_dict(),save_pth)
        self.logger.info(f'last model saved with accuracy: {val_acc:2.2%}')
        self.logger.info(f'Transfer result: {best_acc:.4f}')

    def _train_one_epoch(self,model,source_loader,target_loader,opt):
        model.train()
        train_loss_clf = AverageMeter()
        train_loss_transfer = AverageMeter()
        train_loss_total = AverageMeter()
        correct, total = 0, 0
        n_batch = min(len(target_loader),len(source_loader))
        iter_target = iter(target_loader)
        iter_source = iter(source_loader)
        for i in range(n_batch):
            source_batch = next(iter_source)
            data_source = source_batch['data']
            label_source = source_batch['label']

            target_batch = next(iter_target)
            data_target = target_batch['data']

            data_source, label_source = data_source.to(
                self.device,dtype=torch.float), label_source.to(self.device,dtype=torch.float)
            data_target = data_target.to(self.device,dtype=torch.float)

            out = model.predict(data_source)
            
            clf_loss, transfer_loss,other_loss = model(data_source, data_target, label_source)
            loss = clf_loss + self.args.transfer_loss_weight * transfer_loss + other_loss*0.01

            pred = F.log_softmax(out, dim=1).argmax(dim=1)
            correct += (pred == label_source.data.argmax(dim=1)).sum().item()
            total += len(source_batch['data'])
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
        acc = correct/total
        return train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, acc
    
    def _validation(self,model,dataloader):
        model.eval()
        test_loss = AverageMeter()
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        len_target_dataset = len(dataloader.dataset)
        with torch.no_grad():
            for batch in dataloader:
                x = batch['data']
                y = batch['label']
                x, y = x.to(self.device,dtype=torch.float), y.to(self.device,dtype=torch.float)
                out = model.predict(x)
                loss = criterion(out,y)
                test_loss.update(loss.item())
                pred = F.log_softmax(out, dim=1).argmax(dim=1)
                correct += (pred == y.data.argmax(dim=1)).sum().item()
        acc = correct/len_target_dataset
        return acc, test_loss.avg
    
   
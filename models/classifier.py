import torch
import torch.nn as nn

#from models.loss_funcs import *

from models.transfer_losses import *
from models.Resnet import *
from models.four_conv import *
from models.ten_conv import *
#from transfer_losses import *

class BaseNet(nn.Module):
    def __init__(self,backbone='4-conv'):
        super(BaseNet, self).__init__()
        if backbone == "10-conv":
            self.classifier = Ten_conv()
        elif backbone == "resnet18":
            self.classifier = ResNet18()
        elif backbone == "resnet34":
            self.classifier = ResNet34()
        elif backbone == '4-conv':
            self.classifier = Four_conv()
    def forward(self, x):
        feature, clf = self.classifier(x)
        return feature, clf
    def predict(self, x):
        _, clf = self.classifier(x)
        return clf

class TransferNet(nn.Module):
    def __init__(self,transfer_loss='mmd', backbone='4-conv'):
        super(TransferNet, self).__init__()
        if backbone == "10-conv":
            self.classifier = Ten_conv()
        elif backbone == "resnet18":
            self.classifier = ResNet18()
        elif backbone == "resnet34":
            self.classifier = ResNet34()
        elif backbone == '4-conv':
            self.classifier = Four_conv()
        self.transfer_loss = transfer_loss
        self.adapt_loss = TransferLoss(loss_type=transfer_loss, backbone=backbone)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, source_data, target_data, source_label):
        source_feature, source_clf = self.classifier(source_data)
        target_feature, target_clf = self.classifier(target_data)
        clf_loss = self.criterion(source_clf, source_label)
        # transfer
        kwargs = {}
        if self.transfer_loss in [r"adv",r'mmd']:
            transfer_loss = self.adapt_loss(source_feature, target_feature, **kwargs)
            return clf_loss, transfer_loss,0
        elif self.transfer_loss == r'advmmd':
            mmd_loss,adv_loss = self.adapt_loss(source_feature, target_feature, **kwargs)
            return clf_loss,mmd_loss,adv_loss
    
    def predict(self, x):
        _, clf = self.classifier(x)
        return clf
    
    def get_parameters(self, initial_lr=1.0e-6):
        params = [
            {'params': self.classifier.parameters(), 'lr': 1.0 * initial_lr}
        ]
        if self.transfer_loss in [r"adv",r'advmmd']:
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

class DisNet(nn.Module):
    def __init__(self,transfer_loss='mmd', backbone='4-conv'):
        super(DisNet, self).__init__()
        if backbone == "10-conv":
            self.classifier = Ten_conv()
        elif backbone == "resnet18":
            self.classifier = ResNet18()
        elif backbone == "resnet34":
            self.classifier = ResNet34()
        elif backbone == '4-conv':
            self.classifier = Four_conv()
        self.transfer_loss = transfer_loss
        self.adapt_loss = TransferLoss(loss_type=transfer_loss, backbone=backbone)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, source_data, target_data, source_label):
        source_feature, source_clf = self.classifier(source_data)
        target_feature, target_clf = self.classifier(target_data)

        clf_loss = self.criterion(source_clf, source_label)

        # step 1: calculate the distance matrix
        distance_matrix = torch.cdist(source_feature, target_feature, p=2)
        # step 2: for each row in the distance matrix, find the smallest target element for each source element
        _, min_distance_index = torch.min(distance_matrix, dim=1)

        # step 3: for each column in the distance matrix, find the smallest source element for each target element
        _, min_distance_index_t = torch.min(distance_matrix, dim=0)


        # step 4: calculate the distillation loss according to the indices
        dis_loss = distill_loss(source_clf, target_clf[min_distance_index,:])
        dis_loss += distill_loss(source_clf[min_distance_index_t,:], target_clf)

        # transfer loss
        kwargs = {}
        if self.transfer_loss in [r"adv",r'mmd']:
            transfer_loss = self.adapt_loss(source_feature, target_feature, **kwargs)
            return clf_loss,dis_loss,transfer_loss,0
        elif self.transfer_loss == r'advmmd':
            mmd_loss,adv_loss = self.adapt_loss(source_feature, target_feature, **kwargs)
            return clf_loss,dis_loss,mmd_loss,adv_loss
    
    def predict(self, x):
        _, clf = self.classifier(x)
        return clf
    
    def get_parameters(self, initial_lr=1.0e-6):
        params = [
            {'params': self.classifier.parameters(), 'lr': 1.0 * initial_lr}
        ]
        if self.transfer_loss in [r"adv",r'advmmd']:
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params
    
def distill_loss(teacher_out, student_out):
    #loss = nn.BCELoss()
    #out = loss(student_out, teacher_out)
    teacher_out = F.softmax(teacher_out, dim=-1)    
    loss = torch.sum( -teacher_out * F.log_softmax(student_out, dim=-1), dim=-1) # element-wise multiplication. sum along the classes
    return loss.mean() # take the mean over the batch
    #return out # take the mean over the batch



if __name__ == "__main__":

    data = torch.rand(10,128,45)
    label = torch.randint(0,2,(10,1)).float()
    model = DisNet(transfer_loss='advmmd', backbone='resnet18')
    y = model(data,data,label)

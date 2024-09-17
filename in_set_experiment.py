import os
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from torch import optim
from torch.utils.data import DataLoader,random_split
from torch.optim.lr_scheduler import MultiStepLR

from utils import *
from Dataset_v12 import MD_detection
from MD_transform import *

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int,default=1)
experiment_index= parser.parse_args().experiment_index

if experiment_index == 1:
    root = ['Sep12','Sep13']
elif experiment_index == 2:
    root = ['Sep12','Sep14']
elif experiment_index == 3:
    root = ['Sep12','Sep15']
elif experiment_index == 4:
    root = ['Sep12','Sep16']
elif experiment_index == 5:
    root = ['Sep13','Sep14']
elif experiment_index == 6:
    root = ['Sep13','Sep15']
elif experiment_index == 7:
    root = ['Sep13','Sep16']
elif experiment_index == 8:
    root = ['Sep14','Sep15']
elif experiment_index == 9:
    root = ['Sep14','Sep16']
elif experiment_index == 10:
    root = ['Sep15','Sep16']
root = [os.path.join(r'G:\Jiarui',f) for f in root]

# define logger
logging_file_name = datetime.now().strftime("%Y-%b-%d-%H-%M")
logging_file_name = f"{logging_file_name}_resnet18_inset.log"
if not os.path.exists('log'):
    os.makedirs('log')
log_pth = os.path.join('log',logging_file_name)
logging.basicConfig(filename=log_pth, level=logging.DEBUG)
logger = logging.getLogger('myapp')
logger.info('Start log recording')

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seed
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#define dataset
dataset = MD_detection(root=root)
train, test = random_split(dataset, [int(len(dataset)*0.7), len(dataset)-int(len(dataset)*0.7)])
train_dataloader = DataLoader(train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test, batch_size=128, shuffle=False)

# define model
model = BaseNet(backbone='resnet18')
model = model.to(device)

# define optimizer
opt = optim.Adam(model.parameters(), lr=1e-6)

# define loss function
loss_fn = nn.CrossEntropyLoss()

# Start training
model.train()

for epoch in trange(100,desc='Training Loop'):
    size = len(train_dataloader.dataset)
    train_epoch_loss = AverageMeter()
    correct = 0
    torch.manual_seed(42)
    model.train()
    source_iter = iter(train_dataloader)
    #for batch in dataloader:
    for i in range(len(train_dataloader)):
        batch = next(source_iter)
        x = batch['data']
        y = batch['label']
        x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
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
    logger.info(f"Epoch {epoch}: Loss: {train_epoch_loss.avg}, Accuracy: {acc}")

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
        x, y = x.to(device,dtype=torch.float), y.to(device,dtype=torch.float)
        out = model.predict(x)
        loss = loss_fn(out,y)
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
n_acc = cm[0,0]/(cm[0,0]+cm[0,1])
p_acc = cm[1,1]/(cm[1,1]+cm[1,0])
f1_score = 2*cm[1,1]/(2*cm[1,1]+cm[1,0]+cm[0,1])
print(f'acc:{acc},positive_acc:{p_acc},negative_acc:{n_acc}, f1_score:{f1_score}')
logging.info(f'DA result:\n acc {str(acc)}, positive_acc: {p_acc}, negative_acc: {n_acc}, f1_score: {f1_score}')

with open('result.csv','a') as f:
    f.write(f'{experiment_index},{acc},{p_acc},{n_acc},{f1_score}\n')





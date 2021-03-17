#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import imageio
import numpy as np
import torch

df = pd.read_csv('/scratch/yz6121/data/csv_file/labels')
df.head()
df = df[df['ImageDir']==54]
print(df.shape)
df.head()


# In[2]:


import torch
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# ## 1. Clean dataset(pair PA and L view)

# In[3]:


allid = list(df.PatientID)
pa = []
l = []
df = df.reset_index(drop=True)
for i in range(len(df)):
    ids = df.loc[i].PatientID
    view = df.loc[i].Projection
    if view == 'PA':
        pa.append(ids)
    elif view == 'L':
        l.append(ids)
intersect = list(set(pa).intersection(set(l)))
print(len(intersect))


# In[5]:


pa1 = pa.copy()
l1 = l.copy()
for i in pa:
    if i in intersect:
        pass
    else:
        pa1.remove(i)
for i in l:
    if i in intersect:
        pass
    else:
        l1.remove(i)


# In[6]:


print(len(pa1), len(l1))


# In[7]:


from collections import Counter

temp = Counter(pa1+l1)
for i in temp.keys():
    if temp[i]>=3:
        intersect.remove(i)
print(len(intersect))


# In[8]:


df=df[df['PatientID'].isin(intersect)]
print(df.shape)
df.head()


# In[10]:


temp = Counter(list(df.PatientID))
intersect1= list(set(list(df.PatientID)))
for i in temp.keys():
    if temp[i]>=3:
        intersect1.remove(i)
print(len(intersect1))
df = df[df['PatientID'].isin(intersect1)]
print(df.shape)
df = df.sort_values(by='PatientID')
df = df.reset_index(drop = True)


# ## 2. Image loader

# In[12]:


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


# In[13]:


#change, new version
import imageio
import numpy as np
import torch
from skimage import io, transform
def Generator_2D_slices_new(df, batchsize,inputKey='dataMR',outputKey='dataCT'): #MR: P view, CT: L view

    path_patients='/scratch/yz6121/data/images/'

    #print(path_patients)

    #patients = os.listdir(path_patients)#every file  is a hdf5 patient
    pa_view = df[df['Projection']=='PA']
    l_view = df[df['Projection']=='L']

    while True:
        #for idx,namepatient in enumerate(patients):
        for i in range(0,len(df),2):
            view_1 = df.loc[i].Projection
            #view_2 = df.loc[i+1].Projection
            location_1 = path_patients+df.loc[i].ImageID
            location_2 = path_patients+df.loc[i+1].ImageID
            if view_1 == 'PA':
                dataMR = imageio.imread(location_1)
                dataCT = imageio.imread(location_2)
            else:
                dataCT = imageio.imread(location_1)
                dataMR = imageio.imread(location_2)
            #print('check here 2',dataMR.shape, dataCT.shape)
            
            dataCT = transform.resize(dataCT, (2096, 2096)) #resize
            dataMR = transform.resize(dataMR, (2096, 2096))
            height = 1024
            width = 1024
            
            dataCT = crop_center(dataCT, width, height)
            dataMR = crop_center(dataMR, width, height)
            
            dataCT = np.expand_dims(dataCT, axis = 0)
            dataMR = np.expand_dims(dataMR, axis = 0)
            dataMR = np.expand_dims(dataMR, axis = 3)
            #print('datact, data mr shape',dataCT.shape, dataMR.shape)
            
            yield (dataMR, dataCT)



# ## 3. Model-UNet

# In[14]:


# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from utils import *
from Unet2d_pytorch import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from Unet3d_pytorch import UNet3D
from nnBuildUnits import CrossEntropy3d, topK_RegLoss, RelativeThreshold_RegLoss, gdl_loss, adjust_learning_rate, calc_gradient_penalty
import time
import SimpleITK as sitk
class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1,self).__init__()
        #you can make abbreviations for conv and fc, this is not necessary
        #class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(1,32,(9,9))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,(5,5))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,(5,5))
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(984064,512)
        #self.bn3= nn.BatchNorm1d(6)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,1)
        
        
    def forward(self,x):
#         print 'line 114: x shape: ',x.size()
        #x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))),(2,2))#conv->relu->pool
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))#conv->relu->pool

        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))#conv->relu->pool
        
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))#conv->relu->pool
        
        #print('x.shape',x.shape)

        x = x.view(-1,self.num_of_flat_features(x))
        #return x
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        #x = F.sigmoid(x)
        #print 'min,max,mean of x in 0st layer',x.min(),x.max(),x.mean()

        return x
    
    def num_of_flat_features(self,x):
        size=x.size()[1:]#we donot consider the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features


# ## 4. Training

# In[ ]:


# from __future__ import print_function
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
from utils import *
from Unet2d_pytorch import UNet, ResUNet, UNet_LRes, ResUNet_LRes, Discriminator
from Unet3d_pytorch import UNet3D
from nnBuildUnits import CrossEntropy3d, topK_RegLoss, RelativeThreshold_RegLoss, gdl_loss, adjust_learning_rate, calc_gradient_penalty
import time
import SimpleITK as sitk
import pickle

inputs_test = []
exinputs_test = []
labels_test = []
outputG_test = []

# Training settings

parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--gpuID", type=int, default=1, help="how to normalize the data")
parser.add_argument("--isAdLoss", action="store_true", help="is adversarial loss used?", default=False)#
parser.add_argument("--isWDist", action="store_true", help="is adversarial loss with WGAN-GP distance?", default=False)
parser.add_argument("--lambda_AD", default=0.05, type=float, help="weight for AD loss, Default: 0.05")
parser.add_argument("--lambda_D_WGAN_GP", default=10, type=float, help="weight for gradient penalty of WGAN-GP, Default: 10")
parser.add_argument("--how2normalize", type=int, default=6, help="how to normalize the data")
parser.add_argument("--whichLoss", type=int, default=1, help="which loss to use: 1. LossL1, 2. lossRTL1, 3. MSE (default)")
parser.add_argument("--isGDL", action="store_true", help="do we use GDL loss?", default=False)#
parser.add_argument("--gdlNorm", default=2, type=int, help="p-norm for the gdl loss, Default: 2")
parser.add_argument("--lambda_gdl", default=0.05, type=float, help="Weight for gdl loss, Default: 0.05")
parser.add_argument("--whichNet", type=int, default=1, help="which loss to use: 1. UNet, 2. ResUNet, 3. UNet_LRes and 4. ResUNet_LRes (default, 3)")
parser.add_argument("--lossBase", type=int, default=1, help="The base to multiply the lossG_G, Default (1)")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--isMultiSource", action="store_true", help="is multiple modality used?", default=False)
parser.add_argument("--numOfChannel_singleSource", type=int, default=1, help="# of channels for a 2D patch for the main modality (Default, 5)")
parser.add_argument("--numOfChannel_allSource", type=int, default=1, help="# of channels for a 2D patch for all the concatenated modalities (Default, 5)")
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for") #200000
parser.add_argument("--showTrainLossEvery", type=int, default=100, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=5000, help="number of iterations to save the model")
parser.add_argument("--showValPerformanceEvery", type=int, default=100, help="number of iterations to show validation performance")
parser.add_argument("--showTestPerformanceEvery", type=int, default=500, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=5e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_netD", type=float, default=5e-3, help="Learning Rate for discriminator. Default=5e-3")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="prob to drop neurons to zero: 0.2")
parser.add_argument("--decLREvery", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--lrDecRate", type=float, default=0.5, help="The weight for decreasing learning rate of netG Default=0.5")
parser.add_argument("--lrDecRate_netD", type=float, default=0.1, help="The weight for decreasing learning rate of netD. Default=0.1")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default=True)
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--RT_th", default=0.005, type=float, help="Relative thresholding: 0.005")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="/scratch/yz6121/model/is/gan", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub1_pet_BatchAug_sNorm_resunet_dp_lres_bn_lr5e3_lrdec_base1_lossL1_lossGDL0p05_0705_", type=str, help="prefix of the to-be-saved predicted filename")
parser.add_argument("--test_input_file_name",default='sub13_mr.hdr',type=str, help="the input file name for testing subject")
parser.add_argument("--test_gt_file_name",default='sub13_ct.hdr',type=str, help="the ground-truth file name for testing subject") 

global opt, model 
opt = parser.parse_args(args=[])


def main():    
    print(opt)  
        



    netD = Discriminator1()
    netD.apply(weights_init)
    netD.cuda()
    
    optimizerD = optim.Adam(netD.parameters(),lr=opt.lr_netD)
    criterion_bce=nn.BCELoss()
    criterion_bce.cuda()
    
    #net=UNet()
    if opt.whichNet==1:
        net = UNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==2:
        net = ResUNet(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==3:
        net = UNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1)
    elif opt.whichNet==4:
        net = ResUNet_LRes(in_channel=opt.numOfChannel_allSource, n_classes=1, dp_prob = opt.dropout_rate)
    #net.apply(weights_init)
    net.cuda()
    params = list(net.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    
 
    
    optimizer = optim.Adam(net.parameters(),lr=opt.lr)
    criterion_L2 = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    criterion_RTL1 = RelativeThreshold_RegLoss(opt.RT_th)
    criterion_gdl = gdl_loss(opt.gdlNorm)

    
    given_weight = torch.cuda.FloatTensor([1,4,4,2])
    
    criterion_3d = CrossEntropy3d(weight=given_weight)
    
    criterion_3d = criterion_3d.cuda()
    criterion_L2 = criterion_L2.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_RTL1 = criterion_RTL1.cuda()
    criterion_gdl = criterion_gdl.cuda()
    

    if opt.isMultiSource:
        data_generator = Generator_2D_slicesV1(path_patients_h5,opt.batchSize, inputKey='dataLPET', segKey='dataCT', contourKey='dataHPET')
        data_generator_test = Generator_2D_slicesV1(path_patients_h5_val, opt.batchSize, inputKey='dataLPET', segKey='dataCT', contourKey='dataHPET')
    else:
        data_generator = Generator_2D_slices_new(df,opt.batchSize,inputKey='dataMR',outputKey='dataCT')
        data_generator_test = Generator_2D_slices_new(df,opt.batchSize,inputKey='dataMR',outputKey='dataCT')
        print('okay')

    #data_generator = Generator_2D_slicesV1(path_patients_h5,opt.batchSize, inputKey='dataLPET', segKey='dataCT', contourKey='dataHPET')
    #data_generator_test = Generator_2D_slicesV1(path_patients_h5_val, opt.batchSize, inputKey='dataLPET', segKey='dataCT', contourKey='dataHPET')
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            net.load_state_dict(checkpoint['model'])
            opt.start_epoch = 100000
            opt.start_epoch = checkpoint["epoch"] + 1
            # net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 

    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch, opt.numofIters+50):
        #print('iter %d'%iter)
                #print('iter %d'%iter)
        if opt.isMultiSource:
            inputs, exinputs, labels = next(data_generator)#.next()
        else:
            inputs, labels = next(data_generator)#.next()         
            exinputs = inputs

        inputs = np.transpose(inputs,(0,3,1,2)) #change here: added
        exinputs = np.transpose(exinputs,(0,3,1,2)) #change here: added

        
        #change here
        inputs = inputs.astype(float)
        inputs = torch.from_numpy(inputs)
        inputs = inputs.float()
        exinputs = exinputs.astype(float)
        exinputs = torch.from_numpy(exinputs)
        exinputs = exinputs.float()
        labels = labels.astype(float)
        labels = torch.from_numpy(labels)
        labels = labels.float()


        if opt.isMultiSource:
            source = torch.cat((inputs, exinputs),dim=1)
        else:
            source = inputs
        #source = inputs
        mid_slice = opt.numOfChannel_singleSource//2     
        source = source.cuda()
        labels = labels.cuda()
        #we should consider different data to train
        
        #wrap them into Variable
        #source, residual_source, labels = Variable(source),Variable(residual_source), Variable(labels)
        source, labels = Variable(source), Variable(labels)
        #inputs, exinputs, labels = Variable(inputs),Variable(exinputs), Variable(labels)
        
        ## (1) update D network: maximize log(D(x)) + log(1 - D(G(z)))
        #print('source shape',source.shape) #change here, source shape
        '''
        if opt.isAdLoss:
            #outputG = net(source,residual_source) 
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source) 
            else:
                outputG = net(source) 
                
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
            
            
            
        
        
            
            
            
            outputD_real = netD(labels)   
            #print(outputD_real.shape)
            outputD_real = F.sigmoid(outputD_real)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)
            outputD_fake = F.sigmoid(outputD_fake)
            netD.zero_grad()
            batch_size = inputs.size(0)
            real_label = torch.ones(batch_size,1)
            real_label = real_label.cuda()
            #print(real_label.size())
            real_label = Variable(real_label)
            #print(outputD_real.size())
            loss_real = criterion_bce(outputD_real,real_label)
            loss_real.backward()
            #train with fake data
            fake_label = torch.zeros(batch_size,1)
            fake_label = fake_label.cuda()
            fake_label = Variable(fake_label)
            loss_fake = criterion_bce(outputD_fake,fake_label)
            loss_fake.backward()
            
            lossD = loss_real + loss_fake
            #update network parameters
            optimizerD.step()
            
        if opt.isWDist:
            one = torch.FloatTensor([1])
            mone = one * -1
            one = one.cuda()
            mone = mone.cuda()
            
            netD.zero_grad()
            
            #outputG = net(source,residual_source) #5x64x64->1*64x64
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  # 5x64x64->1*64x64
            else:
                outputG = net(source)  # 5x64x64->1*64x64
                
            if len(labels.size())==3:
                labels = labels.unsqueeze(1)
                
            outputD_real = netD(labels)
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
                
            outputD_fake = netD(outputG)

            
            batch_size = inputs.size(0)
            
            D_real = outputD_real.mean()
            # print D_real
            D_real.backward(mone)
        
        
            D_fake = outputD_fake.mean()
            D_fake.backward(one)
        
            gradient_penalty = opt.lambda_D_WGAN_GP*calc_gradient_penalty(netD, labels.data, outputG.data)
            gradient_penalty.backward()
            
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            
            optimizerD.step()
        '''
        
        
        ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
        
        if opt.whichNet == 3 or opt.whichNet == 4:
            outputG = net(source, residual_source)  
        else:
            outputG = net(source)  
       
        net.zero_grad()
        if opt.whichLoss==1:
            lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(labels))        
        elif opt.whichLoss==2:
            lossG_G = criterion_RTL1(torch.squeeze(outputG), torch.squeeze(labels))
        else:
            lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(labels))
        lossG_G = opt.lossBase * lossG_G
        lossG_G.backward(retain_graph=True) #compute gradients

        if opt.isGDL:
            lossG_gdl = opt.lambda_gdl * criterion_gdl(outputG,torch.unsqueeze(torch.squeeze(labels,1),1))
            lossG_gdl.backward() #compute gradients
        '''
        if opt.isAdLoss:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  
            else:
                outputG = net(source)  
            
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD = netD(outputG)
            outputD = F.sigmoid(outputD)
            lossG_D = opt.lambda_AD*criterion_bce(outputD,real_label) #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward()
            
        if opt.isWDist:
            #we want to fool the discriminator, thus we pretend the label here to be real. Actually, we can explain from the 
            #angel of equation (note the max and min difference for generator and discriminator)
            
            if opt.whichNet == 3 or opt.whichNet == 4:
                outputG = net(source, residual_source)  
            else:
                outputG = net(source)  
            if len(outputG.size())==3:
                outputG = outputG.unsqueeze(1)
            
            outputD_fake = netD(outputG)

            outputD_fake = outputD_fake.mean()
            
            lossG_D = opt.lambda_AD*outputD_fake.mean() #note, for generator, the label for outputG is real, because the G wants to confuse D
            lossG_D.backward(mone)
        '''
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizer.step() #update network parameters

        running_loss = running_loss + lossG_G.data
        
        
        if iter%opt.showTrainLossEvery==0: #print every 2000 mini-batches
            print('************************************************')
            print('time now is: ' + time.asctime(time.localtime(time.time())))
#             print 'running loss is ',running_loss
            print('average running loss for generator between iter [%d, %d] is: %.5f'%(iter - 100 + 1,iter,running_loss/100))
            
            print('lossG_G is %.5f respectively.'%(lossG_G.data))
            ############################################
            
            if opt.isGDL:
                print('loss for GDL loss is %f'%lossG_gdl.data[0])

            if opt.isAdLoss:
                print('loss_real is ',loss_real.data,'loss_fake is ',loss_fake.data,
                      'outputD_real is',outputD_real.data)
                print('loss for discriminator is %f'%lossD.data)  
                print('lossG_D for discriminator is %f'%lossG_D.data)  

            if opt.isWDist:
                print('loss_real is ',torch.mean(D_real).data[0],'loss_fake is ',torch.mean(D_fake).data[0])
                print('loss for discriminator is %f'%Wasserstein_D.data[0], ' D cost is %f'%D_cost)                
                print('lossG_D for discriminator is %f'%lossG_D.data[0])  
            
  
            print('cost time for iter [%d, %d] is %.2f'%(iter - 100 + 1,iter, time.time()-start))
            print('************************************************')
            running_loss = 0.0
            start = time.time()
       
        if iter%opt.saveModelEvery==0: #save the model
            state = {
                'epoch': iter+1,
                'model': net.state_dict()
            }
            torch.save(state, opt.prefixModelName+'both_2'+'%d.pt'%iter)
            print('save model: '+opt.prefixModelName+'%d.pt'%iter)
                   
            

            if opt.isAdLoss or opt.isWDist:
                torch.save(netD.state_dict(), opt.prefixModelName+'both_2'+'_net_D%d.pt'%iter)
        if iter%opt.decLREvery==0:
            opt.lr = opt.lr*opt.lrDecRate
            adjust_learning_rate(optimizer, opt.lr)
            if opt.isAdLoss or opt.isWDist:
                opt.lr_netD = opt.lr_netD*opt.lrDecRate_netD
                adjust_learning_rate(optimizerD, opt.lr_netD)
        

  
        if iter%opt.showValPerformanceEvery==0: #test one subject
            with torch.no_grad():
            # to test on the validation dataset in the format of h5 
#            inputs,exinputs,labels = data_generator_test.next()
                if opt.isMultiSource:
                    inputs, exinputs, labels = next(data_generator)#.next()
                else:
                    inputs, labels = next(data_generator)#.next()
                    exinputs = inputs

                inputs = np.transpose(inputs,(0,3,1,2)) #ADD HERE


                exinputs = np.transpose(exinputs, (0, 3, 1, 2)) #ADD HERE


                labels = np.squeeze(labels)
                inputs = inputs.astype(float)
                inputs = torch.from_numpy(inputs)
                inputs = inputs.float()
            
                inputs_test.append(inputs)
            
                exinputs = exinputs.astype(float)
                exinputs = torch.from_numpy(exinputs)
                exinputs = exinputs.float()
                exinputs_test.append(exinputs)
            
                labels = labels.astype(float)
                labels = torch.from_numpy(labels)
                labels = labels.float()
                labels_test.append(labels)
            
                mid_slice = opt.numOfChannel_singleSource // 2
                residual_source = inputs[:, mid_slice, ...]
                if opt.isMultiSource:
                    source = torch.cat((inputs, exinputs), dim=1)
                else:
                    source = inputs
                source = source.cuda()
                residual_source = residual_source.cuda()
                labels = labels.cuda()
                source,residual_source,labels = Variable(source),Variable(residual_source), Variable(labels)


                if opt.whichNet == 3 or opt.whichNet == 4:
                    outputG = net(source, residual_source)  
                else:
                    outputG = net(source)  
                outputG_test.append(outputG) 
            
    
            if iter%2000==0:  #save

                file=open(r"./results/inputs_test512_both_2.bin","wb")
                pickle.dump(inputs_test,file) 
                file.close()
            
                file=open(r"./results/exinputs_test512_both_2.bin","wb")
                pickle.dump(exinputs_test,file) 
                file.close()
            
                file=open(r"./results/labels_test512_both_2.bin","wb")
                pickle.dump(labels_test,file) 
                file.close()
            
                file=open(r"./results/outputG_test512_both_2.bin","wb")
                pickle.dump(outputG_test,file) 
                file.close()
          

            
            if opt.whichLoss == 1:
                lossG_G = criterion_L1(torch.squeeze(outputG), torch.squeeze(labels))
            elif opt.whichLoss == 2:
                lossG_G = criterion_RTL1(torch.squeeze(outputG), torch.squeeze(labels))
            else:
                lossG_G = criterion_L2(torch.squeeze(outputG), torch.squeeze(labels))
            lossG_G = opt.lossBase * lossG_G
            print('.......come to validation stage: iter {}'.format(iter),'........')
            print('lossG_G is %.5f.'%(lossG_G.data))

            if opt.isGDL:
                lossG_gdl = criterion_gdl(outputG, torch.unsqueeze(torch.squeeze(labels,1),1))
                print('loss for GDL loss is %f'%lossG_gdl.data)

    #print('Finished Training')

if __name__ == '__main__':   
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuID)  
    main()
    


# In[40]:



file=open(r"./results/outputG_test512_generator_3.bin","rb")
mylist=pickle.load(file) 
print(len(mylist))


# In[46]:


outputG_0 = mylist[299]
outputG_0 = outputG_0.squeeze()
outputG_0 = outputG_0.detach().cpu().numpy()
print(outputG_0.shape)


# In[47]:


from PIL import Image
import numpy as np
imageio.imwrite(r'./results/generator_resized_299.jpg',outputG_0)
#img = Image.fromarray(outputG_0.astype('uint8'))
#img.save('./results/generator_only_100.png')
#img.show()


# In[ ]:





# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch

import numpy
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


from DiscriminativeCell import DiscriminativeCell
from GenerativeCell import GenerativeCell

from attention import Attention
import time
 

# Define some constants
OUT_LAYER_SIZE = (3,) + tuple(2 ** p for p in range(4, 10))
ERR_LAYER_SIZE = tuple(size * 2 for size in OUT_LAYER_SIZE)
IN_LAYER_SIZE = (3,) + ERR_LAYER_SIZE

DIRECTION  = 2 

LAYERS = 2

N_HIDDEN = 4

class PrednetModel(nn.Module):
    """
    Build the Prednet model
    """

    def __init__(self, error_size_list, num_of_layers):
        super(PrednetModel,self).__init__()
        # print "error_size_list", error_size_list
        self.number_of_layers = num_of_layers
        # print "self.number_of_layers", self.number_of_layers
        for layer in range(0, self.number_of_layers):
            
            setattr(self, 'discriminator_' + str(layer + 1), DiscriminativeCell(
                        input_size={'input': IN_LAYER_SIZE[layer], 'state': OUT_LAYER_SIZE[layer]},
                        hidden_size=OUT_LAYER_SIZE[layer],
                        first=(not layer)
                        ))
            
            for d in range(0, DIRECTION):
                setattr(self, 'generator_' + str(layer + 1) + "_" + str(d), GenerativeCell(
                        input_size={'error': ERR_LAYER_SIZE[layer], 'up_state':
                        OUT_LAYER_SIZE[layer + 1] if layer != self.number_of_layers - 1 else 0},
                        hidden_size=OUT_LAYER_SIZE[layer],
                        error_init_size=error_size_list[layer]
                        ))
                    
        
    def forward(self, bottom_up_input, error, state, action_in):

        # generative branch
        up_state = [None] * self.number_of_layers
        
        #self.action = [Attention(10) for count in range(self.number_of_layers-1)]
        self.action = Attention(N_HIDDEN) 
        if torch.cuda.is_available():
            self.action = self.action.cuda()
        
        for layer in reversed(range(0, self.number_of_layers)):
            
             
            if not layer < self.number_of_layers - 1 :
                
                for d in range(0, DIRECTION):
                   
                    state[d][layer] = getattr(self, 'generator_' + str(layer + 1) + "_" + str(d))(
                        error[layer], None, state[d][layer]
                        )
            else:
                  
                for d in range(0, DIRECTION):
                    state[d][layer] = getattr(self, 'generator_' + str(layer + 1) + "_" + str(d))(
                        error[layer], up_state[layer+1], state[d][layer]
                        )
             
            
            #up_state[layer] = self.action[layer-1]([i[layer][0] for i in state], action_in) 
            # print state[layer][0].is_cuda
            up_state[layer] = self.action([i[layer][0] for i in state], action_in) 

        # discriminative branch
        for layer in range(0, self.number_of_layers):
            if layer == 0:
                error[layer] = getattr(self, 'discriminator_' + str(layer + 1))(
                bottom_up_input,
                up_state[layer]
                #state[layer][0]
            )
            else:
                error[layer] = getattr(self, 'discriminator_' + str(layer + 1))(
                error[layer - 1],
                up_state[layer]
            )
        #print up_state[0].size()
        return error, state, up_state





def _test_training(inData, outData, action):
    number_of_layers = LAYERS
     
    T =  inData.size()[0]
   
    # print inData
    #print outData
    # T = 6  # sequence length
    max_epoch = 1  # number of epochs
    lr_l1 = 1e-3     # learning rate
    lr_l2 = 1e-4
    momentum = 0.8
    # set manual seed
    torch.manual_seed(0)
    
    error_list = numpy.zeros((max_epoch,1))

    L = number_of_layers - 1
    print('\n---------- Train a', str(L + 1), 'layer network ----------')
    print('Create the input image and target sequences')
    # inData = inData.view(T, 1, 8, 8)
    
    sequence_pred = torch.zeros((T, 1, 3, 4 * 2 ** (L),6 * 2 ** (L)))
    
    # input_sequence = Variable(torch.rand(T, 1, 3, 4 * 2 ** L, 6 * 2 ** L))
    if torch.cuda.is_available():
        input_sequence = Variable(inData).cuda()
        action_in = Variable(action).cuda()
    else:
    
        input_sequence = Variable(inData)
        action_in = Variable(action)
    print('Input has size', list(input_sequence.data.size()))
    
    
    # error_init_size_list = input_sequence.data.size()
    error_init_size_list = tuple(
        (1, ERR_LAYER_SIZE[l], 4 * 2 ** (L - l),  6 * 2 ** (L - l)) for l in range(0, L + 1)
    )
    
    print('The error initialisation sizes are', error_init_size_list)
    
    #target_sequence = Variable(outData)
    target_sequence = Variable(torch.zeros(T, error_init_size_list[0][0], error_init_size_list[0][1], error_init_size_list[0][2], error_init_size_list[0][3]))
    
    
    #TODO: more elegent way

    print('Define a', str(L + 1), 'layer Prednet')
    model = PrednetModel(error_init_size_list, number_of_layers)
    
    
    if torch.cuda.is_available():
        print "Using GPU"
        model = model.cuda()
        
    optimizer = optim.SGD([{'params': model.generator_2_0.parameters()},
                    {'params': model.generator_2_1.parameters(), 'lr': lr_l2}], lr=lr_l1, momentum=momentum())

    #model = model.cuda()

    print('Create a MSE criterion')
    loss_fn = nn.MSELoss()
    
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    
    #print numpy.shape(sequence_pred)

    print('Run for', max_epoch, 'iterations')
    for epoch in range(0, max_epoch):
        
        t1 = time.time()
        state = [[None] * (L+1)] * DIRECTION
        error = [None] * (L + 1)
        loss = 0
        for t in range(0, T):
            error, state, prediction = model(input_sequence[t], error, state, action_in[t,:,:])
            
            if target_sequence.is_cuda == False and torch.cuda.is_available():
                target_sequence = target_sequence.cuda()
                
             
            loss += loss_fn(error[0], target_sequence[t])
            sequence_pred[t,:,:,:,:] = (prediction[0].data)
            
            if epoch  == max_epoch - 1 and (t == 54 or t==147):
                torch.save(prediction, "savetxt/mid_prediction"+ str(epoch) + "_"+ str(t) + ".txt")
                torch.save(state, "savetxt/mid_state"+ str(epoch) + "_"+ str(t) + ".txt")

        print(' > Epoch {:2d} loss: {:.3f}'.format((epoch + 1), loss.data[0]))
        error_list[epoch] = loss.data[0]
        
        if epoch % 100 == 0:
            
            for i in range(0, T):
               
                savematrix(sequence_pred[i,0,:,:,:], "savetxt/prediction_" + str(epoch) + "_" + str(i) +".txt")

        # zero grad parameters
        optimizer.zero_grad()

        # compute new grad parameters through time!
        loss.backward()
        optimizer.step()
        # learning_rate step against the gradient
        #for p in model.parameters():
            
        #    p.data.sub_(p.grad.data * lr)
            
        t2 = time.time()
        print "Epoch time: ~%f milliseconds" % ((t2 - t1) * 1000.)
    
    numpy.savetxt("error_list_lr"+str(lr)+"_layer"+str(LAYERS)+"_hid"+str(N_HIDDEN)+".txt", error_list)
    return model
            
def loadImage(size1, size2):
    
    import glob 
    import numpy as np
    
    length = len(glob.glob("images/*.txt"))
    
    InputData = torch.zeros((length, 1, 3, size1, size2)) 
    #InputData_gpu = InputData.cuda()
    
    i = 0
    
    for infile in sorted(glob.glob("images/*.txt")):
        print("Current File Being Processed is: " + infile)
        
        InputData[i, 0, 0, :, :] = torch.Tensor(normalisation(np.loadtxt(infile)))
        InputData[i, 0, 1, :, :] =  (InputData[i, 0, 0, :, :])
        InputData[i, 0, 2, :, :] =  (InputData[i, 0, 0, :, :]) 
        i += 1
        
    
        
    print("load image files completed!")
        
    action = torch.zeros((length, 1,  2))
    
    action_np = np.loadtxt("2-sorted.csv", delimiter=",")
    
    step = 4.77
    
    for i in range(length):
        
        
        if round(step*i) <= np.shape(action_np)[0]:
             
            action[i, 0, :] = torch.Tensor(action_np[int(round(step*i)),:])
            
        else:
            
            break
        
     
    #action_gpu = action.cuda()
    OutputData = np.copy(InputData)
        
    return InputData, OutputData, action  
 
def generation(inData, action, model, factor):
    number_of_layers = LAYERS
    L = number_of_layers - 1
     
    T =  inData.size()[0]
    
    
    torch.manual_seed(0)

    
    print('\n---------- Train a', str(L + 1), 'layer network ----------')
    print('Create the input image and target sequences')
    # inData = inData.view(T, 1, 8, 8)
    
    sequence_pred = torch.zeros((T, 1, 3, 4 * 2 ** (L),6 * 2 ** (L)))
    
    # input_sequence = Variable(torch.rand(T, 1, 3, 4 * 2 ** L, 6 * 2 ** L))
    if torch.cuda.is_available():
        input_sequence = Variable(inData).cuda()
        action_in = Variable(action).cuda()
    else:
    
        input_sequence = Variable(inData)
        action_in = Variable(action)
    print('Input has size', list(input_sequence.data.size()))
    
    
    # error_init_size_list = input_sequence.data.size()
    error_init_size_list = tuple(
        (1, ERR_LAYER_SIZE[l], 4 * 2 ** (L - l),  6 * 2 ** (L - l)) for l in range(0, L + 1)
    )
    
    print('The error initialisation sizes are', error_init_size_list)
    
    

    print('Load a', str(L + 1), 'layer Prednet')
    
    
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    
    # print L
    t1 = time.time()
    state = [[None] * (L+1)] * DIRECTION
    error = [None] * (L + 1)
    if torch.cuda.is_available() and sequence_pred.is_cuda == False:
        sequence_pred = Variable(sequence_pred).cuda()
    for t in range(0, T):
        if t == 0:
            inImage = input_sequence[t]
        else:
            
            inImage = input_sequence[t] * factor + sequence_pred[t,:,:,:,:] * (1-factor)
        
            
        error, state, prediction = model(inImage, error, state, action_in[t,:,:])
            
        sequence_pred[t,:,:,:,:] = (prediction[0].data)
            
            

    for i in range(0, T):
        
        sequence_save = sequence_pred.data[i,0,:,:,:].cpu()
               
        savematrix(sequence_save, "savetxt/prediction_" + "_" + str(i) +".txt")

        #print sequence_save
            
    t2 = time.time()
    print "Compute time: ~%f milliseconds" % ((t2 - t1) * 1000.)       
    
 # ADD THIS LINE
def normalisation(matrix):
    
    #print numpy.shape(matrix)
    
    cmax, cmin = matrix.max(), matrix.min()
    
    
    
    denominator  = cmax - cmin
    
    
    if denominator==0:
        denominator=numpy.finfo(matrix.dtype).eps
        
        
    return (matrix-cmin)/denominator


def gensmallData(size1, size2, nDirection):
        
        #print size1, size2
        
       
        
        length = size1 * size2 * nDirection
        
        
        OutputData = torch.ones((length, 1, 3, size1, size2))
        InputData = torch.ones((length, 1, 3, size1, size2)) 
        
        #InputData = InputData.cuda()
        
        action = torch.zeros((length, 1,  2))
        
        step = 0
        
        for direction in range(0, nDirection):
        
            if direction == 0:
        
            
                for l in range(0, size1*size2):
                    h = l / size2 
                    w = l % size2
                 
                    tmp = numpy.ones((size1,size2), dtype = numpy.double)    
                    #InputData[l,0, :, h,w]=1/1.1
                     
                    tmp[h,w] = 0.1
                    
                    
                    #else:
                        #InputData[l,0, :, h,0] = 1
                        
                    
                        
                        
                    action[step,0, 0] = 1
                    step += 1
                    
                    tmp = normalisation(tmp)
                    
                    InputData[l,0,0,:,:] = torch.from_numpy(tmp)
                    InputData[l,0,1,:,:] = InputData[l,0,0,:,:]
                    InputData[l,0,2,:,:] = InputData[l,0,0,:,:]
                        
            elif direction == 1:
            
             
                for l in range(0, size1*size2):
                        w = l / size1 
                        h = l % size1
                
                
                        tmp = numpy.ones((size1,size2), dtype = numpy.double)  
                        tmp[h,w] = 0.1
                        #InputData[l,0, :, h,w]=1/1.1
                        #if w + 1 < size2:
                        #    InputData[l,0, :, h,w+1] = 1
                        #else:
                        #    InputData[l,0, :, h,0] = 1
                        # InputData[l+size1*size2,0, :, h,w] = 0.1 
            
                        action[step,0, 1] = 1
                        step += 1
                        
                        
                        
                        tmp = normalisation(tmp)
                    
                        InputData[size1*size2 + l,0,0,:,:] = torch.from_numpy(tmp)
                       
                        InputData[size1*size2 + l,0,1,:,:] = InputData[size1*size2 + l,0,0,:,:]
                        InputData[size1*size2 + l,0,2,:,:] = InputData[size1*size2 + l,0,0,:,:]
                        
                                 
        for i in range(0, length):
             
            savematrix(InputData[i,0,:,:,:], "savetxt/input_"+str(i)+".txt")
            
       # OutputData = OutputData.cuda()      
        #action = action.cuda()
        
        #InputData = torch.tensor(InputData)
        return InputData, OutputData, action



def savematrix(array, filename):
    
    #save pytorch tensor into file
    
    size1 = array.size()[1]
    size2 = array.size()[2]
    pic = numpy.zeros((size1, size2), dtype=numpy.uint8)
    pic  = normalisation(array[0,:,:].numpy())
     
    numpy.savetxt(filename, pic)
 

   

if __name__ == '__main__':
    in_data1, out_data1, action1 = gensmallData(4*2**(LAYERS-1), 6*2**(LAYERS-1),2)
    #in_data1, out_data1, action1 = loadImage(4*2**(LAYERS-1), 6*2**(LAYERS-1))
    # in_data2, out_data2, action2 = gensmallData(4*2**1, 6*2**1,2)
    model = _test_training(in_data1, out_data1, action1)
    generation(in_data1, action1, model, 1.0)



import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets.utils import download_url
import torchvision.utils


def show_image(classes,img,label):
    print('Label is ' + str(classes[label]))
    plt.imshow(img.permute(1,2,0))

def show_batch(loader) :
   for images , labels in loader :
    grid = torchvision.utils.make_grid(images.to(torch.device('cpu')),nrow = 16)
    print(grid.shape)
    grid = grid.permute(1,2,0)
    print(grid.shape) 
    plt.axis('off')
    plt.imshow(grid)
    break
   

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for item in self.dl :
            yield to_device(item,self.device)
               

    def to_device(data,device):
        if isinstance(data,(list,tuple)) :
            return[to_device(x,device) for x in data]
        return data.to(device,non_blocking = True)

    def __len__(self):
        return len(self.dl)

def get_default_device():
    device_name = ''
    if torch.cuda.is_available() :
        device_name = 'cuda'
    else: 
        device_name = 'cpu' 
        
    return torch.device(device_name)

def to_device(data,device):
    if isinstance(data,(list,tuple)) :
        return[to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)

def to_default_device(data):
    return to_device(data,get_default_device())




class TrainingResult():

    def __init__(self):
        self.epoch_val_result_list = []
        self.current_epoch_result = EpochResult()
	
    def add_validation_result(self , validation_step_result):
    	self.current_epoch_result.validation_result_list.append(validation_step_result)

    def add_train_result(self , train_step_result):
    	self.current_epoch_result.train_result_list.append(train_step_result)
	
    def val_epoch_end(self):
        self.epoch_val_result_list.append(self.current_epoch_result)
        self.current_epoch_result.epoch_end()
        self.current_epoch_result = EpochResult()
        return self.epoch_val_result_list[-1]

    def log_last_epoc_result(self):
        num_epoch = len(self.epoch_val_result_list)
        last_epoch = self.epoch_val_result_list[-1]
        loss , acc = last_epoch.val_loss,last_epoch.val_accuracy
        print('Epoch ' + str(num_epoch) +  ' Validation Loss : ' + str(loss) + ' Validation Accuracy' + str(acc) + "\n")
		

class ValidationStepResult():
    def __init__(self,val_loss , acc) :
        self.val_loss = val_loss
        self.acc = acc

class TrainStepResult():
    def __init__(self,train_loss) :
        self.train_loss = train_loss


class EpochResult():

    def __init__(self) :
        self.validation_result_list = []
        self.train_result_list = []
		
    def add_validation_result(self , validation_step_result):
        self.validation_result_list.append(validation_step_result)

    def add_train_result(self , train_step_result):
        self.train_result_list.append(train_step_result)

    def epoch_end(self):
        self.calculate_val_loss()
        self.calculate_train_loss()
        
        self.validation_result_list = []
        self.train_result_list = []
    
    def calculate_val_loss(self):
        batch_losses = [x.val_loss for x in self.validation_result_list]
        batch_accuracy = [x.acc for x in self.validation_result_list]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        self.val_loss = epoch_loss.item()
        self.val_accuracy = epoch_accuracy.item()
	

    def calculate_train_loss(self):
        batch_losses = [x.train_loss for x in self.train_result_list]
        epoch_loss = torch.stack(batch_losses).mean()
        self.train_loss = epoch_loss.item()
            


def draw_training_result(result):
    accuracies = [each_epoch_result.val_accuracy for each_epoch_result in result.epoch_val_result_list]
    plt.plot(accuracies,'-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

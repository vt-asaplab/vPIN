import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchmetrics import Precision, Recall, F1Score

class LeNet5(nn.Module):
    """
    LeNet5 model implementation.
    """    
    def __init__(self, num_classes):
        """
        Initialize the LeNet5 model.
        """        
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU())
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU())
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forwardRegular(self, x):
        """
        Forward pass of the regular LeNet5 model.
        """
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.pool2(out)        
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)     
        return out

    def forwardFixedPointWithTrunc(self, x):
        """
        Forward pass of the fixed-point LeNet5 model with truncation.
        """
        out = self.layer1(x)
        out = self.truncation(out, 35)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.truncation(out, 33)
        out = self.pool2(out)  
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.truncation(out, 32)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.truncation(out, 33)
        out = self.fc2(out)
        return out

    def truncation(self, out, bits):
        """
        Truncate tensor values.
        """        
        out_real = self.fixedPointToRealNumbers(out, bits)
        truncated_out = self.realNumbersToFixedPoint(out_real, 16)

        return truncated_out

    def realNumbersToFixedPoint(self, inf, bits):
        """
        Convert real numbers to fixed-point representation.
        """        
        scale_factor = 2 ** bits
        scaled_image = inf * scale_factor
        rounded_image = torch.round(scaled_image)
        fixed_point_image = rounded_image.to(torch.float)

        return fixed_point_image

    def fixedPointToRealNumbers(self, inf, bits):
        """
        Convert fixed-point representation to real numbers.
        """
        scale_factor = 2 ** bits
        real_numbers = inf / scale_factor
        
        return real_numbers
        
def setVariables():
    """
    Set hyperparameters and device configuration.
    """
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10
    f = 16 # number of fractional bits
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return batch_size, num_classes, learning_rate, num_epochs, device, f

def loadDataset(batch_size):
    """
    Load and prepare the MNIST dataset.
    """
    train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize(mean = (0.1307,), std = (0.3081,))]), download = True)
    test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize(mean = (0.1325,), std = (0.3105,))]), download = True)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, test_loader

def initializeModelAndOptimizer(num_classes, device, learning_rate, train_loader):
    """
    Set up the model, loss function, and optimizer.
    """
    model = LeNet5(num_classes).to(device)
    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, cost, optimizer

def training(train_loader, num_epochs, device, model, cost, optimizer):
    """
    Train the LeNet5 model.
    """
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model.forwardRegular(images)
            loss = cost(outputs, labels)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            if (i+1) % 400 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

def testingRealNumber(test_loader, device, model, num_classes):
    """
    Test the LeNet5 model with real-number inputs.
    """
    precision = Precision(num_classes=num_classes, task='multiclass', average='macro').to(device)
    recall = Recall(num_classes=num_classes, task='multiclass', average='macro').to(device)
    f1 = F1Score(num_classes=num_classes, task='multiclass', average='macro').to(device)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forwardRegular(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            precision(predicted, labels)
            recall(predicted, labels)
            f1(predicted, labels)

        print('RealNumber: Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
        """
        print('RealNumber: Precision:', precision.compute()*100, "%")
        print('RealNumber: Recall:', recall.compute()*100,"%")
        print('RealNumber: F1-score:', f1.compute()*100,"%")
        """

def testingFixedPointWithTrunction(test_loader, device, model, bits, num_classes):
    """
    Test the LeNet5 model with fixed-point inputs and truncation.
    """
    precision = Precision(num_classes=num_classes, task='multiclass', average='macro').to(device)
    recall = Recall(num_classes=num_classes, task='multiclass', average='macro').to(device)
    f1 = F1Score(num_classes=num_classes, task='multiclass', average='macro').to(device)

    with torch.no_grad():
        correct1 = 0
        total1 = 0

        convertModelParametersToFixedPoint(model, bits)

        for images, labels in test_loader:
            
            fixed_point_images = realNumbersToFixedPointRepresentation(images, bits)
            fixed_point_images = fixed_point_images.to(torch.float)
            #print(fixed_point_images)
            fixed_point_images = fixed_point_images.to(device)
            labels = labels.to(device)
            
            outputs1 = model.forwardFixedPointWithTrunc(fixed_point_images)
            _, predicted1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted1 == labels).sum().item()

            precision(predicted1, labels)
            recall(predicted1, labels)
            f1(predicted1, labels)

        print('FixedPointWithTrunc: Accuracy of the network on the 10000 test images: {} %'.format(100 * correct1 / total1))
        """
        print('FixedPointWithTrunc: Precision:', precision.compute()*100, "%")
        print('FixedPointWithTrunc: Recall:', recall.compute()*100,"%")
        print('FixedPointWithTrunc: F1-score:', f1.compute()*100,"%")
        """

def realNumbersToFixedPointRepresentation(image, bits):
    """
    Convert a tensor of real numbers to fixed-point representation.
    """
    scale_factor = 2 ** bits
    scaled_image = image * scale_factor
    rounded_image = torch.round(scaled_image)
    fixed_point_image = rounded_image.to(torch.int)
    
    return fixed_point_image  

def convertModelParametersToFixedPoint(model, bits):
    """
    Convert all model parameters to fixed-point representation.
    """
    scale_factor = 2 ** bits
    for param in model.parameters():
        param.data = torch.round(param.data * scale_factor).to(torch.float)
        
def main():
    """
    Main function to set up, train, and test the LeNet5 model in both original and fixed-point representation settings.
    """
    batch_size, num_classes, learning_rate, num_epochs, device, f = setVariables()
    print(f"\nf (number of fractional bits) is {f} bits.")

    train_loader, test_loader = loadDataset(batch_size)
    model, cost, optimizer = initializeModelAndOptimizer(num_classes, device, learning_rate, train_loader)
    
    print("\nTraining...")
    training(train_loader, num_epochs, device, model, cost, optimizer)

    print("\nTesting...")
    testingRealNumber(test_loader, device, model, num_classes)
    testingFixedPointWithTrunction(test_loader, device, model, f, num_classes)
    print("\n")

if __name__ == "__main__":
    main()
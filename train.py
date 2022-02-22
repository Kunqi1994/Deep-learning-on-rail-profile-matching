import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import KQnet
from data import train_loader,validation_loader,train_dataset,validation_dataset

learning_rate = 0.0001
epoches = 1000
weights_path = 'matchingNet.pth'
train_number = len(train_dataset)
train_step = len(train_loader)
validation_number = len(validation_dataset)
validation_step = len(validation_loader)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    net = KQnet()
    net.to(device)
    loss_functional = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=learning_rate)
    best_accuracy = 0

    for epoch in range(epoches):
        net.train()
        running_loss = 0.0
        trainnumber = 0.0  #for calculating the right number of high-precision matching in train set
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(image)
            loss = loss_functional(output, label)
            loss.backward()
            optimizer.step()
            running_loss = running_loss+loss

            for number in range(32): # 32 is the batch size
                prediction_x = (output[number][0]-label[number][0]).item()
                prediction_y = (output[number][1]-label[number][1]).item()
                if prediction_x < 0.02 and prediction_y < 0.02:
                    trainnumber = trainnumber + 1

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epoches,
                                                                     loss)

        validation_loss = 0.0
        validationnumber = 0.0 #for calculating the right number of high-precision matching in validation set
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(validation_loader)
            for data in val_bar:
                image, label = data
                image = image.to(device)
                label = label.to(device)
                output = net(image)
                loss = loss_functional(output, label)
                validation_loss = validation_loss + loss

                for number in range(32):
                    prediction_x = (output[number][0] - label[number][0]).item()
                    prediction_y = (output[number][1] - label[number][1]).item()
                    if prediction_x < 0.02 and prediction_y < 0.02: #0.02 is the criterion of right matching
                        validationnumber = validationnumber + 1

        running_loss = (running_loss/train_step).item() #关于running_loss需要确认一下啊
        validation_loss = (validation_loss/validation_step).item()
        prediction_train = trainnumber/train_number
        prediction_val = validationnumber/validation_number

        print(epoch + 1, running_loss, validation_loss, prediction_train, prediction_val)

        with open('output.txt','w') as f:
            f.write(str(running_loss)+' '+str(validation_loss)+' '+str(prediction_train)+' '+str(prediction_val)+'\n')

        if prediction_val > best_accuracy:
            best_accuracy = prediction_val
            save_path = 'matchingNet.pth'
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()






from tqdm import tqdm
import torch
from model import KQnet
from data import test_loader,test_dataset


def main():
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(device)

    number = 0
    test_number = len(test_dataset)
    net = KQnet()
    weights_path = 'matchingNet.pth'
    net.load_state_dict(torch.load(weights_path))
    net.to(device)

    net.eval()

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            image, label = data
            image = image.to(device)
            label = label.to(device)
            output = net(image)
            prediction_x = (output[0][0]-label[0][0]).item()
            prediction_y = (output[0][1]-label[0][1]).item()
            if prediction_x < 0.02 and prediction_y < 0.02:
                number = number + 1

        prediction_accuracy = number/test_number
        print(prediction_accuracy)


if __name__ == '__main__':
    main()








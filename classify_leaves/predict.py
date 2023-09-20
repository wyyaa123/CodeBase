import torch
from tqdm import tqdm
from util.data_loader import BaseDataset, read_csv
from torch.utils.data import DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from util.net import Net

if __name__ == "__main__":

    _, unique_labels = read_csv("/home/nrc/classify_leaves/data/train.csv")

    test_transform = albu.Compose([
            albu.Resize(320, 320),
            albu.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)
        ])

    net = Net()
    net.load_state_dict(torch.load("./checkpoints/best0.pth"))
    net.to("cuda:0")

    val_data = read_csv("./data/sample_submission.csv", True)
    val_dataset = BaseDataset(val_data, test_transform, True)
    val_loader = DataLoader(val_dataset, 64, shuffle=False, drop_last=False)
    predictions = []
    predict_label = []
    with torch.no_grad():       
        for x in tqdm(val_loader):
            x = torch.as_tensor(x, dtype=torch.float)
            x = x.to("cuda:0")
            y_hat = net(x)
            predict = torch.argmax(y_hat,dim=1).reshape(-1)
            predict = list(predict.cpu().detach().numpy())
            predictions.extend(predict)

    for i in predictions:
        predict_label.append(unique_labels[i])

    val_data["label"] = predict_label

    val_data.to_csv("./test.csv", index=False)
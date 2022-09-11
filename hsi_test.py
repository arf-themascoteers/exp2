import torch
from hsi_dataset import HsiDataset
from torch.utils.data import DataLoader
import sklearn


def test(device):
    batch_size = 10
    cid = HsiDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss(reduction='mean')
    model = torch.load("models/hsi.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    loss_cum = 0
    itr = 0
    results = []

    ys = []
    yhats = []

    print(f"Actual SOC\t\t\tPredicted SOC")
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_hat = y_hat.reshape(-1)
        loss = criterion(y_hat, y)
        itr = itr+1
        loss_cum = loss_cum + loss.item()

        for i in range(y_hat.shape[0]):
            # gt = cid.unscale([y[i].item()])
            # hat= cid.unscale([y_hat[i].item()])
            # actual = f"{gt[0]:.1f}".ljust(20)
            # predicted = f"{hat[0]:.1f}".ljust(20)
            # print(f"{actual}{predicted}")
            ys.append(y[i].item())
            yhats.append(y_hat[i].item())

    r2 = sklearn.metrics.r2_score(ys, yhats)
    print(f"R-squared {r2}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)

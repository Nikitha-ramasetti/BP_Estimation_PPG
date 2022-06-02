import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, utils



#load saved data numpy files
ppg = np.load("data.npy")
target = np.load("label.npy")



def minMax(x):
    normalized = (x - min(x)) / (max(x) - min(x))
    return normalized


ppg = np.apply_along_axis(minMax, 1, ppg)
plt.plot(ppg.ravel())
plt.show()



#Split datasets into train test and val
X_train_val, X_test, y_train_val, y_test = train_test_split(ppg, target, test_size=0.2, random_state=1, shuffle=False)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,  test_size=0.2, random_state=1,shuffle=False)


# Initialize training dataloader
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32)
#print(len(train_loader))



# Initialize validation dataloader
valid_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
valid_loader = DataLoader(valid_dataset, batch_size=32)
#print(len(valid_loader))



# Initialize test dataloader
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=32)
#print(len(test_loader))



# Define Neural Network Architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Sequential block of layer1
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))


        # Sequential block of layer2
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))


        self.adaptive = nn.AdaptiveMaxPool1d(4)


        # lstm and fully connected layer
        self.fc1 = nn.Linear(128, 56)
        self.fc2 = nn.Linear(56, 3)


    def forward(self, x):
        #x = x.unsqueeze(-2)
        out = self.layer1(x)
        out = self.layer2(out)

        # adaptivemaxpool
        out = self.adaptive(out)
        # print(out.shape)

        # flatten
        out1 = out.reshape(out.size(0), -1)
        #print(out1.shape)


        # output layer
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.squeeze(-1)
        #print(out.shape)
        return out


#Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#Initialize the model
model = ConvNet()
model.to(device)


print(model)


# ========================================
# Training and Evaluation
# ========================================


#Model Parameters
batch_size = 32
n_epochs = 2
learning_rate = 0.001


criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Model
loss_stats = {
    'train': [],
    "val": []
}


print("Begin training")
for epoch in range(1, n_epochs + 1):
    #Training
    train_epoch_loss = 0

    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch)
        train_loss = criterion(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()

    # Validation
    with torch.no_grad():
        val_epoch_loss = 0

        model.eval()
        for X_val_batch, y_val_batch in valid_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch)
            val_loss = criterion(y_val_pred, y_val_batch)
            val_epoch_loss += val_loss.item()

    loss_stats['train'].append(train_epoch_loss)
    loss_stats['val'].append(val_epoch_loss)

    print(f"[{epoch}/{n_epochs}] | Train loss: {train_epoch_loss:.5f}\t | Val loss: {val_epoch_loss:.5f}")


# visualize Loss and Accuracy
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
plt.figure(figsize=(15,8))

sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')


# test Model
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
print(len(y_pred_list))


# mse error
mse = mean_squared_error(y_test, y_pred_list)
r_square = r2_score(y_test, y_pred_list)
print("Mean Squared Error :",mse)
print("R^2 :",r_square)
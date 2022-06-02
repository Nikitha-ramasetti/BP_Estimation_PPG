
from model.Optimization import*
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#load saved model numpy files
ppg = np.load("model.npy")
target = np.load("label.npy")


scaler = MinMaxScaler()
def minMax(x):
    #normalized = scaler.fit_transform(x.reshape(-1, 1))
    normalized = (x - min(x)) / (max(x) - min(x))
    return normalized


ppg_norm = np.apply_along_axis(minMax, 1, ppg)
plt.plot(ppg_norm.ravel())
plt.show()

#Split datasets into train test and val
X_train_val, X_test, y_train_val, y_test = train_test_split(ppg_norm, target, test_size=0.2, random_state=1, shuffle=False)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,  test_size=0.2, random_state=1,shuffle=False)


# Initialize training dataloader
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64)
print(len(train_loader))


# Initialize validation dataloader
valid_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
valid_loader = DataLoader(valid_dataset, batch_size=64)
print(len(valid_loader))


# Initialize test dataloader
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=64)
print(len(test_loader))


# Define Neural Network Architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Sequential block of layer1
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))

        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))

        self.adaptive = nn.AdaptiveMaxPool1d(4)


        # lstm and fully connected layer
        self.lstm = nn.LSTM(256, 56)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 3)


    def forward(self, x):
        x = x.unsqueeze(-2)
        out = self.layer1(x)
        out = self.layer2(out)

        # adaptive maxpool
        out = self.adaptive(out)
        # print(out.shape)

        # flatten
        out = out.reshape(out.size(0), -1)
        #print(out1.shape)

        # lstm layer
        out = out.unsqueeze(0)
        out, hid = self.lstm(out)
        # print(out.shape)

        # output layer
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.squeeze(-1)
        print(out.shape)
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

batch_size = 64
n_epochs = 100
learning_rate = 0.001

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, valid_loader, batch_size=batch_size, n_epochs=n_epochs)
opt.plot_losses()

predictions, values = opt.evaluate(test_loader)

print(predictions[0])
print(values[0])



#predictions
def format_predictions(predictions, values):
    vals = np.vstack(values)
    preds = np.vstack(predictions)
    df_result1 = pd.DataFrame(data={"value": vals[:, 0], "prediction": preds[:, 0]}, index=df_test.head(len(vals[:, 0])).index)
    df_result2 = pd.DataFrame(data={"value": vals[:, 1], "prediction": preds[:, 1]}, index=df_test.head(len(vals[:, 1])).index)
    df_result3 = pd.DataFrame(data={"value": vals[:, 2], "prediction": preds[:, 2]}, index=df_test.head(len(vals[:, 2])).index)
    return df_result1, df_result2, df_result3



#actuals labels
vals = np.vstack(values)
print(vals.shape)

#predicted labels
preds = np.vstack(predictions)
print(preds.shape)



# Calculate error metrics
def calculate_metrics(df):
   return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2' : r2_score(df.value, df.prediction)}

result_metrics1 = calculate_metrics(df_result1)
result_metrics2 = calculate_metrics(df_result2)
result_metrics3 = calculate_metrics(df_result3)

print("Label HR results: ", result_metrics1)
print("Label SBP results: ", result_metrics2)
print("Label DBP results: ", result_metrics3)


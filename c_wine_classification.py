import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

training_size = 120
test_size = 50

class_labels = [r"A", r"B", r"C"]

data, target = datasets.load_wine(return_X_y=True)
n_samples, n_features = data.shape

sample_train, sample_test, label_train, label_test = train_test_split(
    data, target, test_size=test_size, random_state=2234
)
#test
sc = StandardScaler()
sample_train = sc.fit_transform(sample_train)
sample_test = sc.transform(sample_test)

sample_train = torch.from_numpy(sample_train.astype(np.float32))
sample_test = torch.from_numpy(sample_test.astype(np.float32))
label_train = torch.from_numpy(label_train.astype(np.float32))
label_test = torch.from_numpy(label_test.astype(np.float32))

label_train = label_train.view(label_train.shape[0], 1)
label_test = label_test.view(label_test.shape[0], 1)

print(type(sample_train[1]))
print(label_test[4])
print(n_features, n_samples)

# model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        self.linear2 = nn.Linear(n_input_features, 1)

    def forward(self, x):
        # kann ich y_predicted hier auch anders nennen ohne es später zu verändern?
        y_predicted = torch.sigmoid(self.linear(x))
        y_predicted = torch.relu(self.linear2(x))
        return y_predicted


model = LogisticRegression(n_features)

# loss and optimizer
learning_rate = 0.2
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# training loopt
num_epochs = 100000

for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(sample_train)
    loss = criterion(y_predicted, label_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # empty gradients
    optimizer.zero_grad()

    if (epoch + 1) % 1000 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(sample_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(label_test).sum() / float(label_test.shape[0])
    print(f"accuracy = {acc:.4f}")

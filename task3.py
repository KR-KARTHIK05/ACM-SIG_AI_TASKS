import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
data=load_breast_cancer()
x=data.data
y=data.target
scaler=StandardScaler()
x=scaler.fit_transform(x)
x_trainset,x_testset,y_trainset,y_testset=train_test_split(x,y,test_size=0.2,random_state=42)
#test_size=0.2: This means 20% of the data will be allocated to the test set, and 80% will be used for training.
#random_state=42: A fixed random seed to ensure the split is reproducible. Every time you run the code, you will get the same split of data.
x_trainset=torch.tensor(x_trainset,dtype=torch.float32)
x_testset=torch.tensor(x_testset,dtype=torch.float32)
y_trainset=torch.tensor(y_trainset,dtype=torch.float32).unsqueeze(1)
y_testset=torch.tensor(y_testset, dtype=torch.float32).unsqueeze(1)
class BinaryclassifierNN(nn.Module):
    def __init__(self):
        super(BinaryclassifierNN, self).__init__()
        self.layer1 = nn.Linear(30, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        X = self.sigmoid(self.layer3(X))
        return X
model=BinaryclassifierNN()
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
num_epochs=100
loss_values=[]
for epoch in range(num_epochs):
     model.train()
     optimizer.zero_grad()
     outputs=model(x_trainset)
     loss=criterion(outputs,y_trainset)
     loss_values.append(loss.item())
     loss.backward()
     optimizer.step()
     if(epoch+1)%10==0:
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
model.eval()
with torch.no_grad():
    outputs = model(x_testset)
    predicted = (outputs > 0.5).float()
    acc = accuracy_score(y_testset, predicted)
    precision = precision_score(y_testset, predicted)
    recall = recall_score(y_testset, predicted)
    f1 = f1_score(y_testset, predicted)
    print(f'\n TEST SET EVALUATION')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve Over Epochs')
plt.show()


#the error is reducing from eppoch 10 to 100 from 0.6526 to 0.1948 . which tells us that the machine is becoming better and the model is learning from the training data
#In this case a lower loss means that the model's predictions is becoming accurate
     
         

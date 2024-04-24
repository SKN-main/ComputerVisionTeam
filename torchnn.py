import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Resize, Compose

# Pobieranie danych CIFAR-10
train = CIFAR10(root="data", download=True, train=True, transform=Compose([Resize((32, 32)), ToTensor()]))
dataset = DataLoader(train, batch_size=32, shuffle=True)

# Definiowanie sieci neuronowej klasyfikatora obrazów
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(), 
            nn.Linear(128 * 8 * 8, 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instancja sieci neuronowej, funkcja straty i optymalizator
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Trening modelu
if __name__ == "__main__": 
    for epoch in range(10): # trenujemy przez 10 epok
        for batch in dataset: 
            X, y = batch 
            X, y = X.to('cpu'), y.to('cpu') 
            yhat = clf(X) 
            loss = loss_fn(yhat, y) 

            # Zastosowanie wstecznej propagacji
            opt.zero_grad()
            loss.backward() 
            opt.step() 

        print(f"Epoka:{epoch} strata {loss.item()}")

    # Zapisanie stanu modelu
    with open('model_state_cifar10.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

    # Wczytanie modelu
    with open('model_state_cifar10.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    # Testowanie modelu na przykładowym obrazie
    img = Image.open('img_1.jpg') 
    transform = Compose([Resize((32, 32)), ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to('cpu')

    print(torch.argmax(clf(img_tensor)))

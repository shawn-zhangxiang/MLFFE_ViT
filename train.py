import torch
import numpy as np
import torchvision
import torch.optim as optim
from MLDWT_ViT import MLDWT_ViT
from typing import Dict, List, Tuple
from tqdm.notebook import tqdm
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import roc_curve, auc
torch.manual_seed(42)

model = MLDWT_ViT(image_size=224, num_classes=2)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    model.train()
    
    train_loss, train_acc = 0, 0
  
    for batch, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item()/len(y_pred)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              min_val_loss:float,
              device: torch.device) -> Tuple[float, float]:
    
    model.eval() 
    min_loss = min_val_loss
    print( f"min_loss: {min_loss:.4f}" )   
    val_loss, val_acc = 0, 0
   
    with torch.inference_mode():
      
        for batch, ( inputs, labels) in enumerate(dataloader):
        
            inputs, labels= inputs.to(device), labels.to(device)
            test_pred_logits = model(inputs)
            loss = loss_fn(test_pred_logits, labels)
            val_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            val_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    
    
    if(val_loss < min_loss):
        min_loss = val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_state_dict': loss_fn.state_dict(),
            }, 'save_best_model.pth')      
    
    return val_loss, val_acc, min_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
    }
    
    min_val_loss=10000
    
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        
        val_loss, val_acc, min_loss = val_step(model=model,
                                                dataloader=val_dataloader,
                                                loss_fn=loss_fn,
                                                min_val_loss=min_val_loss,
                                                device=device)

        
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f}"
        )
            
        min_val_loss=min_loss
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results



from timeit import default_timer as timer 
start_time = timer()

results = train(model=model,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=50,
                       device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

def plot_loss(results):
    
    loss = results["train_loss"]
    val_loss = results["val_loss"]
    epochs = range(len(results["train_loss"]))
    
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

plot_loss(results)

def plot_accuracy(results):   

    accuracy = results["train_acc"]
    val_accuracy = results["val_acc"]

    epochs = range(len(results["train_loss"]))
    plt.plot(epochs,torch.tensor(accuracy,device='cpu'), label="train_accuracy")
    plt.plot(epochs, torch.tensor(val_accuracy,device='cpu'), label="val_accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

plot_accuracy(results)
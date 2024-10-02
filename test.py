

# load the best model
checkpoint = torch.load('save_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


nb_classes = 2

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    
    test_running_corrects = 0.0
    test_total = 0.0
    model.eval()
    
    for i,(test_inputs, test_labels) in enumerate( test_dataloader, 0):
        test_inputs, test_labels = test_inputs.to(device='cuda:1'), test_labels.to(device='cuda:1')

        test_outputs = model(test_inputs.to(device='cuda:1'))
        _, test_outputs = torch.max(test_outputs, 1)
        
        test_total += test_labels.size(0)
        test_running_corrects += (test_outputs == test_labels).sum().item()
        
        for t, p in zip(test_labels.view(-1), test_outputs.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        
    print(f'Testing Accuracy: {(100 * test_running_corrects / test_total)}%')
    print(f'Confusion Matrix:\n {confusion_matrix}')
    

precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
f1_score = 2 * precision * recall / (precision + recall)

print(f'Precision: {precision: .4f}')
print(f'Recall   : {recall: .4f}')
print(f'F1 Score : {f1_score: .3f}')

# Calculate auc score
def test():
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_dataloader):
            output = model(data.to(device='cuda:1'))
            y_pred.append(output[:, 1].data.cpu().numpy())
            y_true.append(target.numpy())
            
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    print('AUC',roc_auc)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0.0, 1.0], [0.0, 1.0], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

test()    

#Plot the Confusion Matrix
test_preds = []
test_trues = []

with torch.no_grad():
  for i,(test_data_batch, test_data_label) in enumerate(test_dataloader):

    test_outputs = model(test_data_batch.to(device='cuda:0'))
    test_outputs = test_outputs.argmax(dim=1)
    test_preds.extend(test_outputs.detach().cpu().numpy())
    test_trues.extend(test_data_label.detach().cpu().numpy())

label_names=['fake','real']
confusion=confusion_matrix(test_trues, test_preds,labels=[i for i in range(len(label_names))])
plt.matshow(confusion, cmap=plt.cm.Blues)
plt.colorbar()

for i in range(len(confusion)):
  for j in range(len(confusion)):
    plt.annotate(confusion[j,i], xy=(i,j), horizontalalignment='center',verticalalignment='center')
  
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(range(len(label_names)), label_names)
plt.yticks(range(len(label_names)), label_names)
plt.show()
 


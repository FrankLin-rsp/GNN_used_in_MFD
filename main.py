import torch.nn as nn
import torch
import torch.nn.functional as F
from function_set import Batch_for_train, MFLPN

file_num = 500
batchsize = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MFLPN().to(device)
crossentropyloss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
Epoch_num = 100

for epoch in range(Epoch_num):
    loss_epoch = 0
    acc_epoch = 0

    for step in range(int(file_num / batchsize)):
        graph, node_rd, train_edge, train_edge_label, edge_for_mp, edge_for_predict = Batch_for_train(step, batchsize, device)

        #---------train----------
        model.train()
        optimizer.zero_grad()
        z = model.encode(graph, edge_for_mp, node_rd)
        link_logits = model.decode(z, train_edge, edge_for_predict)
        loss = crossentropyloss(link_logits, train_edge_label)
        loss.backward()
        optimizer.step()

        link_probs = F.softmax(link_logits, dim=1)
        train_pred = torch.argmax(link_probs, dim=1)
        train_correct = train_pred == train_edge_label
        train_acc = int(train_correct.sum()) / train_edge_label.shape[0]

        loss_epoch = loss.item() + loss_epoch
        acc_epoch = train_acc + acc_epoch

    log = 'Epoch: {:03d}, Loss: {:.4f}, Acc: {:.4f},'
    print(log.format(int(epoch+1), loss_epoch / int(file_num/batchsize), acc_epoch / int(file_num/batchsize)))















import argparse
import torch
import time
from model import TA_GNN
from torch_geometric.loader import DataLoader
from autotraining import autotraining
from dataset import session_graph
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--massege', type=str, default='', help='massege of rewrite')
opt = parser.parse_args()

start_time = time.time()
# Define hyper-parameters
varsion = 'TA_GNN'
dataset = 'diginetica'
batch_size = 100
learning_rate = 0.001
step_size = 3
device = torch.device('cpu')
gamma = 0.1
l2 = 1e-05
epochs = 10
topk = 20
# yoochoose_1_64 souers 37483  
# diginetica n_node 43097  
n_node = 43097
print(f"Using {device} device")

print('Starting to get dataset')
train = session_graph(f'datasets/{dataset}', 'train.dataset', f'./datasets/{dataset}/raw/train.txt')
test = session_graph(f'datasets/{dataset}', 'test.dataset', f'./datasets/{dataset}/raw/test.txt')

train_dataloader= DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size)

logdir = f'./log/{dataset}'
writer = SummaryWriter(logdir + opt.massege + '--' + varsion)

# Init all parameters of model
model = TA_GNN(n_node).to(device)
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_set = autotraining(
    model,
    loss_fun,
    optimizer,
    writer=writer,
    scheduler=scheduler,
    topk=topk,
    device=device
    )

print('Start training')
train_set.fit(train_dataloader, test_dataloader, epochs, log_parameter=False, eval=True)

# Save model results
print('Saving results')
writer.close()
## torch.save(model, f'./model_repository/{dataset}.model')
end_time = time.time()
total_seconds = end_time - start_time
minutes, seconds = divmod(total_seconds, 60)
print("Total duration {}min {}s".format(int(minutes), int(seconds)))
print('Done')


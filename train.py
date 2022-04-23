import torch.nn
from utils.dataset import ijgiDataset as Dataset
from models.sequenceencoder import LSTMSequentialEncoder
from utils.logger import Logger, Printer
import argparse
from utils.snapshot import save, resume
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=1 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=0, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=10, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default=None, type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default=None, type=str, help="directory to save checkpoints")
    return parser.parse_args()

def main(
    datadir,
    batchsize = 16,
    workers = 0,
    epochs = 10,
    lr = 1e-3,
    snapshot = None,
    checkpoint_dir = None
    ):

    traindataset = Dataset(datadir, tileids="tileids/train_fold0.tileids")
    testdataset = Dataset(datadir, tileids="tileids/test_fold0.tileids")

    nclasses = len(traindataset.classes)

    traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=True,num_workers=workers)
    testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize,shuffle=False,num_workers=workers)

    logger = Logger(columns=["loss"], modes=["train", "test"])


    network = Model()

    optimizer = torch.optim.Adam(network.parameters(), lr=lr,weight_decay=1e-3)
    loss = torch.nn.NLLLoss()

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network).cuda()
        loss = loss.cuda()

    start_epoch = 0

    if snapshot is not None:
        state = resume(snapshot,model=network, optimizer=optimizer)

        if "epoch" in state.keys():
            start_epoch = state["epoch"] + 1

        if "data" in state.keys():
            logger.resume(state["data"])

    for epoch in range(start_epoch, epochs):

        logger.update_epoch(epoch)

        print("\nEpoch {}".format(epoch))
        print("train")
        
        train_epoch(traindataloader, network, optimizer, loss, logger)
        print("\ntest")
        
        test_epoch(testdataloader, network,loss, logger)

        data = logger.get_data()

        if checkpoint_dir is not None:
            checkpoint_name = os.path.join(checkpoint_dir,"model_{:02d}.pth".format(epoch))
            save(checkpoint_name, network, optimizer, epoch=epoch, data=data)
        logger.save_csv('/content/model/epoch_data.csv')

def train_epoch(dataloader, network, optimizer, loss, logger):
    network.train()
    printer = Printer(N=len(dataloader))
    logger.set_mode("train")

    for iteration, data in enumerate(dataloader):
        if iteration%4 == 0:
            optimizer.zero_grad()
            batch_loss = 0
        
        input, target = data
        output,target = network.forward(input,target)
        l = loss(output, target)
        batch_loss += l/4
        l.backward()

        if (iteration+1)%4 == 0:
            stats = {"loss":batch_loss.data.cpu().numpy()}
            optimizer.step()
            
            printer.print(stats, iteration)
            logger.log(stats, (iteration+1)/4)

def test_epoch(dataloader, network, loss, logger):
    network.eval()
    printer = Printer(N=len(dataloader))
    logger.set_mode("test")

    with torch.no_grad():
        for iteration, data in enumerate(dataloader):
            if iteration%4 == 0:
                batch_loss = 0

            input, target = data
            
            output,target = network.forward(input,target)
            l = loss(output, target)
            batch_loss += l/4

            if (iteration+1)%4 == 0:
                stats = {"loss":batch_loss.data.cpu().numpy()}

                printer.print(stats, iteration)
                logger.log(stats, (iteration+1)/4)

if __name__ == "__main__":


    main(
        'data',
        batchsize=4,
        workers=8,
        epochs=100,
        lr=1e-3,
        snapshot=None,
        checkpoint_dir='/content/model'
    )

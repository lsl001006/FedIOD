import torch
import torch.optim as optim
import config
import utils
import logging
import os

def trainLocal(model, train_loader, savename, test_loader=None, writer=None, writermark='local'):
    epochs = config.INIT_EPOCHS
    criterion = torch.nn.CrossEntropyLoss() #include softmax
    optimizer = optim.SGD(model.parameters(), 
                            config.lr,
                            momentum=0.9,
                            weight_decay=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(epochs), eta_min=config.lrmin)
    #
    bestacc = 0 
    bestname = ''
    for epoch in range(epochs):
        #train
        model.train()
        tracc = utils.AverageMeter()
        trloss = utils.AverageMeter()
        for i, (images, target, _) in enumerate(train_loader):
            images = images.cuda()
            target = target.cuda()
            output = model(images)
            # import ipdb; ipdb.set_trace()
            loss = criterion(output, target)
            acc,  = utils.accuracy(output, target)
            tracc.update(acc)
            trloss.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info(f'loss={trloss.avg}, acc={tracc.avg}')
        if writer is not None:
            writer.add_scalars(str(writermark)+'train', {'loss': trloss.avg}, epoch)
            writer.add_scalars(str(writermark)+'train', {'acc': tracc.avg}, epoch)
        #val
        if test_loader is not None:
            model.eval()
            testacc = utils.AverageMeter()
            with torch.no_grad():
                for i, (images, target, _) in enumerate(test_loader):
                    images = images.cuda()
                    target = target.cuda()
                    output = model(images)
                    acc, = utils.accuracy(output, target)
                    testacc.update(acc)
                if writer is not None:
                    writer.add_scalar(str(writermark)+'testacc', testacc.avg, epoch)
                if testacc.avg > bestacc:
                    bestacc = testacc.avg
                    if bestname:
                        os.system(f'rm {bestname}')
                    bestname = f'{savename[:-3]}_{(bestacc):.2f}.pt'
                    torch.save(model.state_dict(), bestname)
                    os.system(f'cp {bestname} {savename}')
                logging.info(f'{writermark}, Epoch={epoch}: testacc={testacc.avg}, Best======{bestacc}======')
        #
        scheduler.step()

    return bestacc
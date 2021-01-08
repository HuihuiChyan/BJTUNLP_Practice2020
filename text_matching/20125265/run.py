import torch
import torch.nn as nn
from data_process import load_data
from utils import train, validate, test
from model import ESIM
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--hidden_size', type=int, default=512)
#parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--max_grad_norm', type=float, default=0.5)
parser.add_argument('--target_dir', type=str, default='./models_6')
#parser.add_argument('--ckp', type=str, default=None)
parser.add_argument('--ckp', type=str, default='best.pth.tar')
args = parser.parse_args()

id2label = {0:'neutral', 1:'contradiction', 2:'entailment'}

def main():
    device = args.device
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    # -------------------- Data loading ------------------- #
    print("Loading data......")
    train_loader, dev_loader, test_loader, SEN1, SEN2 = load_data(args.batch_size, device)
    embedding = SEN1.vectors
    vocab_size = len(embedding)
    print("vocab_size:", vocab_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = ESIM(args.hidden_size, embedding=embedding, dropout=args.dropout, num_labels=args.num_classes, device=device).to(device)
    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    best_score = 0.0
    if args.ckp:
        checkpoint = torch.load(os.path.join(args.target_dir, args.ckp))
        best_score = checkpoint["best_score"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _, valid_loss, valid_accuracy = validate(model, dev_loader, criterion)
        print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss,(valid_accuracy * 100)))

    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training ESIM model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(args.num_epoch):
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       criterion, args.max_grad_norm, device)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validate(model, dev_loader, criterion, device)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            print("save model！！！！")
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "optimizer": optimizer.state_dict(),
                        },
                       os.path.join(args.target_dir, "best.pth.tar"))

        if patience_counter >= 5:
            print("-> Early stopping: patience limit reached, stopping...")
            break


    # ##-------------------- Testing epochs ------------------- #
    # print(20 * "=", " Testing ", 20 * "=")
    # if args.ckp:
    #     checkpoint = torch.load(os.path.join(args.target_dir, args.ckp))
    #     best_score = checkpoint["best_score"]
    #     model.load_state_dict(checkpoint["model"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #
    # print("best_score:", best_score)
    # all_labels = test(model, test_loader, device)
    # print(all_labels[:10])
    # target_label = [id2label[id] for id in all_labels]
    # print(target_label[:10])
    # with open(os.path.join(args.target_dir, 'result.txt'), 'w+') as f:
    #     for label in target_label:
    #         f.write(label + '\n')

    del train_loader
    del dev_loader
    del test_loader
    del SEN1
    del SEN2
    del embedding


if __name__ == "__main__":
    main()




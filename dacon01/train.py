import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


def run_epoch(data_loader, model, criterion, epoch, is_training, optimizer=None):
    model = model.cuda()
    if is_training:
        model.train()
        logger_prefix = 'train'
    else:
        model.eval()
        logger_prefix = 'val'

    confusion_matrix = tnt.meter.ConfusionMeter(num_class)
    acc = tnt.meter.ClassErrorMeter(accuracy=True)
    meter_loss = tnt.meter.AverageValueMeter()

    for batch_idx, sample in enumerate(data_loader):
        sequence = sample['seq']
        label = sample['label']
        input_sequence_var = Variable(sequence.cuda())
        input_label_var = Variable(label.cuda())

        # compute output
        # output_logits: [batch_size, num_class]
        output_logits = model(input_sequence_var)
        #         output_logits = output_logits.cpu()
        loss = criterion(output_logits, input_label_var)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        meter_loss.add(loss.data[0])
        acc.add(output_logits.data, input_label_var.data)
        confusion_matrix.add(output_logits.data, input_label_var.data)

    print('%s Epoch: %d  , Loss: %.4f,  Accuracy: %.2f' % (logger_prefix, epoch, meter_loss.value()[0], acc.value()[0]))
    return acc.value()[0]


num_epochs = 1000
evaluate_every_epoch = 5

for e in range(num_epochs):
    run_epoch(trLD, model, criterion, e, True, optimizer)
    if e % evaluate_every_epoch == 0:
        run_epoch(valLD, model, criterion, e, False, None)
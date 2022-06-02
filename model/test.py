#specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#specify loss function
loss_function = nn.MSELoss(reduction='mean')

es = EarlyStopping(patience=15)

# Initialization of our experimentation and network training
exp = Experiment('./simple_example',
model,
optimizer=optimizer,
task='regression',
loss_function=loss_function,
epoch_metrics=[SKLearnMetrics([r2_score])]
)

#train
exp.train(train_loader, valid_loader, epochs=85, callbacks=[es])



# Test
exp.test(test_loader)

logs = pd.read_csv('/Users/eshwa/PycharmProjects/cnn-lstm/simple_example/log.tsv', sep='\t')
print(logs)

best_epoch_idx = logs['val_loss'].idxmax()
best_epoch = int(logs.loc[best_epoch_idx]['epoch'])
print("Best epoch: %d" % best_epoch)


metrics = ['loss', 'val_loss']
plt.plot(logs['epoch'], logs[metrics])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(metrics)
plt.title('Loss vs. No. of epochs')
plt.show()
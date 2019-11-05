import matplotlib.pyplot as plt
import os

def draw_loss(name='test', iters=None, train_loss=None, test_loss=None, accuracy=None):
    plt.clf()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.set_xlabel('iters')
    ax1.set_ylabel('loss')

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')

    lns = []

    if train_loss:
        p, = ax1.plot(iters, train_loss, label='train loss')
        lns.append(p)
    if test_loss:
        p, = ax1.plot(iters, test_loss, label='test loss')
        lns.append(p)
    if accuracy:
        p, = ax2.plot(iters, accuracy, label='accuracy', color='tab:red')
        lns.append(p)

    ax1.legend(handles=lns, loc='upper left')
    fig.tight_layout()
    plt.savefig( 'plots/' +  name + '.png')
    plt.close('all')
    
def draw_accuracy(train_acc=None, test_acc=None):
    pass

# тренирует и сохраняет параметры(torch.save)
# %%
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # for making figures

# read in all the words
words = open('train.txt', 'r').read().splitlines()
print(words[:8])

print('length: ', len(words))
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
print(itos)

# build the dataset
block_size = 3  # context length: how many characters do we take to predict the next one? the more the better, but only if we have a great engine


def build_dataset(words):
    X, Y = [], []
    for w in words:

        # print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            #print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print('X.shape: ', X.shape, 'Y.shape: ', Y.shape)
    return X, Y


random.seed(42)
random.shuffle(words)
# n1 = int(0.8*len(words))
# n2 = int(0.9*len(words))

# divide dataset
# Xtr, Ytr = build_dataset(words[:n1])
# Xdev, Ydev = build_dataset(words[n1:n2])
# Xte, Yte = build_dataset(words[n2:])
Xtr, Ytr = build_dataset(words)

C = torch.randn((len(itos), 2))
ys = C[:, 1]
xs = C[:, 0]
plt.scatter(xs, ys)

tmp = torch.arange(6).view(-1, 3)
print('tmp: ', tmp)
print('C[tmp]:', C[tmp])

emb = C[Xtr]
# output example: torch.Size([7939 (Xtr size), 3 (symbols), 2 (instead of len(itos))])
print('emb.shape: ', emb.shape)

# 1st layer w1 - matrix of weights (6 enters, 100 exits), b1 - bias:
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
print('emb.view(-1, 6).shape:', emb.view(-1, 6).shape)
# from {emb.shape:  torch.Size([7939, 3, 2])}: -1 - to not change the size of the 1st element, 6 - to concatenate 3 and 2

# to get the 1st layer (tanh - activation function, may be any):
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
print('h: ', h)
print('h.shape: ', h.shape)

# 2nd layer
W2 = torch.randn((100, len(itos)))
b2 = torch.randn(len(itos))

logits = h @ W2 + b2
print('logits.shape: ', logits.shape)

counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
print('prob.shape: ', prob.shape)

print('Xtr.shape: ', Xtr.shape, 'Ytr.shape: ', Ytr.shape)

# ?????
#loss = -prob[torch.arange(27), Y].log().mean()
# loss

# all in all:
g = torch.Generator().manual_seed(2147483647)  # for reproducibility
C = torch.randn((len(itos), 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, len(itos)), generator=g)
b2 = torch.randn(len(itos), generator=g)
parameters = [C, W1, b1, W2, b2]

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
             ha="center", va="center", color='white')
plt.grid('minor')

print('sum: ', sum(p.nelement()
      for p in parameters))  # number of parameters in total

for p in parameters:
    p.requires_grad = True

# search for the training parameter, learning rate, should be great at first, then small
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []

for i in range(20000):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emb = C[Xtr[ix]]  # (32, 3, 10)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 200)
    logits = h @ W2 + b2  # (32, 27)
    loss = F.cross_entropy(logits, Ytr[ix])
    # print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    #lr = lrs[i]
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())

# print(loss.item())
plt.plot(stepi, lossi)

# loss the smaller the better (in lecture 2.35 was ok)
emb = C[Xtr]  # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
logits = h @ W2 + b2  # (32, 27)
loss = F.cross_entropy(logits, Ytr)
print('loss: ', loss)

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
             ha="center", va="center", color='white')
plt.grid('minor')

context = [0] * block_size
print('C[torch.tensor([context])].shape: ', C[torch.tensor([context])].shape)

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

    out = []
    context = [0] * block_size  # initialize with all ...
    while True:
        emb = C[torch.tensor([context])]  # (1,block_size,d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

# Save the model
torch.save({'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, 'model.torch')

# %%

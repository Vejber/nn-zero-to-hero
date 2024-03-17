# загружает параметры нейронной сети из файла model.torch
# загружает тест датасет и его прогоняет

# %%
import random
import torch
import torch.nn.functional as F

# read file 'test.txt'
words = open('test.txt', 'r').read().splitlines()
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


# random.seed(42)
# random.shuffle(words)
# n1 = int(0.8*len(words))
# n2 = int(0.9*len(words))

# divide dataset
# Xtr, Ytr = build_dataset(words[:n1])
# Xdev, Ydev = build_dataset(words[n1:n2])
# Xte, Yte = build_dataset(words[n2:])
Xte, Yte = build_dataset(words)

# Loading a Saved Model
model = torch.load('model.torch')
C, W1, b1, W2, b2 = model['C'], model['W1'], model['b1'], model['W2'], model['b2']
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

emb = C[Xte]  # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
logits = h @ W2 + b2  # (32, 27)
loss = F.cross_entropy(logits, Yte)
print('loss: ', loss)

for _ in range(5):

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

    for i in out:
        try:
            print(itos[i], end='')
        except KeyError:
            # print(f'[UNKNOWN INDEX: {i}]', end='')
            print(f'*', end='')
    print()
    # print(''.join(itos[i] for i in out))

# %%

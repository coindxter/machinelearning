from collections import Counter

def merge(tokens, pair, new_token):
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(new_token)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

def train_bpe(text, num_merges):
    tokens = list(text)
    merges = []
    for _ in range(num_merges):
        pairs = Counter(zip(tokens, tokens[1:]))
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        new_token = best[0] + best[1]
        tokens = merge(tokens, best, new_token)
        merges.append((best, new_token))
    return merges

def tokenize(text, merges):
    tokens = list(text)
    for pair, new_token in merges:
        tokens = merge(tokens, pair, new_token)
    return tokens

corpus = (
    "the quick brown fox jumps over the lazy dog. "
    "the lowest lower low newer newest wider widest. "
    "learning is fun and training the network is interesting. "
    "the thing is that the weather is nothing like the other things. "
) * 5

merges = train_bpe(corpus, 60)

vocab = {t: i for i, t in enumerate(sorted(set(corpus) | {n for _, n in merges}))}
vocab["<unk>"] = len(vocab)

print(f"vocab size: {len(vocab)}")
print("-" * 30)

while True:
    word = input("text: ")
    tokens = tokenize(word, merges)
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    print(tokens)
    print(ids)
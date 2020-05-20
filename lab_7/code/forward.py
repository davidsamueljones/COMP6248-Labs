def forward(self, src):
    embedded = self.embedding(src)
    _, (hidden, cell) = self.rnn(embedded)
    return (hidden, cell)
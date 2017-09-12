import textGenerator.graph as graph
import textGenerator.reader
import time
import matplotlib.pyplot as plt


def main():
    reader = textGenerator.reader.Reader("input.txt")
    g = graph.build_graph(reader.len, cell_type='LSTM', num_steps=72,learning_rate=0.1,state_size=128, build_with_dropout=True)
    t = time.time()
    losses = graph.train_network(g, reader.gen_epochs(1, 32, 72), save="model/GRU_20_epochs")
    print("It took", time.time() - t, "seconds to train for 20 epochs.")
    print("The average loss on the final epoch was:", losses[-1])
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()

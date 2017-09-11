import textGenerator.graph as graph
import textGenerator.reader
import time


def main():
    reader = textGenerator.reader.Reader("三国演义.txt")
    g = graph.build_graph(reader.len, cell_type='GRU', num_steps=72,learning_rate=0.001,state_size=256)
    t = time.time()
    losses = graph.train_network(g, reader.gen_epochs(20, 32, 72), save="model/GRU_20_epochs")
    print("It took", time.time() - t, "seconds to train for 20 epochs.")
    print("The average loss on the final epoch was:", losses[-1])


if __name__ == "__main__":
    main()

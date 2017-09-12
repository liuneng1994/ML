import textGenerator.graph as graph
import textGenerator.reader

reader = textGenerator.reader.Reader("input.txt")
print(reader.to_ids())
g = graph.build_graph(reader.len, cell_type='LSTM', num_steps=1, batch_size=1,state_size=128,num_layers=4)
graph.generate_characters(g, "model/GRU_20_epochs", 750, reader, prompt='A', pick_top_chars=5)
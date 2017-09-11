import textGenerator.graph as graph
import textGenerator.reader

reader = textGenerator.reader.Reader("三国演义.txt")
g = graph.build_graph(reader.len, cell_type='GRU', num_steps=1, batch_size=1,state_size=128)
graph.generate_characters(g, "model/GRU_20_epochs", 750, reader, prompt='关', pick_top_chars=20)
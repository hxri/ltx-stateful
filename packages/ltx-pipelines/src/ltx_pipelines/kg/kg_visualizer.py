import networkx as nx
import matplotlib.pyplot as plt

def render_graph(graph):
    G = nx.Graph()

    for node_id in graph.nodes:
        G.add_node(node_id)

    pos = nx.spring_layout(G)
    fig = plt.figure()
    nx.draw(G, pos, with_labels=True, node_color="lightblue")
    return fig

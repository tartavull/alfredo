"""
Code for visualization of network architectures.
"""
import matplotlib.pyplot as plt
import mplcursors
import networkx as nx


def draw_graph(G):
    pos = {item: (i, 0) for i, item in enumerate(nx.topological_sort(G))}
    ax = plt.gca()
    for edge in G.edges:
        source, target = edge
        rad = 0.8 if pos[source][0] % 2 else -0.8
        ax.annotate(
            "",
            xytext=(pos[source][0], 0),  # start of the arrow
            xy=(pos[target][0], 0),  # end of the arrow
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                connectionstyle=f"arc3,rad={rad}",
                alpha=0.3,
                linewidth=1.5,
            ),
        )
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_size=800, node_color="black")
    labels = nx.draw_networkx_labels(G, pos=pos, font_color="white", font_size=8)

    cursor = mplcursors.cursor(nodes, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_visible(False)
        node_index = sel.index
        node_label = list(labels.keys())[node_index]
        node_data = G.nodes[node_label]

        table_rows = "\n".join([f"{k}: {v}" for k, v in node_data.items()])
        sel.annotation.set_text(table_rows)
        sel.annotation.get_bbox_patch().set_boxstyle("round,pad=0.3")
        sel.annotation.set_visible(True)

    @cursor.connect("remove")
    def on_remove(sel):
        sel.annotation.set_visible(False)

    plt.box(False)
    plt.show()

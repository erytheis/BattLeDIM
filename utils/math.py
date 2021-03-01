def convert_flow_to_hourly(flow):
    return flow * 3600


def conver_frlow_seconds(flow):
    return flow / 3600


def get_nodes_of_edge(G, pipe):
    for node_1, node_2, pipe_name in G.edges:
        if pipe_name == pipe:
            return (node_1, node_2)

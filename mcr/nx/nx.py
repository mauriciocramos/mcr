from collections import defaultdict
from multiprocessing import Pool, cpu_count
from itertools import combinations, islice
import networkx as nx
from matplotlib import pyplot as plt


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None, chunk_factor=4):
    """
    Parallel betweenness centrality  function
    https://networkx.org/documentation/stable/auto_examples/algorithms/plot_parallel_betweenness.html
    """
    # TODO: adjust processes to max number of nodes
    if processes is None:
        processes = cpu_count()
    processes = min(processes, max(1, G.order() / (processes * chunk_factor)))
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * chunk_factor
    # list of node chunks
    node_chunks = list(chunks(G.nodes(), G.order() // node_divisor))
    num_chunks = len(node_chunks)
    # starmap support multiple arguments to func
    bt_sc = p.starmap(
        # betweenness_centrality_subset(G, sources targets, normalized=False, weight=None) -> dict
        func=nx.betweenness_centrality_subset,  # normalized=True, weight=None
        iterable=zip([G] * num_chunks,  # num_chunks sized list of references to G: G
                     node_chunks,  # num_chunks sized list of node slices [(1,2,3,...),(4,5,6)]: sources
                     [list(G)] * num_chunks,  # num_chunks sized repeated list of all nodes: targets
                     [True] * num_chunks,  # num_chunks sized list of Trues: normalized=True
                     [None] * num_chunks)  # num_chunks sized list of Nones: weight=None
    )
    # Reduce the partial solutions: list of dictionaries betweenness centrality of all chunks
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


def plot_betweenness_centrality(G, figsize=(19.2, 10.8 / 2), bins=None, processes=None):
    bet_cen = list(betweenness_centrality_parallel(G, processes=processes).values())
    deg_cen = list(nx.degree_centrality(G).values())
    plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.hist(bet_cen, bins=bins)
    plt.xlabel('Betweenness centrality')
    plt.subplot(1, 3, 2)
    plt.hist(deg_cen, bins=bins)
    plt.xlabel('Degre centrality')
    plt.subplot(1, 3, 3)
    plt.scatter(bet_cen, deg_cen)
    for i, txt in enumerate(G):
        plt.annotate(txt, (bet_cen[i], deg_cen[i]))
    plt.xlabel('Betweenness centrality')
    plt.ylabel('Degree centrality')
    plt.show()


def plot_degree_centrality(G, figsize=(19.2, 10.8 / 2), bins=None):
    dec_cent = list(nx.degree_centrality(G).values())
    degrees = [len(list(G.neighbors(n))) for n in G.nodes()]
    plt.figure(figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.hist(dec_cent, bins=bins)
    plt.xlabel('Degree Centrality')
    plt.subplot(1, 3, 2)
    plt.hist(degrees, bins=bins)
    plt.xlabel('Degree (number of neighbors)')
    plt.subplot(1, 3, 3)
    plt.scatter(x=degrees, y=dec_cent)
    plt.xlabel('Degree (number of neighbors)')
    plt.ylabel('Degree Centrality')
    plt.show()


def find_nodes_with_highest_betweenness_centrality(G, processes=None):
    """Find the node(s) that has the highest betweenness centrality in the network"""

    # Compute betweenness centrality
    # Experimenting betweenness_centrality_parallel
    bet_cent = betweenness_centrality_parallel(G, processes=processes)

    # Compute maximum betweenness centrality: max_bc
    max_bc = max(list(bet_cent.values()))
    nodes = set()
    # Iterate over the betweenness centrality dictionary
    for k, v in bet_cent.items():
        # Check if the current value has the maximum betweenness centrality
        if v == max_bc:
            # Add the current node to the set of nodes
            nodes.add(k)
    return nodes


def find_nodes_with_highest_degree_centrality(G):
    """Find the node(s) that has the highest degree centrality in a graph"""

    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)
    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values()))
    nodes = []
    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():
        # Check if the current value has the maximum degree centrality
        if v == max_dc:
            # Add the current node to the set of nodes
            nodes.append(k)
    return nodes


def find_selfloop_nodes(G):
    """Finds all nodes that have self-loops in the graph G."""
    nodes_in_selfloops = []
    # Iterate over all the edges of G
    for u, v in G.edges():
        # Check if node u and node v are the same
        if u == v:
            # Append node u to nodes_in_selfloops
            nodes_in_selfloops.append(u)
    return nodes_in_selfloops


# Check whether number of self loops equals the number of nodes in self loops
# assert T.number_of_selfloops() == len(find_selfloop_nodes(T))


def nodes_with_m_neighbors(G, m):
    """Returns all nodes in graph G that have m neighbors."""
    nodes = set()
    # Iterate over all nodes in G
    for n in G.nodes():
        # Check if the number of neighbors of n matches m
        if len(list(G.neighbors(n))) == m:
            # Add the node n to the set
            nodes.add(n)
    # Return the nodes with m neighbors
    return nodes


def path_exists(G, node1, node2):
    """Checks whether a path exists between two nodes (node1, node2) in graph G."""
    assert node1!=node2, "must be different nodes"
    visited_nodes = set()
    # starting queue
    queue = [node1]
    # queue dinamically extended with unvisited neighbors of the node if node2 isn't a neighbor of node
    for node in queue:
        neighbors = list(G.neighbors(node))
        # Add current node to visited nodes
        visited_nodes.add(node)
        if node2 in neighbors:
            return True
        else:
            # Add neighbors of current node that have not yet been visited...
            # ...avoiding loops and self-loops:
            not_visited_neighbors = [n for n in neighbors if n not in visited_nodes]
            # print(f'{not_visited_neighbors}')
            queue.extend(not_visited_neighbors)
            # Check to see if the final element of the queue has been reached
            if node == queue[-1]:
                return False


def is_in_triangle(G, n):
    """Checks whether a node `n` in graph `G` is in a triangle relationship or not."""
    in_triangle = False
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
        # Check if an edge exists between n1 and n2
        if G.has_edge(n1, n2):
            in_triangle = True
            # early stop
            break
    return in_triangle


def nodes_in_triangle(G, n):
    """Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`."""
    triangle_nodes = {n}
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
        # Check if n1 and n2 have an edge between them
        if G.has_edge(n1, n2):
            # Add n1 to triangle_nodes
            triangle_nodes.add(n1)
            # Add n2 to triangle_nodes
            triangle_nodes.add(n2)
    return triangle_nodes


def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):
            in_open_triangle = True
            break

    return in_open_triangle


def recommend_connections(G, top=10):
    recommended = defaultdict(int)
    for n, d in G.nodes(data=True):
        for n1, n2 in combinations(G.neighbors(n), 2):
            if not G.has_edge(n1, n2):  # open triangle
                recommended[(n1, n2)] += 1
    sorted_counts = sorted(recommended.values())
    return [pair for pair, count in recommended.items() if count > sorted_counts[-10]]


def maximal_cliques(G, size):
    """Finds all maximal cliques in graph `G` that are of size `size`."""
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs
# # Check that there are 33 maximal cliques of size 3 in the graph T
# assert len(maximal_cliques(T, 3)) == 33


def get_nodes_and_neighbors(G, nodes_of_interest):
    """Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors."""
    nodes_to_draw = []
    # Iterate over the nodes of interest
    for n in nodes_of_interest:
        # Append the nodes of interest to nodes_to_draw
        nodes_to_draw.append(n)
        # Iterate over all the neighbors of node n
        for nbr in G.neighbors(n):
            # Append the neighbors of n to nodes_to_draw
            nodes_to_draw.append(nbr)
    return G.subgraph(nodes_to_draw)


def get_node_attributes(G):
    return {k for n, kv in G.nodes(data=True) for k, v in kv.items()}


def get_edge_attributes(G):
    return {k for n1, n2, kv in G.edges(data=True) for k, v in kv.items()}


def get_node_attributes_values(G):
    return {k: v for n, kv in G.nodes(data=True) for k, v in kv.items()}


def get_edge_attributes_values(G):
    return {k: v for n1, n2, kv in G.edges(data=True) for k, v in kv.items()}


def find_largest_clique(G):
    return sorted(nx.find_cliques(G), key=lambda x: len(x))[-1]


def get_largest_clique_with_neighbors(G):
    largest_clique = find_largest_clique(G)
    G_lc = G.subgraph(largest_clique).copy()
    # Go out 1 degree of separation
    for node in list(G_lc.nodes()):
        G_lc.add_nodes_from(G.neighbors(node))
        G_lc.add_edges_from(zip([node] * len(list(G.neighbors(node))), G.neighbors(node)))
    return G_lc

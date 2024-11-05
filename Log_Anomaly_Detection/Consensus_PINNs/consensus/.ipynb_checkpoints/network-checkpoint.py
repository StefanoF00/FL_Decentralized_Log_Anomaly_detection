import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random 

# GRAPH GENERATION
def create_specific_undirected_graph():
    """
    Create a specific undirected graph with predefined nodes and edges.
    
    Returns:
    - A: Adjacency matrix of the graph.
    - D: Degree matrix of the graph.
    - L: Laplacian matrix of the graph.
    """
    # Create an undirected graph
    G = nx.Graph()
    
    # Add nodes (not strictly necessary as edges will add nodes automatically)
    G.add_nodes_from([0, 1, 2, 3, 4])
    
    # Add edges based on the given graph
    edges = [(0, 4), (1, 0), (1, 4), (2, 4), (3, 1)]
    G.add_edges_from(edges)
    
    # Create the adjacency matrix from the graph
    A = nx.adjacency_matrix(G).toarray()

    # Degree matrix (diagonal matrix where D[i][i] is the degree of node i)
    D = np.diag(np.sum(A, axis=1))
    
    # Laplacian matrix (L = D - A)
    L = D - A
    
    # Optionally draw the graph
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=16)
    plt.show()

    return A, D, L
    
def create_random_undirected_graph(N, p=0.3, seed=None):
    """
    Create a random undirected graph with N nodes.
    
    Parameters:
    - N: Number of nodes.
    - p: Probability of edge creation between any two nodes (default is 0.3).
    - seed: Random seed for reproducibility (default is None).
    
    Returns:
    - A: Adjacency matrix of the graph.
    - D: Degree matrix of the graph.
    - L: Laplacian matrix of the graph.
    """
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Generate a random undirected graph using the Erdos-Renyi model with a seed
    G = nx.erdos_renyi_graph(N, p, seed=seed)

    # Ensure the graph is connected
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(N, p, seed=seed)

    # Create the adjacency matrix from the graph
    A = nx.adjacency_matrix(G).toarray()

    # Degree matrix (diagonal matrix where D[i][i] is the degree of node i)
    D = np.diag(np.sum(A, axis=1))
    
    # Laplacian matrix (L = D - A)
    L = D - A
    
    return A, D, L
def create_connected_undirected_graph(N, extra_edges=5):
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(range(N))
    # Create a ring to ensure initial connectivity
    for i in range(N):
        G.add_edge(i, (i + 1) % N)
    
    # Randomly add more edges to ensure connectivity
    while not nx.is_connected(G):
        u, v = np.random.randint(0, N), np.random.randint(0, N)
        if u != v:
            G.add_edge(u, v)
    
    # Add extra edges
    for _ in range(extra_edges):
        u, v = np.random.randint(0, N), np.random.randint(0, N)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
    
    A = nx.adjacency_matrix(G).toarray()
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return A, D, L

def create_strongly_connected_directed_graph(N, extra_edges=5):
    G = nx.DiGraph()
    # Add nodes
    G.add_nodes_from(range(N))
    # Create a directed ring
    for i in range(N):
        G.add_edge(i, (i + 1) % N)
    # Randomly add more edges to ensure strong connectivity
    while not nx.is_strongly_connected(G):
        u, v = np.random.randint(0, N), np.random.randint(0, N)
        if u != v:
            G.add_edge(u, v)
    # Add extra edges
    for _ in range(extra_edges):
        u, v = np.random.randint(0, N), np.random.randint(0, N)
        if u != v:
            G.add_edge(u, v)
    A = nx.adjacency_matrix(G).toarray()
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return A, D, L
       
# The undirected graph consisted of follower nodes is connected and at least one follower is connected to the leader.
def generate_leader_follower_graph(N):
    # Create an undirected connected graph for follower nodes (1 to N)
    G = nx.connected_watts_strogatz_graph(N, 3, 0.5, tries=100)
    
    # Initialize adjacency matrix
    A = np.zeros((N + 1, N + 1), dtype=int)
    
    # Fill adjacency matrix with edges from the follower graph
    for (u, v) in G.edges():
        A[u + 1][v + 1] = 1
        A[v + 1][u + 1] = 1
    
    # Connect leader node (0) to at least one follower node
    leader_connection = np.random.randint(1, N + 1)
    A[0][leader_connection] = 1
    A[leader_connection][0] = 1

    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return A, D, L


def plot_network(A):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', arrows=True)
    plt.title('Network Graph')
    #plt.savefig('fig/Simulation_graph.png')
    plt.show()

def plot_eigenvalues(L):
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = eigenvalues[np.argmax(np.abs(eigenvalues))]  # Find the eigenvalue with the maximum magnitude
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalues.real, eigenvalues.imag, 'o', markersize=10)
    plt.plot(lambda_max.real, lambda_max.imag, 'ro', markersize=12,label=f'$\lambda_{{max}}={lambda_max:.4f}$')
    plt.xlabel('Real Part',fontsize=12)
    plt.ylabel('Imaginary Part',fontsize=12)
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(fontsize=18)
    #plt.savefig('fig/Spectrum_L.png')
    plt.show()
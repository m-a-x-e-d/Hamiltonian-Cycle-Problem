import subprocess
import itertools
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Encode variables into natural numbers
def encode(i, j, vertices_count, type):
    result = (i - 1) * vertices_count + j
    if type == "o":
        result += vertices_count * vertices_count
    return result

# Decode variable mapping
def satsolve_to_xy(satsolve, maxvertex):
    coords = {}
    if satsolve >= maxvertex * maxvertex:
        satsolve -= maxvertex * maxvertex
        coords["type"] = "o"
    else:
        coords["type"] = "s"
    coords["x"] = 1 + ((satsolve - 1) // maxvertex)
    coords["y"] = ((satsolve - 1) % maxvertex) + 1
    return coords

# Constraint (1): Each vertex except the last has at least one successor
def condition1(output, edge_matrix, vertices_count):
    clauses_count = 0
    for i in range(vertices_count - 1):
        cnf = []
        for j in range(vertices_count):
            if edge_matrix[i][j] == 1 and i != j:
                cnf.append(encode(i + 1, j + 1, vertices_count, "s"))
        if cnf:
            output.append(cnf)
            clauses_count += 1
    return clauses_count

# Constraint (2): Each vertex has at most one successor
def condition2(output, edge_matrix, vertices_count):
    clauses_count = 0
    for i in range(vertices_count):
        for j, k in itertools.combinations(range(vertices_count), 2):
            if i != j and i != k and edge_matrix[i][j] == 1 and edge_matrix[i][k] == 1:
                output.append([-encode(i + 1, j + 1, vertices_count, "s"), -encode(i + 1, k + 1, vertices_count, "s")])
                clauses_count += 1
    return clauses_count

# Constraint (3): Each vertex except the first is the successor of at least one vertex
def condition3(output, edge_matrix, vertices_count):
    clauses_count = 0
    for i in range(1, vertices_count):
        cnf = []
        for j in range(vertices_count):
            if edge_matrix[j][i] == 1 and i != j:
                cnf.append(encode(j + 1, i + 1, vertices_count, "s"))
        if cnf:
            output.append(cnf)
            clauses_count += 1
    return clauses_count

# Constraint (4): No vertex is the successor of more than one node
def condition4(output, edge_matrix, vertices_count):
    clauses_count = 0
    for j in range(vertices_count):
        for i, k in itertools.combinations(range(vertices_count), 2):
            if i != j and k != j and edge_matrix[i][j] == 1 and edge_matrix[k][j] == 1:
                output.append([-encode(i + 1, j + 1, vertices_count, "s"), -encode(k + 1, j + 1, vertices_count, "s")])
                clauses_count += 1
    return clauses_count

# Constraint (5): Transitivity constraint
def condition5(output, vertices_count):
    clauses_count = 0
    for i, j, k in itertools.combinations(range(1, vertices_count + 1), 3):
        output.append([-encode(i, j, vertices_count, "o"), -encode(j, k, vertices_count, "o"), encode(i, k, vertices_count, "o")])
        output.append([encode(i, j, vertices_count, "o"), encode(j, k, vertices_count, "o"), -encode(i, k, vertices_count, "o")])
        clauses_count += 2
    return clauses_count

# Constraint (7): Irreflexivity constraint
def condition7(output, vertices_count):
    clauses_count = 0
    for i in range(1, vertices_count + 1):
        output.append([-encode(i, i, vertices_count, "o")])
        clauses_count += 1
    return clauses_count

# Constraint (8): Ordering relation must apply to all pairs of vertices
def condition8(output, vertices_count):
    clauses_count = 0
    for i in range(1, vertices_count + 1):
        for j in range(i + 1, vertices_count + 1):
            output.append([encode(i, j, vertices_count, "o"), encode(j, i, vertices_count, "o")])
            clauses_count += 1
    return clauses_count

# Constraint (9): The first vertex precedes all others
def condition9(output, vertices_count):
    clauses_count = 0
    for i in range(2, vertices_count + 1):
        output.append([encode(1, i, vertices_count, "o")])
        clauses_count += 1
    return clauses_count

# Constraint (10): The last vertex succeeds all others
def condition10(output, vertices_count):
    clauses_count = 0
    for i in range(1, vertices_count):
        output.append([encode(i, vertices_count, vertices_count, "o")])
        clauses_count += 1
    return clauses_count

# Constraint (11): Relationship between successor and ordering relations
def condition11(output, vertices_count):
    clauses_count = 0
    for i in range(1, vertices_count + 1):
        for j in range(1, vertices_count + 1):
            if i > j:
                output.append([-encode(i, j, vertices_count, "s"), -encode(j, i, vertices_count, "o")])
            else:
                output.append([-encode(i, j, vertices_count, "s"), encode(i, j, vertices_count, "o")])
            clauses_count += 1
    return clauses_count

def construct_edge_list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        edge_section_started = False
        edge_list = []
        for line in lines:
            if line.startswith("-1"):
                return edge_list
            if edge_section_started:
                edge = line.strip().split(" ")
                edge_tuple = (int(edge[0]), int(edge[1]))
                edge_list.append(edge_tuple)
            if line.startswith("EDGE_DATA_SECTION"):
                edge_section_started = True

def construct_tour_list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        tour_section_started = False
        tour =  []
        for line in lines:
            if line.startswith("-1"):
                tour.append(tour[0])
                return tour
            if tour_section_started:
                tour.append(int(line.strip()))
            if line.startswith("TOUR_SECTION"):
                tour_section_started = True

def generate_adj_matrix(path):
    edge_list = construct_edge_list(path)

    # Determine the number of vertices
    vertices = set()
    for edge in edge_list:
        vertices.update(edge)
    num_vertices = len(vertices)

    # Initialize the adjacency matrix with zeros
    adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

# Populate the adjacency matrix
    for edge in edge_list:
        i, j = edge
        adj_matrix[i-1][j-1] = 1  # Subtract 1 to convert to 0-based index
        adj_matrix[j-1][i-1] = 1

    return adj_matrix

# Create the new adjacency matrix f_G, as the reduction of Hamiltonian Cycle to Hamiltonian Path
def construct_f_G(adj):
    n = len(adj)  # Number of vertices in the original graph G

    # Initialize an adjacency matrix for f(G), size (n + 3) x (n + 3)
    new_size = n + 3
    adj_f_G = np.zeros((new_size, new_size), dtype=int)

    # Copy the original adjacency matrix into the submatrix starting from index 1
    adj_f_G[1:n+1, 1:n+1] = adj

    # Choose an arbitrary vertex v from G, say v = 0 (original index)
    v = 0
    
    # Add the new vertices: 
    # s (index 0), the first node
    # v' (index n+1), a copy of vertex v
    # t (index n+2), the last node
    v_prime = n + 1
    s = 0
    t = n + 2

    # Add edges: (s, v), (v, v'), (v', t)
    adj_f_G[s][v+1] = 1  # (s, v) where v is now at index v+1
    adj_f_G[v+1][s] = 1  # (v, s)

    adj_f_G[v+1][v_prime] = 1  # (v, v')
    adj_f_G[v_prime][v+1] = 1  # (v', v)

    adj_f_G[v_prime][t] = 1  # (v', t)
    adj_f_G[t][v_prime] = 1  # (t, v')

    # For every neighbor (v, w) in G, add the edge (v', w) in f(G)
    for w in range(n):
        if adj[v][w] == 1:
            adj_f_G[v_prime][w+1] = 1  # (v', w) where w is now at index w+1
            adj_f_G[w+1][v_prime] = 1  # (w, v')

    return adj_f_G

def generate_result(solved, vertices_count):
    lines = solved.splitlines()
    successorconstraints = []
    orderingconstraints = []
    for line in lines:
        if line.startswith('UNSAT'):
            return []
        if line.startswith('SAT'):
            continue
        else:
            tokens = line.split()
            for token in tokens:
                if not token.startswith('-'):
                    tempvec = satsolve_to_xy(int(token), vertices_count)
                    if tempvec["type"] == "s":
                        successorconstraints.append(tempvec)
                    elif tempvec["type"] == "o":
                        orderingconstraints.append(tempvec)

    printing = []
    index = 1
    for _ in range(vertices_count - 1):
        for succ in successorconstraints:
            if succ["x"] == index:
                printing.append(succ)
                index = succ["y"]
                break

    result = []
    for succ in printing:
        result.append(succ['x'])
    
    for x in range(1, vertices_count+1):
        if x not in result:
            result.append(x)

    return result

def write_dimacs(clauses, num_vars, filename):
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

def run_minisat(input_file, output_file):
    result = subprocess.run(['minisat', input_file, output_file], capture_output=True, text=True)
    if result.returncode != 10 and result.returncode != 20:
        raise RuntimeError("MiniSat did not finish correctly")

def parse_output(output_file, vertices_count):
    with open(output_file, 'r') as f:
        lines = f.readlines()
    solved = " ".join(lines)
    result = generate_result(solved, vertices_count)
    cycle = []
    for i in range(1, len(result) - 1):
        if i == len(result) - 2:
            cycle.append(result[1] - 1)
            return cycle
        cycle.append(result[i] - 1)


def test(result, path):
    print("""
==================================================
==================== TESTING =====================
==================================================
    """)
    print(f"Checking expected solution in {path}")
    tour = construct_tour_list(path)
    if (tour.reverse() == result or tour == result):
        print("Tour matches result")

def plot(graph, original_graph):
    G_new = nx.Graph()  # New graph (with added s, t, v')
    G_orig = nx.Graph()  # Original graph

    n = len(graph)  # The size of the adjacency matrix (including s, t, and v')

    # Add edges to the new graph
    for i in range(n):
        for j in range(n):
            if graph[i][j] == 1:
                G_new.add_edge(i + 1, j + 1)

    # Add edges to the original graph
    orig_n = len(original_graph)
    for i in range(orig_n):
        for j in range(orig_n):
            if original_graph[i][j] == 1:
                G_orig.add_edge(i + 1, j + 1)

    # Special node labels for the new graph
    s = 1          # s is the first node (index 1)
    v_prime = n - 1 # v' is the second to last node (index n-1)
    t = n          # t is the last node (index n)

    node_labels_new = {i: str(i - 1) for i in G_new.nodes()}
    node_labels_new[s] = '(s)'
    node_labels_new[v_prime] = "(v')"
    node_labels_new[t] = '(t)'

    # Node colors for the new graph
    node_colors_new = []
    for node in G_new.nodes():
        if node == s:
            node_colors_new.append('green')
        elif node == v_prime:
            node_colors_new.append('orange')
        elif node == t:
            node_colors_new.append('blue')
        else:
            node_colors_new.append('lightblue')

    # Plot both graphs side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original graph
    pos_orig = nx.spring_layout(G_orig)
    nx.draw(G_orig, pos_orig, ax=ax1, with_labels=True, node_color='lightblue', node_size=500, font_size=16, font_weight='bold')
    ax1.set_title("Original Graph")

    # Plot the new graph (with s, v', t)
    pos_new = nx.spring_layout(G_new)
    nx.draw(G_new, pos_new, ax=ax2, with_labels=True, labels=node_labels_new, node_color=node_colors_new, node_size=500, font_size=16, font_weight='bold')
    ax2.set_title("Graph with New Nodes (s, v', t)")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description= "A SAT Solver for the Hamiltonian Cycle Problem. It works based on the reduction from the Hamiltonian Cycle Problem to the Hamiltonian Path Problem.")
    parser.add_argument("-g", "--graphs", default="./graphs", help="Base path for the graph input folder")
    parser.add_argument("-s", "--solutions", default="./solutions", help="Base path for the solutions folder")
    parser.add_argument('graph', help="Name of the graph to generate a Hamiltonian Cycle for (e.g. graph1)")
    args = parser.parse_args()
    graphs_base_path = args.graphs
    solutions_base_path = args.solutions
    chosen_graph = args.graph

    original_graph = generate_adj_matrix(f"{graphs_base_path}{chosen_graph}.hcp")
    vertices_count = len(original_graph)
    print(f"""
==================================================
==================== HCP TO SAT ==================
==================================================
    """)
    print(f"Graph: {chosen_graph}")
    print(f"    Number of vertices: {vertices_count}")
    
    graph = construct_f_G(original_graph)
    vertices_count = len(graph)


    clauses = []

    plot(graph, original_graph)


    # Generate CNF clauses for all conditions
    clauses_count = 0
    clauses_count += condition1(clauses, graph, vertices_count)
    clauses_count += condition2(clauses, graph, vertices_count)
    clauses_count += condition3(clauses, graph, vertices_count)
    clauses_count += condition4(clauses, graph, vertices_count)
    clauses_count += condition5(clauses, vertices_count)
    clauses_count += condition7(clauses, vertices_count)
    clauses_count += condition8(clauses, vertices_count)
    clauses_count += condition9(clauses, vertices_count)
    clauses_count += condition10(clauses, vertices_count)
    clauses_count += condition11(clauses, vertices_count)

    num_vars = 2 * vertices_count * vertices_count
    write_dimacs(clauses, num_vars, "output.cnf")

    # Run Minisat
    run_minisat("output.cnf", "result.out")

    # Parse and print the result
    result = parse_output("result.out", vertices_count)
    #plot(graph, result)

    print(f"    Hamiltonian Cycle: {result}")
    if len(result) > 0:
        test(result, f"{solutions_base_path}{chosen_graph}.hcp.tou") 

            

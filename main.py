import subprocess
import itertools
import networkx as nx
import matplotlib.pyplot as plt

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

    result.append(result[0])

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
    return generate_result(solved, vertices_count)


def plot(graph, result):
    G = nx.Graph()
    n = len(graph)

    # Add edges to the graph
    for i in range(n):
        for j in range(n):
            if graph[i][j] == 1:
                G.add_edge(i + 1, j + 1)

    if len(result) > 0:

        # Extract edges in the result path
        result_edges = [(result[i], result[i + 1]) for i in range(len(result) - 1)]
        result_edges.append((result[-1], result[0]))  # to make it a cycle

        node_colors = ['red' if node == result[0] else 'lightblue' for node in G.nodes()]

    else:
        node_colors = ['lightblue']
        result_edges= []

    # Draw the graph with different styles for result edges
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=16, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=result_edges, width=5, edge_color='green')

    # Show the plot
    plt.title("Graph Visualization")
    plt.savefig("graph_visualization.png")
    plt.close()

if __name__ == "__main__":
    graph1 = [
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0]

]

    graph2 = [[0 for _ in range(250)] for _ in range(250)]
    for i in range(250):
        graph2[i][(i + 1) % 250] = 1
        graph2[(i + 1) % 250][i] = 1
        graph2[i][(i + 2) % 250] = 1
        graph2[(i + 2) % 250][i] = 1
    

    graph = graph2
    vertices_count = len(graph)
    clauses = []

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

    print(result)

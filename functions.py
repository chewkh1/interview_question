import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_checkerboard_graph(graph, rows, cols):
    fig, ax = plt.subplots(figsize=(len(cols), len(rows)))

    for r_idx, row in enumerate(rows):
        for c_idx, col in enumerate(cols):
            x = c_idx
            y = len(rows) - r_idx - 1  # Flip y-axis to match top-down

            cell = f"{row}{col}"

            # White if walkable and has neighbors, black otherwise
            color = "#ffffff" if cell in graph and graph[cell] else "#000000"

            # Draw square
            rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)

            # Draw node label
            if cell in graph:
                ax.text(
                    x + 0.5, y + 0.5,
                    cell,
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='black' if color == "#ffffff" else 'white'
                )

                # Draw edges to neighbors
                for neighbor in graph[cell]:
                    # Handle (neighbor, weight) or just neighbor
                    neighbor_node = neighbor[0] if isinstance(neighbor, tuple) else neighbor

                    nr = rows.index(neighbor_node[0])
                    nc = cols.index(neighbor_node[1])
                    nx = nc + 0.5
                    ny = len(rows) - nr - 0.5

                    ax.plot([x + 0.5, nx], [y + 0.5, ny], color='gray')

    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Grid Graph")
    plt.show()



def check_graph_4d(graph, rows, cols, walkable):
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    def expected_neighbors(r_idx, c_idx):
        neighbors = []
        for dr, dc in directions:
            nr, nc = r_idx + dr, c_idx + dc
            if 0 <= nr < len(rows) and 0 <= nc < len(cols):
                neighbor = f"{rows[nr]}{cols[nc]}"
                if neighbor in walkable:
                    neighbors.append(neighbor)
        return set(neighbors)

    for r_idx, r in enumerate(rows):
        for c_idx, c in enumerate(cols):
            cell = f"{r}{c}"
            if cell in walkable:
                # Check node exists in graph
                if cell not in graph:
                    print(f"Error: Walkable cell {cell} is missing from graph.")
                    return False

                actual_neighbors = set(graph[cell])
                expected = expected_neighbors(r_idx, c_idx)

                if actual_neighbors != expected:
                    print(f"Error: Incorrect neighbors for {cell}.")
                    print(f"  Expected: {expected}")
                    print(f"  Found:    {actual_neighbors}")
                    return False

    print("Graph passed all checks ✅")
    return True

def bfs_shortest_path(graph, start, goal):
  queue = [[start]]
  visited = set()

  while queue:
      path = queue.pop(0)
      node = path[-1]

      if node == goal:
          return path

      if node not in visited:
          visited.add(node)
          for neighbor in graph.get(node, []):
              new_path = path + [neighbor]
              queue.append(new_path)

  return None

def check_bfs_paths(graph, walkable, test_cases):
    passed = True

    for i, (start, goal) in enumerate(test_cases, 1):
        path = bfs_shortest_path(graph, start, goal)

        print(f"\nTest Case {i}: {start} → {goal}")
        if path is None:
            print("❌ No path found.")
            passed = False
            continue

        # Check path uses only valid nodes
        if not all(node in walkable for node in path):
            print("❌ Path includes non-walkable node(s).")
            print(f"  Path: {path}")
            passed = False
            continue

        # Check start and end
        if path[0] != start or path[-1] != goal:
            print("❌ Path does not start or end correctly.")
            print(f"  Path: {path}")
            passed = False
            continue

        # Check each step is connected
        for a, b in zip(path, path[1:]):
            if b not in graph.get(a, []):
                print(f"❌ Invalid step: {a} → {b} is not a neighbor.")
                passed = False
                break
        else:
            print(f"✅ Valid path: {path}")

    if passed:
        print("\n🎉 All test cases passed.")
    else:
        print("\n⚠️ Some test cases failed.")

def check_dijkstra_path(graph, path, start, end):
    import heapq

    def actual_cost(p):
        cost = 0
        for a, b in zip(p, p[1:]):
            neighbors = dict(graph.get(a, []))
            if b not in neighbors:
                return None  # Invalid edge
            cost += neighbors[b]
        return cost

    # Check path structure
    if not path or path[0] != start or path[-1] != end:
        print("❌ Path does not start or end correctly.")
        return False

    # Check all steps are valid
    for a, b in zip(path, path[1:]):
        neighbors = dict(graph.get(a, []))
        if b not in neighbors:
            print(f"❌ Invalid move: no edge from {a} to {b}")
            return False

    path_cost = actual_cost(path)
    if path_cost is None:
        print("❌ Invalid path: one or more edges are missing.")
        return False

    # Get true shortest path and cost using Dijkstra
    def dijkstra(g, s, goal):
        queue = [(0, s, [])]
        visited = set()
        while queue:
            cost, node, p = heapq.heappop(queue)
            if node == goal:
                return p + [node], cost
            if node not in visited:
                visited.add(node)
                for neighbor, weight in g.get(node, []):
                    if neighbor not in visited:
                        heapq.heappush(queue, (cost + weight, neighbor, p + [node]))
        return None, float('inf')

    true_path, true_cost = dijkstra(graph, start, end)

    if abs(path_cost - true_cost) > 1e-6:
        print(f"❌ Path is valid but not optimal. Cost: {path_cost}, Expected: {true_cost}")
        return False

    print("✅ Path is valid and optimal.")
    return True
def longest_antakshari_sequence(words):
    # Build a graph where there is an edge from word i to word j
    # if the last character of word i is the same as the first character of word j
    graph = {}
    for i, word in enumerate(words):
        if i not in graph:
            graph[i] = []
        for j, next_word in enumerate(words):
            if i != j and word[-1] == next_word[0]:
                graph[i].append(j)

    # Memoization for storing the length of the longest path starting from each word
    memo = {}

    def dfs(node):
        if node in memo:
            return memo[node]
        max_length = 1  # At least the current word itself
        if node in graph:
            for neighbor in graph[node]:
                max_length = max(max_length, 1 + dfs(neighbor))
        memo[node] = max_length
        return max_length

    # Find the longest path in the graph
    longest_path = 0
    for i in range(len(words)):
        longest_path = max(longest_path, dfs(i))

    return longest_path

def process_input():
    # Read input from standard input
    n = int(input().strip())
    results = []

    for _ in range(n):
        line = input().strip()
        words = line.split(',')
        result = longest_antakshari_sequence(words)
        results.append(result)

    for result in results:
        print(result)

# Example usage
process_input()

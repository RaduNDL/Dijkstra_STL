#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <limits>
#include <atomic>
#include <execution>
#include <algorithm>
#include <chrono>
#include <sstream>
using namespace std;
using namespace chrono;
const long long INF = numeric_limits<long long>::max();

struct Edge {
    int to;
    long long weight;
};

void generateGraphToFile(const string& filename, int nodes, int edges, int maxWeight) {
    cout << "Generating graph..." << endl;
    vector<tuple<int, int, int>> edgeList(edges);
    mt19937 rng(random_device{}());
    uniform_int_distribution<int> nodeDist(0, nodes - 1);
    uniform_int_distribution<int> weightDist(1, maxWeight);

    for_each(execution::par_unseq, edgeList.begin(), edgeList.end(), [&](auto& e) {
        thread_local mt19937 lrng(rng());
        int u = nodeDist(lrng);
        int v = nodeDist(lrng);
        while (u == v) v = nodeDist(lrng);
        int w = weightDist(lrng);
        e = { u, v, w };
        });

    ofstream out(filename, ios::out | ios::trunc);
    out << nodes << " " << edges << "\n";

    const int batch_size = 1000000;
    string buffer;
    buffer.reserve(50 * batch_size);

    int count = 0;
    for (auto& [u, v, w] : edgeList) {
        buffer += to_string(u) + " " + to_string(v) + " " + to_string(w) + "\n";
        ++count;
        if (count % batch_size == 0) {
            out << buffer;
            buffer.clear();
        }
    }
    if (!buffer.empty())
        out << buffer;
    out.close();
    cout << "Graph generated (fast text)\n";
}
vector<vector<Edge>> readGraphFromFile(const string& filename, int& nodes, int& edges) {
    ifstream in(filename);
    in >> nodes >> edges;
    string dummy;
    getline(in, dummy); 
    vector<vector<Edge>> graph(nodes);
    string line;
    int u, v, w;
    for (int i = 0; i < edges; ++i) {
        getline(in, line);
        istringstream iss(line);
        iss >> u >> v >> w;
        graph[u].push_back({ v, w });
    }
    in.close();
    return graph;
}
vector<long long> parallelDijkstra(int start, const vector<vector<Edge>>& graph) {
    int n = graph.size();
    vector<atomic<long long>> dist(n);
    vector<atomic<bool>> active(n);
    for (int i = 0; i < n; i++) {
        dist[i].store(INF);
        active[i].store(false);
    }
    dist[start].store(0);
    active[start].store(true);
    bool changed = true;
    while (changed) {
        changed = false;
        for_each(execution::par, graph.begin(), graph.end(), [&](const vector<Edge>& edges) {
            int u = &edges - &graph[0];
            if (!active[u].exchange(false)) return;
            for (const auto& edge : edges) {
                long long newDist = dist[u] + edge.weight;
                long long oldDist = dist[edge.to];
                while (newDist < oldDist && !dist[edge.to].compare_exchange_weak(oldDist, newDist)) {}
                if (newDist < oldDist) {
                    active[edge.to] = true;
                    changed = true;
                }
            }
        });
    }
    vector<long long> final_dist(n);
    for (int i = 0; i < n; ++i)
        final_dist[i] = dist[i].load();
    return final_dist;
}
void writeDistancesToFile(const string& filename, const vector<long long>& dist) {
    const size_t batch_size = 1000000;
    ofstream out(filename);
    string buffer;
    buffer.reserve(25 * batch_size);
    for (size_t i = 0; i < dist.size(); ++i) {
        long long val = (dist[i] == INF ? -1 : dist[i]);
        buffer += to_string(val);
        if (i != dist.size() - 1) buffer += ' ';
        if ((i + 1) % batch_size == 0) {
            out << buffer;
            buffer.clear();
        }
    }
    if (!buffer.empty()) out << buffer;
    out << '\n';
    out.close();
    cout << "Distances written (text)\n";
}
int main() {
    int nodes = 1000000;
    int edges = 10000000;
    int maxWeight = 20000000;
    int startNode = 0;
    string graphFile = "graph.txt";
    string distFile = "distances.txt";
    auto t1 = high_resolution_clock::now();
    generateGraphToFile(graphFile, nodes, edges, maxWeight);
    auto t2 = high_resolution_clock::now();
    int loadedNodes, loadedEdges;
    auto graph = readGraphFromFile(graphFile, loadedNodes, loadedEdges);
    auto t3 = high_resolution_clock::now();
    auto dist = parallelDijkstra(startNode, graph);
    auto t4 = high_resolution_clock::now();
    writeDistancesToFile(distFile, dist);
    auto t5 = high_resolution_clock::now();
    cout << fixed;
    cout << "Generation: " << duration<double>(t2 - t1).count() << " sec\n";
    cout << "Reading:    " << duration<double>(t3 - t2).count() << " sec\n";
    cout << "Dijkstra:   " << duration<double>(t4 - t3).count() << " sec\n";
    cout << "Writing:    " << duration<double>(t5 - t4).count() << " sec\n";
    return 0;
}
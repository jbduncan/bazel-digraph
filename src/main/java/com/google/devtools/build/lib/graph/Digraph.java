// Copyright 2014 The Bazel Authors. All rights reserved.
// Copyright 2021 Jonathan Bluett-Duncan. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.graph;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static java.util.Comparator.comparingLong;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterators;
import com.google.common.collect.Sets;
import com.google.common.graph.AbstractGraph;
import com.google.common.graph.ElementOrder;
import com.google.common.graph.EndpointPair;
import com.google.common.graph.Graphs;
import com.google.common.graph.MutableGraph;
import com.google.common.math.LongMath;
import java.util.AbstractSet;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * {@code Digraph} a generic directed graph or "digraph", suitable for modeling asymmetric binary
 * relations.
 *
 * <p>An instance <code>G = &lt;V,E&gt;</code> consists of a set of nodes or vertices <code>V</code>
 * , and a set of directed edges <code>E</code>, which is a subset of <code>V &times; V</code>. This
 * permits self-edges but does not represent multiple edges between the same pair of nodes.
 *
 * <p>Nodes may be labeled with values of any type (type parameter T). All nodes within a graph have
 * distinct labels. The null pointer is not a valid label.
 *
 * <p>The package supports various operations for modeling partial order relations, and supports
 * input/output in AT&amp;T's 'dot' format. See http://www.research.att.com/sw/tools/graphviz/.
 *
 * <p>Some invariants:
 *
 * <ul>
 *   <li>Each graph instances "owns" the nodes is creates. The behaviour of operations on nodes a
 *       graph does not own is undefined.
 *   <li>{@code Digraph} assumes immutability of node labels, much like {@link HashMap} assumes it
 *       for keys.
 *   <li>Mutating the underlying graph invalidates any sets and iterators backed by it.
 *   <li>Nodes can be added and removed concurrently. Edges can be added and removed concurrently
 *       too. While it is thread safe to add or remove edge, these operations are not atomic. Graph
 *       can be observable in inconsistent state during this operations, for instance: edge linked
 *       to only one node.
 *   <li>
 * </ul>
 *
 * <p>Each node stores successor and predecessor adjacency sets using a representation that
 * dynamically changes with size: small sets are stored as arrays, large sets using hash tables.
 * This representation provides significant space and time performance improvements upon two prior
 * versions: the earliest used only HashSets; a later version used linked lists, as described in
 * Cormen, Leiserson &amp; Rivest.
 */
public final class Digraph<T> extends AbstractGraph<T> implements MutableGraph<T>, Cloneable {

  /** Maps labels to nodes, which are in strict 1:1 correspondence. */
  private final Map<T, Node<T>> nodes = new ConcurrentHashMap<>();

  /** Construct an empty Digraph. */
  public Digraph() {}

  @Override
  public boolean putEdge(T nodeU, T nodeV) {
    return addEdge(checkNotNull(nodeU, "nodeU"), checkNotNull(nodeV, "nodeV"));
  }

  @Override
  public boolean putEdge(EndpointPair<T> endpoints) {
    checkNotNull(endpoints, "endpoints");
    return putEdge(endpoints.source(), endpoints.target());
  }

  /**
   * Adds a directed edge between the nodes labelled 'from' and 'to', creating them if necessary.
   *
   * @return true iff the edge was not already present.
   */
  private boolean addEdge(T from, T to) {
    Node<T> fromNode = createNode(from);
    Node<T> toNode = createNode(to);
    return addEdge(fromNode, toNode);
  }

  /**
   * Adds a directed edge between the specified nodes, which must exist and belong to this graph.
   *
   * @return true iff the edge was not already present.
   *     <p>Note: multi-edges are ignored. Self-edges are permitted.
   */
  private boolean addEdge(Node<T> fromNode, Node<T> toNode) {
    return fromNode.addEdge(toNode);
  }

  @Override
  public boolean removeEdge(T nodeU, T nodeV) {
    Node<T> nodeUMaybe = getNodeMaybe(checkNotNull(nodeU, "nodeU"));
    Node<T> nodeVMaybe = getNodeMaybe(checkNotNull(nodeV, "nodeV"));
    return nodeUMaybe != null && nodeVMaybe != null && removeEdge(nodeUMaybe, nodeVMaybe);
  }

  @Override
  public boolean removeEdge(EndpointPair<T> endpoints) {
    return removeEdge(endpoints.source(), endpoints.target());
  }

  /**
   * Removes the edge between the specified nodes. Idempotent: attempts to remove non-existent edges
   * have no effect.
   *
   * @return true iff graph changed.
   */
  private boolean removeEdge(Node<T> fromNode, Node<T> toNode) {
    return fromNode.removeEdge(toNode);
  }

  /** Remove all nodes and edges. */
  public void clear() {
    nodes.clear();
  }

  @Override
  public String toString() {
    return "Digraph[" + nodes().size() + " nodes]";
  }

  /**
   * Returns a duplicate graph with the same set of node labels and the same connectivity relation.
   * The labels themselves are not cloned.
   */
  @Override
  public Digraph<T> clone() {
    final Digraph<T> that = new Digraph<>();
    visitNodesBeforeEdges(
        new AbstractGraphVisitor<T>() {
          @Override
          public void visitEdge(T lhs, T rhs) {
            that.putEdge(lhs, rhs);
          }

          @Override
          public void visitNode(T node) {
            that.addNode(node);
          }
        },
        nodes(),
        null);
    return that;
  }

  /** Returns an immutable view of the nodes of this graph. */
  @Override
  public Set<T> nodes() {
    return Collections.unmodifiableSet(nodes.keySet());
  }

  /** Returns a deterministic immutable copy of the nodes of this graph. */
  public ImmutableSortedSet<T> nodes(Comparator<? super T> comparator) {
    checkNotNull(comparator, "comparator");
    return ImmutableSortedSet.copyOf(comparator, nodes());
  }

  /**
   * @return the set of root nodes: those with no predecessors.
   *     <p>NOTE: in a cyclic graph, there may be nodes that are not reachable from any "root".
   */
  public ImmutableSet<T> roots() {
    ImmutableSet.Builder<T> roots = ImmutableSet.builder();
    for (T node : nodes()) {
      if (predecessors(node).isEmpty()) {
        roots.add(node);
      }
    }
    return roots.build();
  }

  /** @return the set of leaf nodes: those with no successors. */
  public ImmutableSet<T> leaves() {
    ImmutableSet.Builder<T> leaves = ImmutableSet.builder();
    for (T node : nodes()) {
      if (successors(node).isEmpty()) {
        leaves.add(node);
      }
    }
    return leaves.build();
  }

  /**
   * Finds and returns the node with the specified label. If there is no such node, an exception is
   * thrown. The null pointer is not a valid label.
   *
   * @return the node whose label is "label".
   * @throws IllegalArgumentException if no node was found with the specified label.
   */
  private Node<T> getNode(T label) {
    Node<T> node = nodes.get(checkNotNull(label));
    checkArgument(node != null, "No such node label: %s", label);
    return node;
  }

  /**
   * Find the node with the specified label. Returns null if it doesn't exist. The null pointer is
   * not a valid label.
   *
   * @return the node whose label is "label", or null if it was not found.
   */
  private Node<T> getNodeMaybe(T label) {
    return nodes.get(checkNotNull(label));
  }

  @Override
  public boolean addNode(T node) {
    checkNotNull(node, "node");
    boolean modified = !nodes.containsKey(node);
    createNode(node);
    return modified;
  }

  /**
   * Find or create a node with the specified label. This is the <i>only</i> factory of Nodes. The
   * null pointer is not a valid label.
   */
  private Node<T> createNode(T label) {
    return nodes.computeIfAbsent(label, Node::new);
  }

  @Override
  public Set<T> adjacentNodes(T node) {
    return Sets.union(predecessors(node), successors(node));
  }

  @Override
  public Set<T> predecessors(T node) {
    checkNotNull(node, "node");
    checkArgument(nodes().contains(node), "'node' must be a node of the graph");

    return new AbstractSet<T>() {
      @Override
      public Iterator<T> iterator() {
        return Iterators.transform(getNode(node).getPredecessors().iterator(), Node::getLabel);
      }

      @Override
      public int size() {
        return getNode(node).numPredecessors();
      }
    };
  }

  @Override
  public Set<T> successors(T node) {
    checkNotNull(node, "node");
    checkArgument(nodes().contains(node), "'node' must be a node of the graph");

    return new AbstractSet<>() {
      @Override
      public Iterator<T> iterator() {
        return Iterators.transform(getNode(node).getSuccessors().iterator(), Node::getLabel);
      }

      @Override
      public int size() {
        return getNode(node).numSuccessors();
      }
    };
  }

  @Override
  public boolean isDirected() {
    return true;
  }

  @Override
  public boolean allowsSelfLoops() {
    return true;
  }

  @Override
  public ElementOrder<T> nodeOrder() {
    return ElementOrder.unordered();
  }

  // *** Graph Algorithms ***

  // These only manipulate the graph through methods defined above.

  /** Returns true iff the graph is cyclic. Time: O(n). */
  public boolean isCyclic() {
    return Graphs.hasCycle(this);
  }

  /**
   * Returns the strong component graph of "this". That is, returns a new acyclic graph in which all
   * strongly-connected components in the original graph have been "fused" into a single node.
   *
   * @return a new graph, whose node labels are sets of nodes of the original graph. (Do not get
   *     confused as to which graph each set of Nodes belongs!)
   */
  public Digraph<Set<T>> getStrongComponentGraph() {
    Collection<Set<T>> sccs = getStronglyConnectedComponents();
    Digraph<Set<T>> scGraph = createImageUnderPartition(sccs);
    scGraph.removeSelfEdges(); // scGraph should be acyclic: no self-edges
    return scGraph;
  }

  /**
   * Returns a partition of the nodes of this graph into sets, each set being one strongly-connected
   * component of the graph.
   */
  public Collection<Set<T>> getStronglyConnectedComponents() {
    List<Set<T>> sccs = new ArrayList<>();
    SccVisitor visitor = new SccVisitor();
    nodes().forEach(node -> visitor.visit(sccs::add, node));
    return sccs;
  }

  /**
   * Given a partition of the graph into sets of nodes, returns the image of this graph under the
   * function which maps each node to the partition-set in which it appears. The labels of the new
   * graph are the (immutable) sets of the partition, and the edges of the new graph are the edges
   * of the original graph, mapped via the same function.
   *
   * <p>Note: the resulting graph may contain self-edges. If these are not wanted, call <code>
   * removeSelfEdges()</code>> on the result.
   *
   * <p>Interesting special case: if the partition is the set of strongly-connected components, the
   * result of this function is the strong-component graph.
   */
  public Digraph<Set<T>> createImageUnderPartition(Collection<Set<T>> partition) {
    checkNotNull(partition, "partition");

    // Build mapping function: each node label is mapped to its equiv class:
    Map<T, Set<T>> labelToImage = new HashMap<>();
    for (Set<T> set : partition) {
      // It's important to use immutable sets of node labels when sets are keys
      // in a map; see ImmutableSet class for explanation.
      Set<T> imageSet = ImmutableSet.copyOf(set);
      for (T node : imageSet) {
        labelToImage.put(node, imageSet);
      }
    }

    if (labelToImage.size() != nodes().size()) {
      throw new IllegalArgumentException(
          "createImageUnderPartition(): argument is not a partition");
    }

    return createImageUnderMapping(labelToImage);
  }

  /**
   * Returns the image of this graph in a given function, expressed as a mapping from labels to some
   * other domain.
   */
  public <IMAGE> Digraph<IMAGE> createImageUnderMapping(Map<T, IMAGE> map) {
    Digraph<IMAGE> imageGraph = new Digraph<>();

    for (Node<T> fromNode : nodes.values()) {
      T fromLabel = fromNode.getLabel();

      IMAGE fromImage = map.get(fromLabel);
      if (fromImage == null) {
        throw new IllegalArgumentException("Incomplete function: undefined for " + fromLabel);
      }
      imageGraph.addNode(fromImage);

      for (Node<T> toNode : fromNode.getSuccessors()) {
        T toLabel = toNode.getLabel();

        IMAGE toImage = map.get(toLabel);
        if (toImage == null) {
          throw new IllegalArgumentException("Incomplete function: undefined for " + toLabel);
        }
        imageGraph.putEdge(fromImage, toImage);
      }
    }

    return imageGraph;
  }

  /** Removes any self-edges (x,x) in this graph. */
  public void removeSelfEdges() {
    nodes().forEach(node -> removeEdge(node, node));
  }

  /**
   * Finds the shortest directed path from "fromNode" to "toNode". The path is returned as an
   * ordered list of nodes, including both endpoints. Returns null if there is no path. Uses
   * breadth-first search. Running time is O(n).
   */
  public List<T> getShortestPath(T from, T to) {
    checkNotNull(from, "from");
    checkNotNull(to, "to");
    checkArgument(nodes().contains(from), "'from' is not a member of this digraph");
    checkArgument(nodes().contains(to), "'to' is not a member of this digraph");

    if (from.equals(to)) {
      return Collections.singletonList(from);
    }

    Map<T, T> pathPredecessor = new HashMap<>();

    Set<T> marked = new HashSet<>();

    Deque<T> queue = new ArrayDeque<>();
    queue.add(from);
    marked.add(from);

    while (!queue.isEmpty()) {
      T u = queue.remove();
      for (T v : successors(u)) {
        if (marked.add(v)) {
          pathPredecessor.put(v, u);
          if (v.equals(to)) {
            return getPathToTreeNode(pathPredecessor, v); // found a path
          }
          queue.addLast(v);
        }
      }
    }
    return null; // no path
  }

  /**
   * Given a tree (expressed as a map from each node to its parent), and a starting node, returns
   * the path from the root of the tree to 'node' as a list.
   */
  public static <X> List<X> getPathToTreeNode(Map<X, X> tree, X node) {
    List<X> path = new ArrayList<>();
    while (node != null) {
      path.add(node);
      node = tree.get(node); // get parent
    }
    Collections.reverse(path);
    return Collections.unmodifiableList(path);
  }

  /**
   * Returns the nodes of an acyclic graph in topological order [a.k.a "reverse post-order" of
   * depth-first search.]
   *
   * <p>A topological order is one such that, if (u, v) is a path in acyclic graph G, then u is
   * before v in the topological order. In other words "tails before heads" or "roots before
   * leaves".
   *
   * @return The nodes of the graph, in a topological order
   */
  public List<T> getTopologicalOrder() {
    List<T> order = getMutablePostorder();
    Collections.reverse(order);
    return Collections.unmodifiableList(order);
  }

  /**
   * Returns the nodes of an acyclic graph in topological order [a.k.a "reverse post-order" of
   * depth-first search.]
   *
   * <p>A topological order is one such that, if (u, v) is a path in acyclic graph G, then u is
   * before v in the topological order. In other words "tails before heads" or "roots before
   * leaves".
   *
   * <p>If an ordering is given, returns a specific topological order from the set of all
   * topological orders; if no ordering given, returns an arbitrary (nondeterministic) one, but is a
   * bit faster because no sorting needs to be done for each node.
   *
   * @param edgeOrder the ordering in which edges originating from the same node are visited.
   * @return The nodes of the graph, in a topological order
   */
  public List<T> getTopologicalOrder(Comparator<? super T> edgeOrder) {
    CollectingVisitor<T> visitor = new CollectingVisitor<>();
    DFS<T> visitation = new DFS<>(this, DFS.Order.POSTORDER, edgeOrder, false);
    visitor.beginVisit();
    for (T node : nodes(edgeOrder)) {
      visitation.visit(node, visitor);
    }
    visitor.endVisit();

    List<T> order = visitor.getVisitedNodes();
    Collections.reverse(order);
    return order;
  }

  /** Returns the nodes of an acyclic graph in post-order. */
  public List<T> getPostorder() {
    return Collections.unmodifiableList(getMutablePostorder());
  }

  private List<T> getMutablePostorder() {
    CollectingVisitor<T> collectingVisitor = new CollectingVisitor<>();
    visitPostorder(collectingVisitor);
    return collectingVisitor.getVisitedNodes();
  }

  /**
   * Returns the (immutable) set of nodes reachable from node 'n' (reflexive transitive closure).
   */
  public Set<T> getFwdReachable(T n) {
    return getFwdReachable(Collections.singleton(n));
  }

  /**
   * Returns the (immutable) set of nodes reachable from any node in {@code startNodes} (reflexive
   * transitive closure).
   */
  public Set<T> getFwdReachable(Collection<T> startNodes) {
    // This method is intentionally not static, to permit future expansion.
    DFS<T> dfs = new DFS<>(this, DFS.Order.PREORDER, false);
    for (T n : startNodes) {
      dfs.visit(n, new AbstractGraphVisitor<>());
    }
    return dfs.getMarked();
  }

  /** Returns the (immutable) set of nodes that reach node 'n' (reflexive transitive closure). */
  public Set<T> getBackReachable(T n) {
    return getBackReachable(Collections.singleton(n));
  }

  /**
   * Returns the (immutable) set of nodes that reach some node in {@code startNodes} (reflexive
   * transitive closure).
   */
  public Set<T> getBackReachable(Collection<T> startNodes) {
    // This method is intentionally not static, to permit future expansion.
    DFS<T> dfs = new DFS<T>(this, DFS.Order.PREORDER, true);
    for (T n : startNodes) {
      dfs.visit(n, new AbstractGraphVisitor<>());
    }
    return dfs.getMarked();
  }

  @Override
  public boolean removeNode(T node) {
    Node<T> nodeMaybe = getNodeMaybe(checkNotNull(node, "node"));
    if (nodeMaybe != null) {
      removeNode(nodeMaybe, /* preserveOrder= */ false);
      return true;
    }
    return false;
  }

  /**
   * Removes the specified node in the graph.
   *
   * <p>If preserveOrder flag is set then after removing node this method connects all predecessors
   * and successors.
   *
   * <p>Let's consider graph
   *
   * <pre>
   * a -> n -> c
   * b -> n -> d
   * </pre>
   *
   * <p>After n removed the following edges will be added
   *
   * <pre>
   * a -> c
   * a -> d
   * b -> c
   * b -> d
   * </pre>
   *
   * @param node the node to remove (must be in the graph).
   * @param preserveOrder see removeNode(T, boolean).
   */
  private Collection<Node<T>> removeNode(Node<T> node, boolean preserveOrder) {
    Collection<Node<T>> predecessors = node.removeAllPredecessors();
    Collection<Node<T>> successors = node.removeAllSuccessors();

    List<Node<T>> neighbours = Collections.emptyList();

    if (preserveOrder) {
      neighbours = new ArrayList<>(successors.size() + predecessors.size());
      neighbours.addAll(successors);
      neighbours.addAll(predecessors);

      for (Node<T> p : predecessors) {
        for (Node<T> s : successors) {
          p.addEdge(s);
        }
      }
    }

    Node<T> del = nodes.remove(node.getLabel());
    if (del != node) {
      throw new IllegalStateException(del + " " + node);
    }

    return Collections.unmodifiableList(neighbours);
  }

  /**
   * Extracts the subgraph G' of this graph G, containing exactly the nodes specified by the labels
   * in V', and preserving the original <i>transitive</i> graph relation among those nodes.
   *
   * @param subset a subset of the labels of this graph; the resulting graph will have only the
   *     nodes with these labels.
   */
  public Digraph<T> extractSubgraph(final Set<T> subset) {
    Digraph<T> subgraph = this.clone();
    subgraph.subgraph(subset);
    return subgraph;
  }

  /**
   * Removes all nodes from this graph except those whose label is an element of {@code keepLabels}.
   * Edges are added so as to preserve the <i>transitive</i> closure relation.
   *
   * @param keepLabels a subset of the labels of this graph; the resulting graph will have only the
   *     nodes with these labels.
   */
  private void subgraph(final Set<T> keepLabels) {
    // This algorithm does the following:
    // Let keep = nodes that have labels in keepLabels.
    // Let toRemove = nodes \ keep. reachables = successors and predecessors of keep in nodes.
    // reachables is the subset of nodes of remove that are an immediate neighbor of some node in
    // keep.
    //
    // Removes all nodes of reachables from keepLabels.
    // Until reachables is empty:
    //   Takes n from reachables
    //   for all s in succ(n)
    //     for all p in pred(n)
    //       add the edge (p, s)
    //     add s to reachables
    //   for all p in pred(n)
    //     add p to reachables
    //   Remove n and its edges
    //
    // A few adjustments are needed to do the whole computation.

    final Set<Node<T>> toRemove = new HashSet<>();
    final Set<Node<T>> keepNeighbors = new HashSet<>();

    // Look for all nodes if they are to be kept or removed
    for (Node<T> node : nodes.values()) {
      if (keepLabels.contains(node.getLabel())) {
        // Node is to be kept
        keepNeighbors.addAll(node.getPredecessors());
        keepNeighbors.addAll(node.getSuccessors());
      } else {
        // node is to be removed.
        toRemove.add(node);
      }
    }

    if (toRemove.isEmpty()) {
      // This premature return is needed to avoid 0-size priority queue creation.
      return;
    }

    // We use a priority queue to look for low-order nodes first so we don't propagate the high
    // number of paths of high-order nodes making the time consumption explode.
    // For perfect results we should reorder the set each time we add a new edge but this would
    // be too expensive, so this is a good enough approximation.
    final PriorityQueue<Node<T>> reachables =
        new PriorityQueue<>(
            toRemove.size(),
            comparingLong(
                arg -> LongMath.saturatedMultiply(arg.numPredecessors(), arg.numSuccessors())));

    // Construct the reachables queue with the list of successors and predecessors of keep in
    // toRemove.
    keepNeighbors.retainAll(toRemove);
    reachables.addAll(keepNeighbors);
    toRemove.removeAll(reachables);

    // Remove nodes, least connected first, preserving reachability.
    while (!reachables.isEmpty()) {

      Node<T> node = reachables.poll();

      Collection<Node<T>> neighbours = removeNode(node, /*preserveOrder*/ true);

      for (Node<T> neighbour : neighbours) {
        if (toRemove.remove(neighbour)) {
          reachables.add(neighbour);
        }
      }
    }

    // Final cleanup for non-reachable nodes.
    for (Node<T> node : toRemove) {
      removeNode(node, /* preserveOrder= */ false);
    }
  }

  @FunctionalInterface
  private interface NodeSetReceiver<T> {
    void accept(Set<T> nodes);
  }

  /**
   * Find strongly connected components using path-based strong component algorithm. This has the
   * advantage over the default method of returning the components in postorder.
   *
   * <p>We visit nodes depth-first, keeping track of the order that we visit them in (preorder). Our
   * goal is to find the smallest node (in this preorder of visitation) reachable from a given node.
   * We keep track of the smallest node pointed to so far at the top of a stack. If we ever find an
   * already-visited node, then if it is not already part of a component, we pop nodes from that
   * stack until we reach this already-visited node's number or an even smaller one.
   *
   * <p>Once the depth-first visitation of a node is complete, if this node's number is at the top
   * of the stack, then it is the "first" element visited in its strongly connected component. Hence
   * we pop all elements that were pushed onto the visitation stack and put them in a strongly
   * connected component with this one, then send a passed-in {@link Digraph.NodeSetReceiver} this
   * component.
   */
  private class SccVisitor {

    // The order each node was visited in.
    private final Set<T> assigned = new HashSet<>();

    private final Map<T, Integer> preorder = new HashMap<>();

    // Stack of all nodes visited whose SCC has not yet been determined. When an SCC is found,
    // that SCC is an initial segment of this stack, and is popped off. Every time a new node is
    // visited, it is put on this stack.
    private final List<T> stack = new ArrayList<>();

    // Stack of visited indices for the first-visited nodes in each of their known-so-far
    // strongly connected components. A node pushes its index on when it is visited. If any of
    // its successors have already been visited and are not in an already-found strongly connected
    // component, then, since the successor was already visited, it and this node must be part of a
    // cycle. So every node visited since the successor is actually in the same strongly connected
    // component. In this case, preorderStack is popped until the top is at most the successor's
    // index.
    //
    // After all descendants of a node have been visited, if the top element of preorderStack is
    // still the current node's index, then it was the first element visited of the current strongly
    // connected component. So all nodes on {@code stack} down to the current node are in its
    // strongly connected component. And the node's index is popped from preorderStack.
    private final List<Integer> preorderStack = new ArrayList<>();

    // Index of node being visited.
    private int counter = 0;

    private void visit(NodeSetReceiver<T> visitor, T node) {
      if (preorder.containsKey(node)) {
        // This can only happen if this was a non-recursive call, and a previous
        // visit call had already visited node.
        return;
      }
      preorder.put(node, counter);
      stack.add(node);
      preorderStack.add(counter++);
      int preorderLength = preorderStack.size();
      for (T succ : Digraph.this.successors(node)) {
        Integer succPreorder = preorder.get(succ);
        if (succPreorder == null) {
          visit(visitor, succ);
        } else {
          // Does succ not already belong to an SCC? If it doesn't, then it
          // must be in the same SCC as node. The "starting node" of this SCC
          // must have been visited before succ (or is succ itself).
          if (!assigned.contains(succ)) {
            while (preorderStack.get(preorderStack.size() - 1) > succPreorder) {
              preorderStack.remove(preorderStack.size() - 1);
            }
          }
        }
      }
      if (preorderLength == preorderStack.size()) {
        // If the length of the preorderStack is unchanged, we did not find any earlier-visited
        // nodes that were part of a cycle with this node. So this node is the first-visited
        // element in its strongly connected component, and we collect the component.
        preorderStack.remove(preorderStack.size() - 1);
        Set<T> scc = new HashSet<>();
        T compNode;
        do {
          compNode = stack.remove(stack.size() - 1);
          assigned.add(compNode);
          scc.add(compNode);
        } while (!node.equals(compNode));
        visitor.accept(scc);
      }
    }
  }

  // *** Orders, traversals and visitors ***

  /**
   * A visitation over all the nodes in the graph that invokes <code>visitor.visitNode()</code> for
   * each node in a depth-first post-order: each node is visited <i>after</i> each of its
   * successors; the order in which edges are traversed is the order in which they were added to the
   * graph. <code>visitor.visitEdge()</code> is not called.
   *
   * @param startNodes the set of nodes from which to begin the visitation.
   */
  public void visitPostorder(GraphVisitor<T> visitor, Iterable<T> startNodes) {
    visitDepthFirst(visitor, DFS.Order.POSTORDER, false, startNodes);
  }

  /** Equivalent to {@code visitPostorder(visitor, nodes())}. */
  public void visitPostorder(GraphVisitor<T> visitor) {
    visitPostorder(visitor, nodes());
  }

  /**
   * A visitation over all the nodes in the graph that invokes <code>visitor.visitNode()</code> for
   * each node in a depth-first pre-order: each node is visited <i>before</i> each of its
   * successors; the order in which edges are traversed is the order in which they were added to the
   * graph. <code>visitor.visitEdge()</code> is not called.
   *
   * @param startNodes the set of nodes from which to begin the visitation.
   */
  public void visitPreorder(GraphVisitor<T> visitor, Iterable<T> startNodes) {
    visitDepthFirst(visitor, DFS.Order.PREORDER, false, startNodes);
  }

  /** Equivalent to {@code visitPreorder(visitor, nodes())}. */
  public void visitPreorder(GraphVisitor<T> visitor) {
    visitPreorder(visitor, nodes());
  }

  /**
   * A visitation over all the nodes in the graph in depth-first order. See DFS constructor for
   * meaning of 'order' and 'transpose' parameters.
   *
   * @param startNodes the set of nodes from which to begin the visitation.
   */
  public void visitDepthFirst(
      GraphVisitor<T> visitor, DFS.Order order, boolean transpose, Iterable<T> startNodes) {
    DFS<T> visitation = new DFS<>(this, order, transpose);
    visitor.beginVisit();
    for (T node : startNodes) {
      visitation.visit(node, visitor);
    }
    visitor.endVisit();
  }

  /**
   * Given {@code unordered}, a collection of nodes and a (possibly null) {@code comparator} for
   * their labels, returns a sorted collection if {@code comparator} is non-null, otherwise returns
   * {@code unordered}.
   */
  private static <T> Collection<T> maybeOrderCollection(
      Collection<T> unordered, @Nullable Comparator<? super T> comparator) {
    return comparator == null ? unordered : ImmutableList.sortedCopyOf(comparator, unordered);
  }

  private void visitNodesBeforeEdges(
      GraphVisitor<T> visitor, Iterable<T> startNodes, @Nullable Comparator<? super T> comparator) {
    visitor.beginVisit();
    for (T fromNode : startNodes) {
      visitor.visitNode(fromNode);
      for (T toNode : maybeOrderCollection(successors(fromNode), comparator)) {
        visitor.visitEdge(fromNode, toNode);
      }
    }
    visitor.endVisit();
  }

  /**
   * A visitation over the graph that visits all nodes and edges in topological order such that each
   * node is visited before any edge coming out of that node; ties among nodes are broken using the
   * provided {@code comparator} if not null; edges are visited in order specified by the
   * comparator, <b>not</b> topological order of the target nodes.
   */
  public void visitNodesBeforeEdges(
      GraphVisitor<T> visitor, @Nullable Comparator<? super T> comparator) {
    visitNodesBeforeEdges(
        visitor,
        comparator == null ? getTopologicalOrder() : getTopologicalOrder(comparator),
        comparator);
  }
}

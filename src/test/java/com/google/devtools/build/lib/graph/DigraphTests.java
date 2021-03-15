// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static java.util.stream.Collectors.toList;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.graph.EndpointPair;
import com.google.common.graph.Graphs;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Test for {@link Digraph}. */
class DigraphTests {

  private Digraph<String> digraph;

  @BeforeEach
  void setup() {
    digraph = new Digraph<>();
    //    f
    // / | | \
    // c g e d
    //      / \
    //      a  b
    digraph.putEdge("f", "c");
    digraph.putEdge("f", "g");
    digraph.putEdge("d", "a");
    digraph.putEdge("d", "b");
    digraph.putEdge("f", "e");
    digraph.putEdge("f", "d");
  }

  @Test
  void testNonDeterministicTopologicalOrdering() {
    assertValidTopologicalOrdering(digraph);
  }

  @Test
  void testStableTopologicalOrdering() {
    assertValidStableTopologicalOrdering(digraph);
  }

  static <T> void assertValidTopologicalOrdering(Digraph<T> digraph) {
    List<T> topologicalOrdering = digraph.getTopologicalOrder();

    assertThat(topologicalOrdering).containsExactlyElementsIn(digraph.nodes());
    for (EndpointPair<T> edge : digraph.edges()) {
      assertThat(edge.isOrdered()).isTrue();
      assertThat(topologicalOrdering).containsAtLeast(edge.source(), edge.target()).inOrder();
    }
  }

  static <T extends Comparable<? super T>> void assertValidStableTopologicalOrdering(
      Digraph<T> digraph) {

    // Get them back in topological then alphabetical order; i.e., a "stable" order.
    Comparator<T> naturalOrder = Comparator.naturalOrder();
    List<T> topologicalOrdering = digraph.getTopologicalOrder(naturalOrder.reversed());

    assertThat(topologicalOrdering).containsExactlyElementsIn(digraph.nodes());

    for (T node : digraph.nodes()) {
      for (T descendant : descendantNodes(digraph, node)) {
        assertThat(topologicalOrdering).containsAtLeast(node, descendant).inOrder();
      }
    }

    for (T node : digraph.nodes()) {
      List<T> sortedSuccessors = digraph.successors(node).stream().sorted().collect(toList());
      assertThat(topologicalOrdering).containsAtLeastElementsIn(sortedSuccessors).inOrder();
    }
  }

  private static <T> Set<T> descendantNodes(Digraph<T> digraph, T ancestor) {
    return Sets.difference(Graphs.reachableNodes(digraph, ancestor), ImmutableSet.of(ancestor));
  }
}

# bazel-digraph
This is a demo repository that shows my attempt at adapting Bazel's `Digraph` class to Guava's `Graph` interface.

Why? Because I'm interested in easy-to-use graph data structures in Java, and I thought this would be a fun exercise. :smile:

# Notes
- I made `Digraph`'s `Node` class private and changed `Digraph`'s API accordingly to make the class compatible with Guava's `Graph` interface.
- I expanded the original tests that the Bazel authors wrote for `Digraph` (which can be found in file `DigraphTests.java` in this repo) to assert that `Digraph#getTopologicalOrder` returns a series of nodes that satisfies the general definition of "[topological ordering](https://en.wikipedia.org/wiki/Topological_sorting)", rather than a specific sequence of nodes which is itself a valid topological ordering but not the only one.

  This means that one could, in theory, change the algorithm that `Digraph#topologicalOrdering` uses from [reversed depth-first post order](https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search) to e.g. [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm) without having to change the test suite.

# Run the tests

Install Java 8+, put it on your PATH, and then run the following commands through your terminal:

1. `git clone https://github.com/jbduncan/bazel-digraph.git`
2. `cd bazel-digraph`
3. `./gradlew check`

# Build the library
Run `./gradlew build -x test`. The library can then be found at `build/libs/bazel-digraph-1.0-SNAPSHOT.jar`.

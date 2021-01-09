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
// All Rights Reserved.

package com.google.devtools.build.lib.graph;

/**
 * An graph visitor interface; particularly useful for allowing subclasses to specify how to output
 * a graph. The order in which node and edge callbacks are made (DFS, BFS, etc) is defined by the
 * choice of Digraph visitation method used.
 */
interface GraphVisitor<T> {

  /** Called before visitation commences. */
  void beginVisit();

  /** Called after visitation is complete. */
  void endVisit();

  void visitEdge(T lhs, T rhs);

  void visitNode(T node);
}

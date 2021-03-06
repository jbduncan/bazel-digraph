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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * Tests that Digraph after concurrent access is in consistent state.
 *
 * <p>Inconsistent state means that any of edges is half added.
 */
class DigraphConcurrentTest {

  private static final Random RANDOM = new Random();

  // need to have contention.
  private static final int THREAD_COUNT = Runtime.getRuntime().availableProcessors() * 3;

  /** Creates 100_000 nodes randomly. Adds 10 outgoing and 10 incoming edges for every node. */
  @Test
  void testValidStateOfGraphAfterConcurrentAccess() throws InterruptedException {
    Digraph<Integer> digraph = new Digraph<>();
    final int numberOfNodes = 100_000;

    Runnable workerBoth10 = new CreateNodeWorker(digraph, 10, 10, new AtomicInteger());

    runWorkers(numberOfNodes, workerBoth10);
    assertThat(digraph.nodes().size()).isEqualTo(numberOfNodes);

    assertGraphIsInConsistentState(digraph);
  }

  /**
   * Creates 10_000 nodes randomly. Adds different number outgoing and incoming edges for every
   * node.
   */
  @Test
  void testValidStateOfGraphAfterConcurrentAccessWithVariertyOfNodes() throws InterruptedException {

    Digraph<Integer> digraph = new Digraph<>();
    final int numberOfNodes = 50_000;
    final AtomicInteger idGenerator = new AtomicInteger();

    Runnable workerBoth3 = new CreateNodeWorker(digraph, 3, 3, idGenerator);
    Runnable workerBoth10 = new CreateNodeWorker(digraph, 10, 10, idGenerator);
    Runnable workerBoth100 = new CreateNodeWorker(digraph, 100, 100, idGenerator);
    Runnable workerOutgoing20 = new CreateNodeWorker(digraph, 20, 0, idGenerator);
    Runnable workerIncoming20 = new CreateNodeWorker(digraph, 0, 20, idGenerator);

    runWorkers(
        numberOfNodes / 5, // have already 5 workers
        workerBoth3,
        workerBoth10,
        workerBoth100,
        workerIncoming20,
        workerOutgoing20);

    assertThat(digraph.nodes().size()).isEqualTo(numberOfNodes);
    assertGraphIsInConsistentState(digraph);
  }

  /**
   * Creates 20_000 nodes. Then iterate over node and for every node does: Creates 100 tasks: 50
   * tasks for adding 10 outgoing edges and 50 tasks for 10 incoming edges per each tasks. Then
   * executes all tasks in parallel in high contention. Only after all tasks for this node have
   * finished, switch to next node. this behaviour introduces very high contention around one single
   * node.
   */
  @Test
  void testAddEdgeWithHighContention()
      throws InterruptedException, ExecutionException, TimeoutException {

    final int numberOfNodes = 20_000;
    final int edgePerNode = 1000;

    Digraph<Integer> digraph = new Digraph<>();
    CreateNodeWorker workerZeroEdges = new CreateNodeWorker(digraph, 0, 0, new AtomicInteger());
    runWorkers(numberOfNodes, workerZeroEdges);
    assertThat(digraph.nodes().size()).isEqualTo(numberOfNodes);

    ExecutorService executorService = Executors.newFixedThreadPool(THREAD_COUNT);

    for (Integer node : digraph.nodes()) {

      // TODO(dbabkin): think about moving this interrupt callback logic to common library or make
      // research, may be one exists already in any open source project.
      AtomicBoolean isTestInterrupted = new AtomicBoolean(false);
      Runnable interruptedCallBack =
          () -> {
            isTestInterrupted.set(true);
            executorService.shutdownNow();
          };

      CountDownLatch countDownLatch = new CountDownLatch(1);
      List<Future<?>> futures = new ArrayList<>();
      // fork 100 tasks executed in parallel in high contention for one node:
      // 100 =  50 tasks for adding 10 outgoing edges and 50 tasks for adding 10 incoming edges
      for (int i = 0; i < edgePerNode / 20; i++) {
        // add 10 outgoing edges
        Runnable add10Outgoing =
            new CreateEdgeNightContention(
                digraph, countDownLatch, node, 10, true, interruptedCallBack);
        futures.add(executorService.submit(add10Outgoing));

        // add 10 incoming edges
        Runnable add10Incoming =
            new CreateEdgeNightContention(
                digraph, countDownLatch, node, 10, false, interruptedCallBack);
        futures.add(executorService.submit(add10Incoming));
      }

      // red lights go out, race has begun. http://www.formula1-dictionary.net/start_sequence.html
      countDownLatch.countDown();

      // wait for every tasks completed before switch to the next node.
      for (Future<?> future : futures) {
        if (isTestInterrupted.get()) {
          break;
        }
        future.get(1, TimeUnit.MINUTES);
      }
      if (isTestInterrupted.get()) {
        fail("Test had been interrupted.");
      }
    }

    assertGraphIsInConsistentState(digraph);
  }

  private void runWorkers(int count, Runnable... workers) throws InterruptedException {

    ExecutorService executorService = Executors.newFixedThreadPool(THREAD_COUNT);

    for (int i = 0; i < count; i++) {
      for (Runnable worker : workers) {
        executorService.execute(worker);
      }
    }

    executorService.shutdown();
    while (!executorService.isTerminated()) {
      if (!executorService.awaitTermination(1, TimeUnit.HOURS)) {
        throw new IllegalStateException("executor service termination wait time out");
      }
    }
  }

  /** Asserts that there are no half added edge in the graph. */
  private void assertGraphIsInConsistentState(Digraph<Integer> digraph) {
    for (Integer node : digraph.nodes()) {
      assertNodeIsInConsistentState(digraph, node);
    }
  }

  private void assertNodeIsInConsistentState(Digraph<Integer> digraph, Integer node) {
    for (Integer succ : digraph.successors(node)) {
      assertThat(digraph.predecessors(succ)).contains(node);
    }

    for (Integer pred : digraph.predecessors(node)) {
      assertThat(digraph.successors(pred)).contains(node);
    }
  }

  private static final class CreateNodeWorker implements Runnable {

    private final Digraph<Integer> digraph;
    private final int outgoingEdgesPerNode;
    private final int incomingEdgesPerNode;
    private final AtomicInteger idGenerator;

    private CreateNodeWorker(
        Digraph<Integer> digraph,
        int outgoingEdgesPerNode,
        int incomingEdgesPerNode,
        AtomicInteger idGenerator) {
      this.digraph = checkNotNull(digraph);
      checkArgument(outgoingEdgesPerNode >= 0);
      this.outgoingEdgesPerNode = outgoingEdgesPerNode;
      checkArgument(incomingEdgesPerNode >= 0);
      this.incomingEdgesPerNode = incomingEdgesPerNode;
      this.idGenerator = idGenerator;
    }

    @Override
    public void run() {
      int newNodeId = idGenerator.getAndIncrement();
      digraph.addNode(newNodeId);
      addEdgesFromNew(newNodeId);
      addEdgesToNew(newNodeId);
    }

    private void addEdgesFromNew(int newNodeId) {
      int count = Math.min(newNodeId, outgoingEdgesPerNode);
      addEdges(digraph, newNodeId, count, /*outgoing*/ true);
    }

    private void addEdgesToNew(int newNodeId) {
      int count = Math.min(newNodeId, incomingEdgesPerNode);
      addEdges(digraph, newNodeId, count, /*outgoing*/ false);
    }
  }

  private static final class CreateEdgeNightContention implements Runnable {

    private final Digraph<Integer> digraph;
    private final CountDownLatch countDownLatch;
    private final int nodeId;
    private final int numberEdgeToAdd;
    private final boolean outgoing;
    private final Runnable interruptCallBack;

    private CreateEdgeNightContention(
        Digraph<Integer> digraph,
        CountDownLatch countDownLatch,
        int nodeId,
        int numberEdgeToAdd,
        boolean outgoing,
        Runnable interruptCallBack) {
      this.digraph = checkNotNull(digraph);
      this.countDownLatch = checkNotNull(countDownLatch);
      checkArgument(nodeId >= 0);
      this.nodeId = nodeId;
      checkArgument(numberEdgeToAdd >= 0);
      this.numberEdgeToAdd = numberEdgeToAdd;
      this.outgoing = outgoing;
      this.interruptCallBack = checkNotNull(interruptCallBack);
    }

    @Override
    public void run() {
      try {
        countDownLatch.await();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        interruptCallBack.run();
        // hardly ever will see that in the log. Who cares?
        fail(e.getMessage());
      }
      addEdges(digraph, nodeId, numberEdgeToAdd, outgoing);
    }
  }

  private static void addEdges(Digraph<Integer> digraph, int nodeId, int count, boolean outgoing) {
    for (int i = 0; i < count; i++) {
      int id = RANDOM.nextInt(digraph.nodes().size());
      if (outgoing) {
        digraph.putEdge(nodeId, id);
      } else {
        digraph.putEdge(id, nodeId);
      }
    }
  }
}

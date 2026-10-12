import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import dgram from "node:dgram";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import {
  openClusterMessage,
  sealClusterMessage,
} from "../dist-electron/main/cluster-crypto.js";
import {
  ClusterService,
  clusterDiscoveryInternals,
} from "../dist-electron/main/cluster-service.js";

const secret = `osai-cluster-smoke-${randomUUID()}`;
const root = path.join(os.tmpdir(), `osai-cluster-smoke-${randomUUID()}`);
const service = new ClusterService(root, () => {});
let sender;

try {
  service.setRequestHandler(async (action, payload) => {
    assert.equal(action, "ping");
    return { response: payload };
  });
  await service.initialize({
    clusterMode: "node",
    clusterName: "LAN smoke test",
    clusterSecret: secret,
  });
  const status = service.getStatus();
  assert.equal(status.ready, true, status.message);
  const address = clusterDiscoveryInternals.privateInterfaceAddresses()[0];
  assert.ok(
    address,
    "A private IPv4 interface is required for this smoke test",
  );

  const id = randomUUID();
  const body = sealClusterMessage(secret, {
    id,
    action: "ping",
    payload: "hello",
  });
  const reply = await fetch(`http://${address}:45832/osai-cluster/v1`, {
    method: "POST",
    body,
    headers: { "content-type": "application/osai-cluster+json" },
    signal: AbortSignal.timeout(10_000),
  });
  assert.equal(reply.status, 200);
  const opened = openClusterMessage(
    secret,
    Buffer.from(await reply.arrayBuffer()),
    new Set(),
  );
  assert.deepEqual(opened, { id, ok: true, value: { response: "hello" } });

  const announcement = {
    protocol: "osai-cluster-discovery-v1",
    id: randomUUID(),
    name: "smoke master",
    role: "master",
    timestamp: Date.now(),
    nonce: randomUUID(),
    platform: process.platform,
    architecture: process.arch,
    memoryBytes: 1_000,
    gpuNames: [],
    gpuMemoryBytes: [],
    dataPort: 45832,
  };
  const packet = Buffer.from(
    JSON.stringify({
      ...announcement,
      signature: clusterDiscoveryInternals.signature(secret, announcement),
    }),
  );
  sender = dgram.createSocket("udp4");
  await new Promise((resolve) => sender.bind(0, address, resolve));
  sender.setMulticastInterface(address);
  await new Promise((resolve, reject) =>
    sender.send(packet, 45831, "239.255.42.91", (error) =>
      error ? reject(error) : resolve(),
    ),
  );
  await new Promise((resolve) => setTimeout(resolve, 500));
  assert.ok(
    service.getStatus().peers.some((peer) => peer.id === announcement.id),
    "The multicast announcement was not discovered",
  );
  console.log(`Encrypted cluster HTTP and multicast passed on ${address}`);
} finally {
  sender?.close();
  service.dispose();
  await fs.rm(root, { recursive: true, force: true });
}

import assert from "node:assert/strict";
import test from "node:test";
import {
  openClusterMessage,
  sealClusterMessage,
} from "../dist-electron/main/cluster-crypto.js";
import { clusterDiscoveryInternals } from "../dist-electron/main/cluster-service.js";
import { allocateRecords } from "../dist-electron/main/cluster-training.js";

const announcement = {
  protocol: "osai-cluster-discovery-v1",
  id: "4ba9e3fb-97a5-4ff2-834a-34efad2f4b81",
  name: "training-node",
  role: "node",
  timestamp: 1_800_000_000_000,
  nonce: "490238e8-6100-42ba-a329-c7ff6c266dd3",
  platform: "darwin",
  architecture: "x64",
  memoryBytes: 16 * 1024 ** 3,
  gpuNames: ["Radeon RX 580", "Radeon RX Vega 64"],
};

test("cluster discovery accepts only local/private network addresses", () => {
  const { privateAddress } = clusterDiscoveryInternals;
  for (const address of [
    "127.0.0.1",
    "10.0.0.8",
    "172.16.1.9",
    "172.31.255.1",
    "192.168.40.7",
    "169.254.3.2",
    "::1",
    "fd12::9",
  ]) {
    assert.equal(privateAddress(address), true, address);
  }
  for (const address of [
    "8.8.8.8",
    "172.15.0.1",
    "172.32.0.1",
    "203.0.113.7",
    "2001:4860:4860::8888",
  ]) {
    assert.equal(privateAddress(address), false, address);
  }
});

test("cluster announcements require the shared key and reject tampering", () => {
  const { signature, validSignature } = clusterDiscoveryInternals;
  const signed = {
    ...announcement,
    signature: signature("correct horse battery staple", announcement),
  };
  assert.equal(validSignature("correct horse battery staple", signed), true);
  assert.equal(validSignature("a different cluster key", signed), false);
  assert.equal(
    validSignature("correct horse battery staple", {
      ...signed,
      gpuNames: ["untrusted replacement"],
    }),
    false,
  );
});

test("cluster data is encrypted, authenticated, expiring and replay-safe", () => {
  const secret = "correct horse battery staple";
  const plaintext = { action: "status", privateDatasetText: "do not expose" };
  const sealed = sealClusterMessage(secret, plaintext);
  assert.equal(sealed.includes(Buffer.from("do not expose")), false);
  assert.deepEqual(openClusterMessage(secret, sealed, new Set()), plaintext);
  assert.throws(() =>
    openClusterMessage("wrong shared secret", sealed, new Set()),
  );

  const used = new Set();
  openClusterMessage(secret, sealed, used);
  assert.throws(
    () => openClusterMessage(secret, sealed, used),
    /invalid or expired/,
  );
  assert.throws(
    () => openClusterMessage(secret, sealed, new Set(), Date.now() + 120_000),
    /invalid or expired/,
  );
});

test("cluster records are apportioned by capacity with work for every machine", () => {
  assert.deepEqual(allocateRecords(16, [2, 1, 1]), [8, 4, 4]);
  assert.deepEqual(allocateRecords(3, [100, 1, 1]), [1, 1, 1]);
  assert.throws(() => allocateRecords(2, [1, 1, 1]), /one row/);
});

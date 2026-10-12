import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";
import dgram, { type RemoteInfo, type Socket } from "node:dgram";
import fs from "node:fs/promises";
import http, {
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from "node:http";
import os from "node:os";
import path from "node:path";
import type { ClusterPeer, ClusterStatus, Preferences } from "../types.js";
import { openClusterMessage, sealClusterMessage } from "./cluster-crypto.js";
import { hardwareSnapshot } from "./hardware-snapshot.js";

const multicastAddress = "239.255.42.91";
const discoveryPort = 45831;
const dataPort = 45832;
const protocol = "osai-cluster-discovery-v1";
const announceIntervalMs = 2_000;
const peerExpiryMs = 8_000;
// LAN peers can have slightly different wall clocks, especially when the LAN
// is intentionally isolated from the internet. Data messages remain replay
// protected, and discovery entries still expire after a few missed beacons.
const clockSkewMs = 60_000;
const maximumWireBytes = 350 * 1024 * 1024;

type Announcement = {
  protocol: typeof protocol;
  id: string;
  name: string;
  role: "node" | "master";
  timestamp: number;
  nonce: string;
  platform: string;
  architecture: string;
  memoryBytes: number;
  gpuNames: string[];
  gpuMemoryBytes: number[];
  dataPort: number;
};

type SignedAnnouncement = Announcement & { signature: string };
type ClusterRequest = { id: string; action: string; payload: unknown };
type ClusterResponse =
  | { id: string; ok: true; value: unknown }
  | { id: string; ok: false; error: string };
type RequestHandler = (action: string, payload: unknown) => Promise<unknown>;

function privateAddress(value: string) {
  const address = value
    .replace(/^::ffff:/, "")
    .split("%")[0]
    .toLowerCase();
  if (
    address === "::1" ||
    address.startsWith("fc") ||
    address.startsWith("fd") ||
    address.startsWith("fe8") ||
    address.startsWith("fe9") ||
    address.startsWith("fea") ||
    address.startsWith("feb")
  )
    return true;
  const parts = address.split(".").map(Number);
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part)))
    return false;
  return (
    parts[0] === 10 ||
    parts[0] === 127 ||
    (parts[0] === 169 && parts[1] === 254) ||
    (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) ||
    (parts[0] === 192 && parts[1] === 168)
  );
}

function privateInterfaceAddresses() {
  const addresses = new Set<string>();
  for (const interfaces of Object.values(os.networkInterfaces())) {
    for (const item of interfaces || []) {
      if (
        item.family === "IPv4" &&
        !item.internal &&
        privateAddress(item.address)
      )
        addresses.add(item.address);
    }
  }
  return [...addresses];
}

function unsigned(value: SignedAnnouncement | Announcement) {
  const { signature: _signature, ...payload } = value as SignedAnnouncement;
  return JSON.stringify(payload);
}

function signature(secret: string, value: SignedAnnouncement | Announcement) {
  return createHmac("sha256", secret).update(unsigned(value)).digest("hex");
}

function validSignature(secret: string, value: SignedAnnouncement) {
  const expected = Buffer.from(signature(secret, value), "hex");
  const actual = Buffer.from(value.signature || "", "hex");
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

function safeAnnouncement(value: unknown): value is SignedAnnouncement {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<SignedAnnouncement>;
  return (
    item.protocol === protocol &&
    typeof item.id === "string" &&
    item.id.length <= 80 &&
    typeof item.name === "string" &&
    item.name.length <= 80 &&
    (item.role === "node" || item.role === "master") &&
    typeof item.timestamp === "number" &&
    typeof item.nonce === "string" &&
    item.nonce.length <= 80 &&
    typeof item.platform === "string" &&
    item.platform.length <= 32 &&
    typeof item.architecture === "string" &&
    item.architecture.length <= 32 &&
    typeof item.memoryBytes === "number" &&
    Number.isSafeInteger(item.memoryBytes) &&
    item.memoryBytes >= 0 &&
    Array.isArray(item.gpuNames) &&
    item.gpuNames.length <= 32 &&
    item.gpuNames.every(
      (name) => typeof name === "string" && name.length <= 160,
    ) &&
    Array.isArray(item.gpuMemoryBytes) &&
    item.gpuMemoryBytes.length === item.gpuNames.length &&
    item.gpuMemoryBytes.every(
      (memory) => Number.isSafeInteger(memory) && memory >= 0,
    ) &&
    item.dataPort === dataPort &&
    typeof item.signature === "string" &&
    /^[a-f\d]{64}$/i.test(item.signature)
  );
}

function readRequestBody(request: IncomingMessage) {
  return new Promise<Buffer>((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    request.on("data", (chunk: Buffer) => {
      size += chunk.byteLength;
      if (size > maximumWireBytes) {
        request.destroy(new Error("Cluster message exceeds the size limit"));
        return;
      }
      chunks.push(chunk);
    });
    request.once("end", () => resolve(Buffer.concat(chunks)));
    request.once("error", reject);
  });
}

export class ClusterService {
  private socket: Socket | null = null;
  private timer: NodeJS.Timeout | null = null;
  private preferences: Preferences | null = null;
  private peers = new Map<string, ClusterPeer & { seen: number }>();
  private gpuNames: string[] = [];
  private gpuMemoryBytes: number[] = [];
  private selfId = "";
  private generation = 0;
  private servers: Server[] = [];
  private requestHandler: RequestHandler | null = null;
  private usedNonces = new Set<string>();
  private networkError = "";

  constructor(
    private readonly storageRoot: string,
    private readonly changed: (status: ClusterStatus) => void,
  ) {}

  async initialize(preferences: Preferences) {
    await fs.mkdir(this.storageRoot, { recursive: true, mode: 0o700 });
    const identityPath = path.join(this.storageRoot, "identity");
    this.selfId = (
      await fs.readFile(identityPath, "utf8").catch(() => "")
    ).trim();
    if (!/^[a-f\d-]{20,80}$/i.test(this.selfId)) {
      this.selfId = randomUUID();
      await fs.writeFile(identityPath, `${this.selfId}\n`, { mode: 0o600 });
    }
    await this.update(preferences);
  }

  async update(preferences: Preferences) {
    this.preferences = preferences;
    this.generation += 1;
    const generation = this.generation;
    this.stopSocket();
    await this.stopServers();
    this.peers.clear();
    this.usedNonces.clear();
    this.networkError = "";
    if (preferences.clusterMode === "off") {
      this.emit();
      return;
    }
    if (preferences.clusterSecret.length < 12) {
      this.emit();
      return;
    }
    const snapshot = await hardwareSnapshot().catch(() => null);
    this.gpuNames = snapshot?.gpus.map((gpu) => gpu.name).slice(0, 32) ?? [];
    this.gpuMemoryBytes =
      snapshot?.gpus
        .map((gpu) => gpu.memoryTotalBytes ?? 0)
        .slice(0, this.gpuNames.length) ?? [];
    if (generation !== this.generation) return;
    if (preferences.clusterMode === "node") await this.startServers();
    if (generation !== this.generation) return;
    this.startSocket();
  }

  setRequestHandler(handler: RequestHandler) {
    this.requestHandler = handler;
  }

  async request(peer: ClusterPeer, action: string, payload: unknown) {
    const preferences = this.preferences;
    if (!preferences || preferences.clusterMode !== "master")
      throw new Error("This computer is not the cluster master");
    if (!this.peers.has(peer.id) || peer.role !== "node")
      throw new Error("The selected cluster node is no longer available");
    if (!privateAddress(peer.address) || peer.dataPort !== dataPort)
      throw new Error("The cluster node is not on a permitted LAN address");
    const id = randomUUID();
    const body = sealClusterMessage(preferences.clusterSecret, {
      id,
      action,
      payload,
    } satisfies ClusterRequest);
    const response = await this.post(
      peer.address,
      peer.dataPort,
      body,
      action === "benchmark" ? 65 * 60_000 : 5 * 60_000,
    );
    const opened = openClusterMessage(
      preferences.clusterSecret,
      response,
      this.usedNonces,
    ) as Partial<ClusterResponse>;
    if (opened.id !== id || typeof opened.ok !== "boolean")
      throw new Error("The cluster node returned an invalid response");
    if (!opened.ok)
      throw new Error(
        "error" in opened && typeof opened.error === "string"
          ? opened.error.slice(0, 2_000)
          : "The cluster node rejected the request",
      );
    return opened.value;
  }

  getStatus(): ClusterStatus {
    const mode = this.preferences?.clusterMode || "off";
    const ready =
      mode !== "off" &&
      (this.preferences?.clusterSecret.length || 0) >= 12 &&
      !this.networkError;
    const masters = [...this.peers.values()].filter(
      (peer) => peer.role === "master",
    ).length;
    return {
      mode,
      ready,
      message: this.networkError
        ? this.networkError
        : mode === "off"
          ? "Cluster mode is off"
          : !ready
            ? "Use a cluster key with at least 12 characters"
            : mode === "node"
              ? masters > 1
                ? "More than one master is using this cluster key"
                : masters === 1
                  ? "Connected securely to a master on this LAN"
                  : "Waiting securely for a master on this LAN"
              : [...this.peers.values()].some((peer) => peer.role === "master")
                ? "Another master is already using this cluster key"
                : `Found ${this.peers.size} authenticated cluster machine${this.peers.size === 1 ? "" : "s"}`,
      selfId: this.selfId,
      peers: [...this.peers.values()]
        .sort((left, right) => left.name.localeCompare(right.name))
        .map(({ seen: _seen, ...peer }) => peer),
    };
  }

  dispose() {
    this.stopSocket();
    void this.stopServers();
  }

  private emit() {
    this.changed(this.getStatus());
  }

  private stopSocket() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (this.socket) {
      try {
        this.socket.close();
      } catch {
        // The socket can already be closing while preferences are changed.
      }
    }
    this.socket = null;
  }

  private startSocket() {
    const socket = dgram.createSocket({ type: "udp4", reuseAddr: true });
    this.socket = socket;
    socket.on("error", (error) => {
      if (this.socket === socket) {
        this.networkError = `LAN discovery failed: ${error.message}`;
        this.stopSocket();
        this.emit();
      }
    });
    socket.on("message", (message, remote) => this.receive(message, remote));
    socket.bind(discoveryPort, () => {
      if (this.socket !== socket) return;
      try {
        socket.addMembership(multicastAddress);
        socket.setMulticastTTL(1);
        socket.setMulticastLoopback(false);
      } catch (error) {
        this.networkError = `LAN multicast is unavailable: ${
          error instanceof Error ? error.message : String(error)
        }`;
        this.stopSocket();
        this.emit();
        return;
      }
      this.announce();
      this.timer = setInterval(() => {
        this.prune();
        this.announce();
      }, announceIntervalMs);
      this.emit();
    });
  }

  private announce() {
    const preferences = this.preferences;
    if (!this.socket || !preferences || preferences.clusterMode === "off")
      return;
    const payload: Announcement = {
      protocol,
      id: this.selfId,
      name: preferences.clusterName,
      role: preferences.clusterMode,
      timestamp: Date.now(),
      nonce: randomUUID(),
      platform: process.platform,
      architecture: process.arch,
      memoryBytes: os.totalmem(),
      gpuNames: this.gpuNames,
      gpuMemoryBytes: this.gpuMemoryBytes,
      dataPort,
    };
    const signed: SignedAnnouncement = {
      ...payload,
      signature: signature(preferences.clusterSecret, payload),
    };
    this.socket.send(
      Buffer.from(JSON.stringify(signed)),
      discoveryPort,
      multicastAddress,
    );
  }

  private receive(message: Buffer, remote: RemoteInfo) {
    const preferences = this.preferences;
    if (
      !preferences ||
      preferences.clusterMode === "off" ||
      !privateAddress(remote.address) ||
      message.byteLength > 16_384
    )
      return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(message.toString("utf8"));
    } catch {
      return;
    }
    if (
      !safeAnnouncement(parsed) ||
      parsed.id === this.selfId ||
      Math.abs(Date.now() - parsed.timestamp) > clockSkewMs ||
      !validSignature(preferences.clusterSecret, parsed)
    )
      return;
    const seen = Date.now();
    this.peers.set(parsed.id, {
      id: parsed.id,
      name: parsed.name,
      role: parsed.role,
      address: remote.address.replace(/^::ffff:/, ""),
      platform: parsed.platform.slice(0, 32),
      architecture: parsed.architecture.slice(0, 32),
      memoryBytes: parsed.memoryBytes,
      gpuNames: parsed.gpuNames,
      gpuMemoryBytes: parsed.gpuMemoryBytes,
      dataPort: parsed.dataPort,
      lastSeenAt: new Date(seen).toISOString(),
      seen,
    });
    this.emit();
  }

  private prune() {
    const cutoff = Date.now() - peerExpiryMs;
    let changed = false;
    for (const [id, peer] of this.peers) {
      if (peer.seen < cutoff) {
        this.peers.delete(id);
        changed = true;
      }
    }
    if (changed) this.emit();
  }

  private async startServers() {
    const addresses = privateInterfaceAddresses();
    if (!addresses.length) {
      this.networkError = "No private IPv4 LAN interface is available";
      return;
    }
    const started = await Promise.all(
      addresses.map(
        (address) =>
          new Promise<Server | null>((resolve) => {
            const server = http.createServer(
              (request, response) =>
                void this.receiveRequest(request, response),
            );
            server.requestTimeout = 0;
            server.once("error", () => resolve(null));
            server.listen(dataPort, address, () => resolve(server));
          }),
      ),
    );
    this.servers = started.filter((server): server is Server =>
      Boolean(server),
    );
    if (!this.servers.length)
      this.networkError = `Could not open the private cluster port ${dataPort}`;
  }

  private async stopServers() {
    const servers = this.servers;
    this.servers = [];
    await Promise.all(
      servers.map(
        (server) =>
          new Promise<void>((resolve) => server.close(() => resolve())),
      ),
    );
  }

  private async receiveRequest(
    request: IncomingMessage,
    response: ServerResponse,
  ) {
    const remote = request.socket.remoteAddress || "";
    const preferences = this.preferences;
    if (
      request.method !== "POST" ||
      request.url !== "/osai-cluster/v1" ||
      !privateAddress(remote) ||
      preferences?.clusterMode !== "node" ||
      !this.requestHandler
    ) {
      response.writeHead(403).end();
      return;
    }
    try {
      const body = await readRequestBody(request);
      const opened = openClusterMessage(
        preferences.clusterSecret,
        body,
        this.usedNonces,
      ) as Partial<ClusterRequest>;
      if (
        typeof opened.id !== "string" ||
        !/^[a-f\d-]{20,80}$/i.test(opened.id) ||
        typeof opened.action !== "string" ||
        !/^[a-z-]{1,40}$/.test(opened.action)
      )
        throw new Error("Invalid cluster request");
      let result: ClusterResponse;
      try {
        result = {
          id: opened.id,
          ok: true,
          value: await this.requestHandler(opened.action, opened.payload),
        };
      } catch (error) {
        result = {
          id: opened.id,
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        };
      }
      const sealed = sealClusterMessage(preferences.clusterSecret, result);
      response.writeHead(200, {
        "content-type": "application/osai-cluster+json",
        "content-length": sealed.byteLength,
        "cache-control": "no-store",
      });
      response.end(sealed);
    } catch {
      response.writeHead(400).end();
    }
  }

  private post(address: string, port: number, body: Buffer, timeoutMs: number) {
    return new Promise<Buffer>((resolve, reject) => {
      const request = http.request(
        {
          host: address,
          port,
          path: "/osai-cluster/v1",
          method: "POST",
          headers: {
            "content-type": "application/osai-cluster+json",
            "content-length": body.byteLength,
          },
          timeout: timeoutMs,
          agent: false,
        },
        async (response) => {
          try {
            if (response.statusCode !== 200)
              throw new Error(
                `Cluster node rejected the request (${response.statusCode || "network error"})`,
              );
            resolve(await readRequestBody(response));
          } catch (error) {
            reject(error);
          }
        },
      );
      request.once("timeout", () =>
        request.destroy(new Error("Cluster node timed out")),
      );
      request.once("error", reject);
      request.end(body);
    });
  }
}

export const clusterDiscoveryInternals = {
  privateAddress,
  privateInterfaceAddresses,
  signature,
  validSignature,
};

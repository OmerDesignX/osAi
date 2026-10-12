import {
  createCipheriv,
  createDecipheriv,
  randomBytes,
  scryptSync,
} from "node:crypto";
import { gunzipSync, gzipSync } from "node:zlib";

const protocol = "osai-cluster-data-v1";
const maximumPlaintextBytes = 256 * 1024 * 1024;
const maximumWireBytes = 350 * 1024 * 1024;

type Envelope = {
  protocol: typeof protocol;
  timestamp: number;
  nonce: string;
  iv: string;
  tag: string;
  ciphertext: string;
};

function key(secret: string) {
  if (secret.length < 12)
    throw new Error("The private cluster key is too short");
  return scryptSync(secret, protocol, 32);
}

export function sealClusterMessage(secret: string, value: unknown) {
  const timestamp = Date.now();
  const nonce = randomBytes(18).toString("base64url");
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", key(secret), iv);
  cipher.setAAD(Buffer.from(`${protocol}\n${timestamp}\n${nonce}`));
  const plaintext = gzipSync(Buffer.from(JSON.stringify(value)), { level: 6 });
  if (plaintext.byteLength > maximumPlaintextBytes)
    throw new Error("The cluster message is too large");
  const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  const envelope: Envelope = {
    protocol,
    timestamp,
    nonce,
    iv: iv.toString("base64url"),
    tag: cipher.getAuthTag().toString("base64url"),
    ciphertext: ciphertext.toString("base64"),
  };
  return Buffer.from(JSON.stringify(envelope));
}

export function openClusterMessage(
  secret: string,
  source: Buffer,
  usedNonces: Set<string>,
  now = Date.now(),
) {
  if (source.byteLength > maximumWireBytes)
    throw new Error("The cluster message is too large");
  const envelope = JSON.parse(source.toString("utf8")) as Partial<Envelope>;
  const timestamp = envelope.timestamp;
  if (
    envelope.protocol !== protocol ||
    typeof timestamp !== "number" ||
    !Number.isSafeInteger(timestamp) ||
    typeof envelope.nonce !== "string" ||
    !/^[A-Za-z0-9_-]{20,80}$/.test(envelope.nonce) ||
    typeof envelope.iv !== "string" ||
    typeof envelope.tag !== "string" ||
    typeof envelope.ciphertext !== "string" ||
    Math.abs(now - timestamp) > 60_000 ||
    usedNonces.has(envelope.nonce)
  )
    throw new Error("The cluster message is invalid or expired");
  const iv = Buffer.from(envelope.iv, "base64url");
  const tag = Buffer.from(envelope.tag, "base64url");
  if (iv.byteLength !== 12 || tag.byteLength !== 16)
    throw new Error("The cluster message is malformed");
  const decipher = createDecipheriv("aes-256-gcm", key(secret), iv);
  decipher.setAAD(Buffer.from(`${protocol}\n${timestamp}\n${envelope.nonce}`));
  decipher.setAuthTag(tag);
  const compressed = Buffer.concat([
    decipher.update(Buffer.from(envelope.ciphertext, "base64")),
    decipher.final(),
  ]);
  const plaintext = gunzipSync(compressed, {
    maxOutputLength: maximumPlaintextBytes,
  });
  usedNonces.add(envelope.nonce);
  if (usedNonces.size > 10_000)
    usedNonces.delete(usedNonces.values().next().value!);
  return JSON.parse(plaintext.toString("utf8")) as unknown;
}

export const clusterCryptoInternals = {
  protocol,
  maximumPlaintextBytes,
  maximumWireBytes,
};

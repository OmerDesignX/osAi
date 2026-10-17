function comparableGpuName(value: string) {
  return value
    .toLowerCase()
    .replace(/\b(?:amd|nvidia|intel|apple|graphics|gpu)\b/g, "")
    .replace(/[^a-z0-9]/g, "");
}

export function nativeGpuLabel(device: string, labels: Record<string, string>) {
  const name = labels[device]?.trim();
  return name && comparableGpuName(name) !== comparableGpuName(device)
    ? `${device} · ${name}`
    : device;
}

export function matchNativeGpuDevices(
  gpuNames: string[],
  labels: Record<string, string>,
) {
  const available = Object.entries(labels);
  const used = new Set<string>();
  return gpuNames.map((name) => {
    const comparable = comparableGpuName(name);
    const match = available.find(
      ([device, label]) =>
        !used.has(device) && comparableGpuName(label) === comparable,
    );
    if (!match) return null;
    used.add(match[0]);
    return match[0];
  });
}

export function gpuInventoryIdentity(inventory: {
  accelerator: string;
  devices: string[];
  total_bytes: Record<string, number>;
  labels: Record<string, string>;
}) {
  return JSON.stringify({
    accelerator: inventory.accelerator,
    devices: inventory.devices.map((device) => ({
      id: device,
      totalBytes: inventory.total_bytes[device] ?? null,
      label: inventory.labels[device] ?? "",
    })),
  });
}

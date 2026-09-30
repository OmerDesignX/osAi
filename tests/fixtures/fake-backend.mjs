import fs from "node:fs";
import path from "node:path";

if (process.argv.includes("--checkpoint")) {
  const request = process.env.OSAI_CHECKPOINT_REQUEST;
  const generation = fs.readFileSync(request, "utf8").trim();
  const output = path.join(
    path.dirname(request),
    "outputs",
    "checkpoint",
    "adapter",
    "last.gguf",
  );
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, "adapter");
  process.stdout.write(
    `osai: checkpoint saved path=${output} generation=${generation}\n`,
  );
  setTimeout(() => process.exit(0), 900);
} else if (process.argv.includes("--fail")) {
  process.stderr.write("osai: unsupported dataset schema at train.jsonl:1\n");
  process.stderr.write("osai: command exited with status 2; see log\n");
  process.exit(2);
}

if (process.argv.includes("--auto-retry")) {
  process.stdout.write(
    "osai: auto settings profile=maximum budget=10GiB context=1024 batch=4 layers=2 rank=8 threads=8\n",
  );
  process.stdout.write(
    "osai: training plan examples=2 windows=4 epochs=1 batch=4 steps=1 optimizer_updates=1\n",
  );
  process.stdout.write(
    "osai: auto retry engine=mlx attempt=2 context=1024->512 batch=4->4 reason=oom\n",
  );
  process.stdout.write(
    "osai: training attempt engine=mlx attempt=2 context=512 batch=4 windows=overlap\n",
  );
  setTimeout(() => process.exit(0), 600);
} else if (process.argv.includes("--mlx-progress")) {
  process.stdout.write(
    "osai: training plan examples=15011 epochs=1 batch=1 steps=15011 optimizer_updates=15011\n",
  );
  process.stdout.write("iter   train_loss     tok/s     tokens\n");
  process.stdout.write("3219    1.845 ▼    25    101.0k\n");
  setTimeout(
    () => process.stdout.write("7506    1.204 ▼    22    230.0k\n"),
    500,
  );
  setTimeout(() => process.exit(0), 1_200);
} else {
  process.stdout.write("Iter 1/2 loss=1.0\n");
  if (process.argv.includes("--long")) {
    setInterval(() => process.stdout.write("Iter 1/2 loss=1.0\n"), 200);
  } else {
    setTimeout(() => process.stdout.write("Iter 2/2 loss=0.5\n"), 250);
    setTimeout(() => process.exit(0), 700);
  }
}

import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

function saveFixtureCheckpoint(request, generation, contents) {
  const root = path.dirname(request);
  const output = path.join(
    root,
    ".internal",
    "checkpoint",
    "adapter",
    "last.gguf",
  );
  const model = path.join(root, "outputs", "gguf");
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.mkdirSync(model, { recursive: true });
  fs.writeFileSync(output, contents);
  process.stdout.write(
    `osai: checkpoint saved path=${output} generation=${generation}\n`,
  );
  fs.writeFileSync(path.join(model, "osai_adapter.gguf"), contents);
  fs.writeFileSync(path.join(model, "merged.gguf"), `merged ${contents}`);
  process.stdout.write(
    `osai: checkpoint model ready path=${model} generation=${generation}\n`,
  );
}

// The real CLI exits after saving at a record boundary when Stop is requested.
// Make the detached-worker fixture obey the same control protocol.
const stopRequest = process.env.OSAI_STOP_REQUEST;
if (stopRequest) {
  const stopPoll = setInterval(() => {
    if (fs.existsSync(stopRequest)) {
      clearInterval(stopPoll);
      process.stdout.write("osai: stop requested; checkpoint saved\n");
      process.exit(0);
    }
  }, 50);
  stopPoll.unref();
}

if (
  process.argv.includes("--auto-stop") ||
  process.argv.includes("--auto-stop-stalled")
) {
  const stalled = process.argv.includes("--auto-stop-stalled");
  const request = process.env.OSAI_CHECKPOINT_REQUEST;
  let savedGeneration = "";
  let step = 0;
  if (stalled) {
    const previous = path.join(
      path.dirname(request),
      "outputs",
      "checkpoint",
      "adapter",
      "last.gguf",
    );
    fs.mkdirSync(path.dirname(previous), { recursive: true });
    fs.writeFileSync(previous, "previous verified adapter");
    process.stdout.write(
      `osai: checkpoint saved path=${previous} generation=auto\n`,
    );
  }
  process.stdout.write(
    "osai: training plan examples=80 epochs=1 batch=1 steps=80 optimizer_updates=80\n",
  );
  const timer = setInterval(() => {
    step += 1;
    const loss = [1, 1.02, 1.04, 1.07, 1.09, 1.11, 1.13][
      Math.min(6, Math.floor((step - 1) / 12))
    ];
    process.stdout.write(`train: data=${step}/80 loss=${loss}\n`);
    const generation = fs.existsSync(request)
      ? fs.readFileSync(request, "utf8").trim()
      : "";
    if (generation && generation !== savedGeneration && !stalled) {
      savedGeneration = generation;
      saveFixtureCheckpoint(request, generation, `adapter at ${step}`);
    }
    if (step === 80 && !stalled) {
      clearInterval(timer);
      process.exit(0);
    }
  }, 90);
} else if (
  process.argv.includes("--pause-checkpoint") ||
  process.argv.includes("--pause-checkpoint-long")
) {
  const request = process.env.OSAI_CHECKPOINT_REQUEST;
  const total = process.argv.includes("--pause-checkpoint-long") ? 160 : 80;
  if (process.argv.includes("--nested-trainer")) {
    const heartbeat = path.join(path.dirname(request), "nested-heartbeat.txt");
    const nested = spawn(
      process.execPath,
      [
        "-e",
        "const fs=require('node:fs');setInterval(()=>fs.appendFileSync(process.argv[1],'x'),80)",
        heartbeat,
      ],
      { stdio: "ignore" },
    );
    process.on("exit", () => nested.kill());
  }
  let savedGeneration = "";
  let step = 0;
  process.stdout.write(
    `osai: training plan examples=${total} epochs=1 batch=1 steps=${total} optimizer_updates=${total}\n`,
  );
  const timer = setInterval(() => {
    step += 1;
    process.stdout.write(
      `train: [###] data=${step}/${total} loss=${(1 - step / (total * 2)).toFixed(3)}±0.01 acc=50±1%\n`,
    );
    const generation = fs.existsSync(request)
      ? fs.readFileSync(request, "utf8").trim()
      : "";
    if (generation && generation !== savedGeneration) {
      savedGeneration = generation;
      saveFixtureCheckpoint(request, generation, `adapter at ${step}`);
    }
    if (step === total) {
      clearInterval(timer);
      process.exit(0);
    }
  }, 120);
} else if (process.argv.includes("--checkpoint")) {
  const request = process.env.OSAI_CHECKPOINT_REQUEST;
  const generation = fs.readFileSync(request, "utf8").trim();
  saveFixtureCheckpoint(request, generation, "adapter");
  setTimeout(() => process.exit(0), 900);
} else if (process.argv.includes("--fail")) {
  process.stderr.write("osai: unsupported dataset schema at train.jsonl:1\n");
  process.stderr.write("osai: command exited with status 2; see log\n");
  process.exit(2);
} else if (process.argv.includes("--auto-retry")) {
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

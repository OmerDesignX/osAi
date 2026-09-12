if (process.argv.includes("--fail")) {
  process.stderr.write("osai: unsupported dataset schema at train.jsonl:1\n");
  process.exit(2);
}

if (process.argv.includes("--mlx-progress")) {
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

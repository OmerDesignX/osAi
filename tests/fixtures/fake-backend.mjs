process.stdout.write("Iter 1/2 loss=1.0\n");
if (process.argv.includes("--long")) {
  setInterval(() => process.stdout.write("Iter 1/2 loss=1.0\n"), 200);
} else {
  setTimeout(() => process.stdout.write("Iter 2/2 loss=0.5\n"), 250);
  setTimeout(() => process.exit(0), 700);
}

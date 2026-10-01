let clock = 1_000_000;
Date.now = () => (clock += 7_000);

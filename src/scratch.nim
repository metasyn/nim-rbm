import alea, random/urandom, random/mersenne

var rng = wrap(initMersenneTwister(urandom(16)))
var normal = gaussian(mu = 0, sigma = 0.1)
for i in 0..100:
  echo rng.sample(normal)

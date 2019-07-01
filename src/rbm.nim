import random, random/urandom, random/mersenne, math
import arraymancer, alea

let
  n_components = 256
  learning_rate = 0.1
  batch_size = 10
  n_iter = 10
  verbose = 0
  random_state = 0

var 
  normal_distribution = gaussian(0, 0.1)

randomize(random_state)

proc initialize_gaussian_matrix[T: SomeReal](rng: Random, m, n: int): Tensor[T] = 
  var 
    rng = wrap(initMersenneTwister(urandom(16)))
    normal_distribution = gaussian(0, 0.1)

  let total = m * n

  result = newTensor[T](total)

  for i in 0..<total:
    result[i] = rng.sample(normal_distribution)

  return result.reshape(m, n)

proc means_hidden[T: SomeReal](visible_layer: Tensor[T]): Tensor[T] =
  ## Return the probabilities P(h=1|v)
  ##
  ## visible_layer dot components
  ##  + intercept
  ##
  ##  with logistic function

proc sample_hidden[T: SomeReal](visible_layer: Tensor[T]): Tensor[T] =
  ## Sample from the distribution P(h|v)
  ##
  ## means_hidden
  ## value < random_sample 


proc sample_visible[T: SomeReal](hidden_layer: Tensor[T]): Tensor[T] =
  ## Sample from the distribution P(v|h)
  ##
  ## h dot components
  ##  + intercept
  ##
  ##  with logistic function
  ##  value < random_sample

proc free_energy[T: SomeReal](visible_layer: Tensor[T]): Tensor[T] =
  ## Compute the free energy F(v) = - log sum_h exp(-E(v,h))

proc gibbs[T: SomeReal](visible_layer: Tensor[T]): Tensor[T] =
  ## Perfrom one Gibbs sampling step
  let h = sample_hidden(visible_layer)
  result  = sample_visible(h)

proc mini_batch_fit[T: SomeReal](visible_positive, hidden_samples: Tensor[T], learning_rate: float): Tensor[T] =
  let
    hidden_positive = means_hidden(visible_positive)
    visible_negative = sample_visible(hidden_samples)
    hidden_negative = means_hidden(visible_negative)
    lr = learning_rate ./ visible_positive.shape[0]

  var update = dot(visible_positive.T, hidden_positive)
  update -= dot(hidden_negative.T, visible_negative)

proc fit[T: SomeReal](X, y: Tensor[T], n_components, batch_size: int) =
  let
    (n_rows, n_cols) = X.shape
    rng = wrap(initMersenneTwister(urandom(16)))
    n_batches = int(ceil(float(n_rows) / batch_size))
  var
    components = initialize_gaussian_matrix(rng: n_components, n_cols)
    intercept_hidden = zeros(n_components)
    intercept_visisble = zeros(n_cols)
    hidden_samples = zeros((batch_size, n_components))
  


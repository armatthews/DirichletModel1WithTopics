import math
import random

# This code was taken largely from Victor Chahuneau's vpyp repository.
# See https://github.com/vchahun/vpyp/blob/master/vpyp/prior.py

class SampledPrior:
	def __init__(self):
		self.tied_distributions = []

	def tie(self, distribution):
		self.tied_distributions.append(distribution)

	def full_log_likelihood(self):
		return sum(distribution.log_likelihood() for distribution in self.tied_distributions) + self.log_likelihood()

	def resample(self, num_iters):
		old_log_likelihood = self.full_log_likelihood()
		for _ in range(num_iters):
			old_parameters = self.parameters
			self.sample_parameters()
			new_parameters = self.parameters

			new_log_likelihood = self.full_log_likelihood()
			old_log_q = self.proposal_log_likelihood(new_parameters, old_parameters)
			new_log_q = self.proposal_log_likelihood(old_parameters, new_parameters)
			log_acceptance = (new_log_likelihood - old_log_likelihood) + (old_log_q - new_log_q)

			if log_acceptance >= 0:
				old_log_likelihood = new_log_likelihood
			elif random.random() < math.exp(log_acceptance):
				old_log_likelihood = new_log_likelihood
			else:
				self.parameters = old_parameters

class GammaPrior(SampledPrior):
	"""Prior for parameters with range (0, +inf)"""
	def __init__(self, k, theta, x=None):
		assert k > 0.0
		assert theta > 0.0
		assert x == None or x > 0.0

		SampledPrior.__init__(self)
		if x == None:
			 x = random.gammavariate(self.k, self.theta)

		self.k = k
		self.theta = theta
		self.x = x

	@staticmethod
	def log_pdf(k, theta, x):	
		return math.lgamma(k) - k * math.log(theta) + (k - 1) * math.log(x) - x / theta

	def log_likelihood(self):
		return self.log_pdf(self.k, self.theta, self.x)

	def sample_parameters(self):
		# Resample x, with mean equal to the previous value of x
		# Mean of gamma distribution is k * theta
		k = 1.0
		t = self.x
		self.x = random.gammavariate(k, t)

	def proposal_log_likelihood(self, old_parameters, new_parameters):
		old_value = old_parameters[0]
		new_value = new_parameters[0]
		k = 1.0
		t = old_value
		return self.log_pdf(k, t, new_value)

	def get_parameters(self):
		return (self.x,)

	def set_parameters(self, new_parameters):
		(self.x,) = new_parameters

	parameters = property(get_parameters, set_parameters)

class BetaPrior(SampledPrior):
	def __init__(self, alpha, beta, x=None):
		assert alpha > 0.0
		assert beta > 0.0
		assert x == None or (x >= 0.0 and x <= 1.0)

		SamplePrior.__init__(self)

		if x == None:
			x = random.betavariate(self.alpha, self.beta)

		self.alpha = alpha
		self.beta = beta
		self.x = x

	@staticmethod
	def log_pdf(alpha, beta, x):
		return (alpha - 1) * math.log(x) + (beta - 1) * math.log(1 - x) + math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)

	def log_likelihood(self):
		return self.log_pdf(self.alpha, self.beta, x)

	def sample_parameters(self):
		# Resample x, with mean equal to the previous value of x
		# Mean of beta distribution is alpha / (alpha + beta)
		# Victor used n=10. TODO: Ask him why.
		n = 10.0
		a = n
		b = n * (1 - self.x) / self.x
		self.x = random.betavariate(a, b)

	def proposal_log_likelihood(self, old_parameters, new_parameters):
		old_value = old_parameters[0]
		new_value = new_parameters[0]
		n = 10.0
		a = n
		b = n * (1 - old_value) / old_value
		return self.log_pdf(a, b, new_value)

	def get_parameters(self):
		return (self.x,)

	def set_parameters(self, new_parameters):
		(self.x,) = new_parameters

	parameters = property(get_parameters, set_parameters)

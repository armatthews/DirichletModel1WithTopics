import numpy
import scipy
from collections import Counter

class DirichletMultinomial(object):
	def __init__(self, K, alpha):
		self.K = K
		self.alpha = 1.0 * alpha
		self.counts = Counter()
		self.N = 0

	def increment(self, k):
		assert k >= 0 and k < self.K
		self.counts[k] += 1
		self.N += 1

	def decrement(self, k):
		assert k >= 0 and k < self.K
		self.counts[k] -= 1
		self.N -= 1
		if self.counts[k] == 0:
			del self.counts[k]

	def probability(self, k):
		assert k >= 0 and k < self.K
		numerator = self.alpha + self.counts[k]
		denominator = self.alpha * self.K + self.N
		return numerator / denominator

class ChineseRestaurantProcess(object):
	def __init__(self):
		self.tables_by_dish = {}
		self.customers_by_dish = {}
		self.num_tables = 0
		self.num_customers = 0

	# Seat a customer at the table_index'th table
	# that is labeled with the given dish.
	# If table_index is None, then a new table is created
	# and labled with the given dish.
	# Return value is whether the customer was
	# seated at a new table.
	def seat_customer(self, dish, table_index):
		table_created = False

		if dish not in self.tables_by_dish:
			self.tables_by_dish[dish] = []
			self.customers_by_dish[dish] = 0

		if table_index == None:
			self.tables_by_dish[dish].append(0)
			self.num_tables += 1
			table_index = len(self.tables_by_dish[dish]) - 1
			table_created = True

		assert table_index >= 0
		assert table_index < len(self.tables_by_dish[dish])

		self.tables_by_dish[dish][table_index] += 1
		self.customers_by_dish[dish] += 1
		self.num_customers += 1
		return table_created

	# Eject a customer from the table_index'th table
	# that is labeled with the given dish.
	# Return value is whether the ejectee was the last
	# customer at his table.
	def eject_customer(self, dish, table_index):
		assert dish in self.tables_by_dish
		assert table_index >= 0
		assert table_index < len(self.tables_by_dish[dish])
		table_removed = False

		self.tables_by_dish[dish][table_index] -= 1
		self.customers_by_dish[dish] -= 1
		self.num_customers -= 1

		if self.tables_by_dish[dish][table_index] == 0:
			del self.tables_by_dish[dish][table_index]
			self.num_tables -= 1
			table_removed = True

		if self.customers_by_dish[dish] == 0:
			del self.customers_by_dish[dish]
			del self.tables_by_dish[dish]

		return table_removed

	def eject_random_customer(self, dish):
		i = numpy.random.randint(0, self.customers_by_dish[dish])
		for table_index, customers in enumerate(self.tables_by_dish[dish]):
			if i < customers:
				return self.eject_customer(dish, table_index)
			else:
				i -= customers
		raise Exception()
		
	def output(self):
		for dish in self.tables_by_dish.keys():
			print 'There are %d customers at %d tables serving %s with populations %s.' % \
				(self.customers_by_dish[dish], len(self.tables_by_dish[dish]), str(dish), ' '.join([str(n) for n in self.tables_by_dish[dish]]))

class DirichletProcess(ChineseRestaurantProcess):
	def __init__(self, strength, base):
		ChineseRestaurantProcess.__init__(self)
		self.strength = 1.0 * strength
		self.base = base

	def probability(self, dish):
		numerator = self.strength * self.base.probability(dish)
		numerator += self.customers_by_dish[dish] if dish in self.customers_by_dish else 0.0
		denominator = self.strength + self.num_customers
		assert numerator / denominator >= 0.0
		assert numerator / denominator <= 1.0
		return numerator / denominator

	def tables_serving_dish(self, dish):
		if dish in self.tables_by_dish:
			for table_index, num_customers in enumerate(self.tables_by_dish[dish]):
				yield table_index, num_customers
		yield None, self.strength * self.base.probability(dish)

	def increment(self, dish):
		table = draw_from_multinomial([num_customers for table_index, num_customers in self.tables_serving_dish(dish)])
		if dish not in self.tables_by_dish or table >= len(self.tables_by_dish[dish]):
			table = None
		updateBase = self.seat_customer(dish, table)
		if updateBase:
			self.base.increment(dish)

	def decrement(self, dish):
		updateBase = self.eject_random_customer(dish)
		if updateBase:
			self.base.decrement(dish)

def draw_from_multinomial(probabilities):
	prob_sum = sum(probabilities)
	probabilities = [p / prob_sum for p in probabilities]
	return numpy.random.multinomial(1, probabilities).argmax()

def dirichlet_log_prob(X, a):
	k = len(X)
	P = sum([(a - 1) * math.log(x) for x in X])
	B = k * scipy.special.gammaln(a) - scipy.special.gammaln(k * a)
	return P - B

def dp_log_prob(dp, base, alpha):
	log_prob = 0.0

	customers_by_dish = Counter()
	N = 0

	for dish, tables in dp.tables_by_dish.iteritems():
		for table_index, customer_count in enumerate(tables):
			for person_index in range(customer_count):
				log_prob += numpy.log(alpha * base.probability(dish) + customers_by_dish[dish]) - numpy.log(alpha + N)
				customers_by_dish[dish] += 1
				N += 1

	return log_prob

DirichletModel1WithTopics
=========================

Bayesian implementation of IBM Model 1 with conditioning on latent topics.

To run:
python model1withtopics.py --num_iterations 100 --num_topics 2 --alpha 0.1 --beta0 0.00002 --beta1 1.0 --nonull fake_data.txt fakeOutput > output-fake.txt

To analyze results:
python analyze.py fake_data.txt fakeOutput/ 100

The fake_data.txt file contains some example English-psuedo French sentence pairs. Note that the 'French' is simplified to exaggerate the effect of grammatical gender.
When applying our model to this data, we expect to recover two "topics" representing masculine words and feminine words.

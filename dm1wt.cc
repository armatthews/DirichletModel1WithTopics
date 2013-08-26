#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "corpus/corpus.h"
#include "cpyp/m.h"
#include "cpyp/random.h"
#include "cpyp/crp.h"
#include "cpyp/tied_parameter_resampler.h"
#include "alignment_prior.h"

using namespace std;
using namespace cpyp;

double log_likelihood(const tied_parameter_resampler<crp<unsigned>>& p, 
                      const diagonal_alignment_prior& ap,
                      const vector<vector<unsigned>>& src_corpus,
                      const vector<crp<unsigned>>& underlying_ttable,
                      const vector<vector<unsigned short>>& alignments) {
  double llh = p.log_likelihood();
  for (auto& crp : underlying_ttable)
    llh += crp.log_likelihood();
  //llh += ap.log_likelihood(alignments, src_corpus);
  return llh;
}

void show_ttable(vector<crp<unsigned>>& underlying_ttable, Dict& src_dict, Dict& tgt_dict) {
  for (unsigned src_id = 1; src_id < underlying_ttable.size(); src_id++) {
    crp<unsigned>& p = underlying_ttable[src_id];
    vector<unsigned> ind(tgt_dict.max());
    for (unsigned tgt_id = 0; tgt_id < tgt_dict.max(); tgt_id++)
      ind[tgt_id] = tgt_id;

    cerr << src_dict.Convert(src_id) << "\n";
    partial_sort(ind.begin(), ind.begin() + 10, ind.end(), [&p, &tgt_dict](unsigned alignments, unsigned b) { return p.prob(alignments, 1.0 / tgt_dict.max()) > p.prob(b, 1.0 / tgt_dict.max()); });
    for (unsigned i = 0; i < 10; i++) {
      unsigned tgt_id = ind[i];
      cerr << "\t" << tgt_dict.Convert(tgt_id) << "\t" << p.prob(tgt_id, 1.0 / tgt_dict.max()) << endl;
    }
  }
}

void output_alignments(vector<vector<unsigned>>& tgt_corpus, vector<vector<unsigned short>>& alignments) {
  for (unsigned i = 0; i < tgt_corpus.size(); i++) {
    for (unsigned j = 0; j < tgt_corpus[i].size(); j++) {
      if (alignments[i][j] != 0) {
        cout << alignments[i][j] - 1 << "-" << j << " ";
      }
    }
    cout << "\n";
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << argv[0] << " <source.txt> <target.txt> <nsamples>\n\nEstimate alignments 'Pitman-Yor Model 1' model\n";
    return 1;
  }
  MT19937 eng;
  string train_src_file = argv[1];
  string train_tgt_file = argv[2];
  const bool use_alignment_prior = true;
  const bool use_null = true;
  const unsigned num_topics = 1;
  diagonal_alignment_prior diag_alignment_prior(4.0, 0.01, use_null);
  const unsigned samples = atoi(argv[3]);
  
  Dict src_dict;
  Dict tgt_dict;
  vector<vector<unsigned>> src_corpus;
  vector<vector<unsigned>> tgt_corpus;
  set<unsigned> src_vocab;
  set<unsigned> tgt_vocab;
  ReadFromFile(train_src_file, &src_dict, &src_corpus, &src_vocab);
  ReadFromFile(train_tgt_file, &tgt_dict, &tgt_corpus, &tgt_vocab);
  unsigned document_count = src_corpus.size(); // TODO
  double uniform_target_word = 1.0 / tgt_vocab.size();
  double uniform_topic = 1.0 / num_topics;
  // Add the null word to the beginning of each source segment
  for (unsigned i = 0; i < src_corpus.size(); ++i) {
    src_corpus[i].insert(src_corpus[i].begin(), 0);
  }
  assert(src_corpus.size() == tgt_corpus.size());
  // dicts contain 1 extra word, <bad>, so the values in src_corpus and tgt_corpus
  // actually run from [1, *_vocab.size()], instead of being 0-indexed.
  cerr << "Corpus size: " << src_corpus.size() << " sentences\t (" << src_vocab.size() << "/" << tgt_vocab.size() << " word types)\n";

  vector<vector<unsigned short>> topic_assignments;
  vector<vector<unsigned short>> alignments;
  vector<crp<unsigned>> underlying_ttable(src_vocab.size() + 1, crp<unsigned>(0.0, 0.001));
  vector<vector<crp<unsigned>>> topical_ttables(num_topics, vector<crp<unsigned>>(src_vocab.size() + 1, crp<unsigned>(0.0, 0.001)));
  vector<crp<unsigned>> document_topics(document_count, crp<unsigned>(0.0, 0.001));
  vector<crp<unsigned>> sentence_topics(src_corpus.size(), crp<unsigned>(0.0, 0.001));
  tied_parameter_resampler<crp<unsigned>> underlying_ttable_params(1,1,1,1,0.1,1);
  tied_parameter_resampler<crp<unsigned>> topical_ttable_params(1,1,1,1,0.1,1);
  tied_parameter_resampler<crp<unsigned>> document_topic_params(1,1,1,1,0.1,1);
  tied_parameter_resampler<crp<unsigned>> sentence_topic_params(1,1,1,1,0.1,1); 
  topic_assignments.resize(tgt_corpus.size());
  alignments.resize(tgt_corpus.size()); 
  for (unsigned i = 0; i < tgt_corpus.size(); ++i) {
    topic_assignments[i].resize(tgt_corpus[i].size());
    alignments[i].resize(tgt_corpus[i].size());
  }
  for (unsigned i = 0; i < src_vocab.size() + 1; i++) {
    underlying_ttable_params.insert(&underlying_ttable[i]);
    for(unsigned j = 0; j < num_topics; j++ ) {
      topical_ttable_params.insert(&topical_ttables[j][i]);
    }
  }
  for(unsigned i = 0; i < document_count; i++) {
    document_topic_params.insert(&document_topics[i]);
  }
  for(unsigned i = 0; i < src_corpus.size(); ++i) {
    sentence_topic_params.insert(&sentence_topics[i]);
  }

  unsigned longest_src_sent_length = 0;
  for (unsigned i = 0; i < src_corpus.size(); i++) {
    longest_src_sent_length = (src_corpus[i].size() > longest_src_sent_length) ? src_corpus[i].size() : longest_src_sent_length;
  }

  vector<double> probs(longest_src_sent_length);
  for (unsigned sample=0; sample < samples; ++sample) {
    cerr << "beginning loop with sample = " << sample << endl;
    for (unsigned i = 0; i < tgt_corpus.size(); ++i) {
      const auto& src = src_corpus[i];
      const auto& tgt = tgt_corpus[i];
      for (unsigned j = 0; j < tgt.size(); ++j) {
        unsigned short& a_ij = alignments[i][j];
        unsigned short& z_ij = topic_assignments[i][j];
        const unsigned t = tgt[j];
        if (sample > 0) {
          if (topical_ttables[z_ij][src[a_ij]].decrement(t, eng)) {
            underlying_ttable[src[a_ij]].decrement(t, eng);
          }
        }

        // Resample alignment link
        probs.resize(src.size());
        for (unsigned k = 0; k < src.size(); ++k) {
          probs[k] = topical_ttables[z_ij][src[k]].prob(t, underlying_ttable[src[k]].prob(t, uniform_target_word));
          if (use_alignment_prior) {
            double alignment_prob;
            if (k == 0) {
              if(use_null) {
                alignment_prob = diag_alignment_prior.null_prob(j, tgt.size(), src.size() - 1);
              }
              else {
                alignment_prob = 0.0;
              }
            }
            else {
              alignment_prob = diag_alignment_prior.prob(j + 1, k, tgt.size(), src.size() - 1);
            } 
            probs[k] *= alignment_prob;
          }
        }
        multinomial_distribution<double> mult(probs);
        // random sample during the first iteration
        a_ij = sample ? mult(eng) : static_cast<unsigned>(sample_uniform01<float>(eng) * src.size());  
        alignments[i][j] = a_ij; 
        assert(a_ij >= 0);
        assert(a_ij < src.size());

        probs.resize(num_topics);
        for (unsigned k = 0; k < num_topics; ++k) {
          probs[k] = sentence_topics[i].prob(k, document_topics[i].prob(k, uniform_topic));
        }
        multinomial_distribution<double> mult2(probs);
        z_ij = sample ? mult2(eng) : static_cast<unsigned>(sample_uniform01<float>(eng) * num_topics);
        if (z_ij < 0 || z_ij >= num_topics)
          cerr << sample << " " << z_ij << " " << num_topics << "\n";
        topic_assignments[i][j] = z_ij;
        assert(z_ij >= 0);
        assert(z_ij < num_topics);

        // TODO: Instead of document_topics[i], find the document of sentence i
        if (sentence_topics[i].increment(z_ij, document_topics[i].prob(t, uniform_topic), eng)) {
          document_topics[i].increment(z_ij, uniform_topic, eng);
        }
        if (topical_ttables[z_ij][src[a_ij]].increment(t, underlying_ttable[src[a_ij]].prob(t, uniform_target_word), eng)) {
          underlying_ttable[src[a_ij]].increment(t, uniform_target_word, eng);
        }
      }
    }

    output_alignments(tgt_corpus, alignments);
    if (sample % 10 == 9) {
      cerr << " [LLH=" << log_likelihood(underlying_ttable_params, diag_alignment_prior, src_corpus, underlying_ttable, alignments) << "]" << endl;
      if (sample % 30u == 29) {
        underlying_ttable_params.resample_hyperparameters(eng);
        topical_ttable_params.resample_hyperparameters(eng);
        document_topic_params.resample_hyperparameters(eng);
        sentence_topic_params.resample_hyperparameters(eng);
        diag_alignment_prior.resample_hyperparameters(alignments, src_corpus, eng);
      }
    } else { cerr << '.' << flush; }
  }

  if(src_vocab.size() < 100) {
    show_ttable(underlying_ttable, src_dict, tgt_dict);
    cerr << "====================\n";
    show_ttable(topical_ttables[0], src_dict, tgt_dict);
    show_ttable(topical_ttables[1], src_dict, tgt_dict);
  }
  return 0;
}

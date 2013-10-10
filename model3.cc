#include <execinfo.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cstdlib>
#include "boost/archive/text_oarchive.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/program_options.hpp"

#include "bilingual_corpus.h"
#include "cpyp/m.h"
#include "cpyp/random.h"
#include "cpyp/crp.h"
#include "cpyp/tied_parameter_resampler.h"
#include "alignment_prior.h"

using namespace std;
using namespace cpyp;
namespace po=boost::program_options;

double log_likelihood(const tied_parameter_resampler<crp<unsigned>>& base_ttable_params,
                      const tied_parameter_resampler<crp<unsigned>>& underlying_ttable_params, 
                      const tied_parameter_resampler<crp<unsigned>>& sense_ttable_params,
                      const tied_parameter_resampler<crp<unsigned>>& topic_sense_table_params,
                      const tied_parameter_resampler<crp<unsigned>>& base_discourse_params,
                      const tied_parameter_resampler<crp<unsigned>>& document_discourse_params,
                      const diagonal_alignment_prior& ap,
                      const vector<vector<unsigned>>& src_corpus,
                      const crp<unsigned>& base_ttable,
                      const vector<crp<unsigned>>& underlying_ttable,
                      const vector<vector<crp<unsigned>>>& sense_ttables,
                      const vector<vector<crp<unsigned>>>& topic_sense_table,
                      const crp<unsigned>& base_discourse,
                      const vector<crp<unsigned>>& document_discourses,
                      const vector<vector<unsigned short>>& alignments,
                      const vector<vector<unsigned short>>& sense_assignments,
                      const vector<vector<unsigned short>>& topic_assignments) {
  double llh = 0.0;
  llh += base_ttable_params.log_likelihood();
  llh += underlying_ttable_params.log_likelihood();
  llh += sense_ttable_params.log_likelihood();
  llh += topic_sense_table_params.log_likelihood();
  llh += base_discourse_params.log_likelihood();
  llh += document_discourse_params.log_likelihood();

  llh += base_ttable.log_likelihood();
  for (auto& crp : underlying_ttable)
    llh += crp.log_likelihood();
  // TODO: Figure out why this makes LLH return NaN
  /*for (auto& sense_ttable : sense_ttables)
    for (auto& crp : sense_ttable) {
      float prev = llh;
      llh += crp.log_likelihood();
      if (llh != llh) {
        cerr << "Error detected!" << endl;
        cerr << crp.log_likelihood() << endl;
        cerr << prev << endl;
        cerr << crp.num_customers() << " " << crp.num_tables() << endl;
        cerr << crp.discount() << " " << crp.strength() << endl;
        exit(1);
      }
    }*/

  // TODO: Figure out why this makes LLH return NaN
  /*for (auto& word_topic_sense_table : topic_sense_table)
    for (auto& crp : word_topic_sense_table) {
      float prev = llh;
      llh += crp.log_likelihood();
      if (llh != llh) {
        cerr << "Error detected!" << endl;
        cerr << crp.log_likelihood() << endl;
        cerr << prev << endl;
        cerr << crp.num_customers() << " " << crp.num_tables() << endl;
        cerr << crp.discount() << " " << crp.strength() << endl;
        exit(1);
      }
    }*/

  llh += base_discourse.log_likelihood();
  for (auto& crp : document_discourses)
    llh += crp.log_likelihood();

  llh += ap.log_likelihood(alignments, src_corpus);
  return llh;
}

void show_ttable(vector<crp<unsigned>>& underlying_ttable, Dict& src_dict, Dict& tgt_dict) {
  vector<unsigned> ind(tgt_dict.max() + 1);
  for (unsigned tgt_id = 0; tgt_id < tgt_dict.max() + 1; tgt_id++)
    ind[tgt_id] = tgt_id;

  for (unsigned src_id = 0; src_id < underlying_ttable.size(); src_id++) {
    crp<unsigned>& p = underlying_ttable[src_id];
    if (p.num_customers() == 0)
      continue;

    cerr << src_dict.Convert(src_id) << "\t" << underlying_ttable[src_id].num_customers()
      << "\t" << underlying_ttable[src_id].num_tables() << endl;
    //partial_sort(ind.begin(), ind.begin() + 10, ind.end(), [&p, &tgt_dict](unsigned alignments, unsigned b) { return p.prob(alignments, 1.0 / tgt_dict.max()) > p.prob(b, 1.0 / tgt_dict.max()); });
//    for (unsigned i = 0; i < 10; i++) {
    sort(ind.begin(), ind.end(), [&p, &tgt_dict](unsigned alignments, unsigned b) { return p.prob(alignments, 1.0 / tgt_dict.max()) > p.prob(b, 1.0 / tgt_dict.max()); });
    for (unsigned i = 0; i < ind.size(); ++i) {
      unsigned tgt_id = ind[i];
      if (p.num_tables(tgt_id) > 0) {
        cerr << "\t" << tgt_dict.Convert(tgt_id) << "\t" << p.prob(tgt_id, 1.0 / tgt_dict.max())
          << "\t" << p.num_customers(tgt_id) << "\t" << p.num_tables(tgt_id) << endl;
      }
      else {
        cerr << "\t" << "[other]" << "\t" << p.prob(tgt_id, 1.0 / tgt_dict.max()) * (tgt_dict.max() - i) << endl;
        break;
      }
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

void output_latent_variables(vector<crp<unsigned>>& underlying_ttable, vector<vector<crp<unsigned>>> sense_ttables, crp<unsigned>& base_discourse, vector<crp<unsigned>>& document_discourses, Dict& src_dict, Dict& tgt_dict, Dict& doc_dict, unsigned num_topics) {
  double uniform_topic = 1.0 / num_topics;
  cerr << "=====BEGIN TTABLE=====\n";
  show_ttable(underlying_ttable, src_dict, tgt_dict); 
  for (auto& sense_ttable : sense_ttables) {
    cerr << "====================\n";
    show_ttable(sense_ttable, src_dict, tgt_dict);
  }
  cerr << "=====END TTABLE=====\n";

  cerr << "=====START DOCUMENT TOPIC PROBS=====\n";
  for (unsigned topic = 0; topic < num_topics; ++topic) {
    cerr << "\tTopic#" << topic << "\t" << base_discourse.prob(topic, uniform_topic) << endl;
  }
  for(unsigned doc_id = 1; doc_id < doc_dict.max(); ++doc_id) {
    cerr << doc_dict.Convert(doc_id) << endl;
    for (unsigned topic = 0; topic < num_topics; ++topic) {
      cerr << "\tTopic#" << topic << "\t" << document_discourses[doc_id].prob(topic, uniform_topic) << endl;
    }
  }
  cerr << "=====END DOCUMENT TOPIC PROBS=====\n";
}

vector<vector<unsigned short>> load_alignment_file(const string& filename, const vector<vector<unsigned>>& src_corpus, const vector<vector<unsigned>>& tgt_corpus) {
  ifstream in(filename);
  string line;
  int lc = 0;
  vector<vector<unsigned short>> alignments(src_corpus.size());
  while (getline(in, line)) {
    ++lc;
    int src_size = src_corpus[lc - 1].size();
    int tgt_size = tgt_corpus[lc - 1].size();
    vector<unsigned short> alignment(tgt_size, 0);
    vector<string> links;
    boost::split(links, line, boost::is_any_of(" "));
    for (unsigned k = 0; k < links.size(); ++k) {
      int dash_location = links[k].find('-');
      int i = atoi(links[k].substr(0, dash_location).c_str());
      int j = atoi(links[k].substr(dash_location + 1).c_str());
      assert(i >= 0);
      assert(i < src_size);
      assert(j >= 0);
      assert( j < tgt_size);
      alignment[j] = i + 1;
    }
    alignments[lc - 1] = alignment;
  }
  return alignments;
}

void handler(int sig) {
        void* array[10];
        int size = backtrace(array, 10);
        cerr << "Error: signal " << sig << ":\n";
        backtrace_symbols_fd(array, size, STDERR_FILENO);
        exit(1);
}

int main(int argc, char** argv) {
  signal(SIGSEGV, handler);
  po::options_description options("Options");
  options.add_options()
    ("training_corpus,i", po::value<string>()->required(), "Training corpus, in format of source ||| target or docid ||| source ||| target")
    ("samples,n", po::value<int>()->required(), "Number of samples")
    ("topics,k", po::value<int>(), "Number of topics")
    ("senses,z", po::value<int>(), "Number of topics")
    ("alignments,a", po::value<string>(), "Initial alignments")
    ("fix_alignments,f", "Fix the alignments, not allowing them to vary")
    ("help", "Print help messages");
  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, options), args);
    if (args.count("help")) {
       cerr << options << endl;
       return 0;
    }
    po::notify(args);
  }
  catch (po::error& e) {
    cerr << "ERROR: " << e.what() << endl << endl;
    cerr << options << endl;
    return 1;
  }

  MT19937 eng;
  const string training_corpus_file = args["training_corpus"].as<string>();
  const unsigned num_topics = (args.count("topics")) ? args["topics"].as<int>() : 2;
  const unsigned num_senses = (args.count("senses")) ? args["senses"].as<int>() : 3;
  const bool fix_alignments = args.count("fix_alignments");
  const bool have_initial_alignments = args.count("alignments");
  const string initial_alignment_file = args.count("alignments") ? args["alignments"].as<string>() : "";
  diagonal_alignment_prior diag_alignment_prior(4.0, 0.01, true);
  const unsigned samples = args["samples"].as<int>();
  
  Dict src_dict;
  Dict tgt_dict;
  Dict doc_dict;
  vector<vector<unsigned>> src_corpus;
  vector<vector<unsigned>> tgt_corpus;
  vector<unsigned> document_ids;
  set<unsigned> src_vocab;
  set<unsigned> tgt_vocab;
  ReadFromFile(training_corpus_file, &src_dict, &src_corpus, &src_vocab, &tgt_dict, &tgt_corpus, &tgt_vocab, &doc_dict, &document_ids);
  unsigned document_count = doc_dict.max();
  double uniform_target_word = 1.0 / tgt_vocab.size();
  double uniform_topic = 1.0 / num_topics;
  double uniform_sense = 1.0 / num_senses;
  double new_sense_prob = 0.7;
  vector<double> sense_probs(num_senses);
  for(unsigned z = 0; z < num_senses; ++z) {
    double Z = (1 - pow(new_sense_prob, num_senses)) / (1 - new_sense_prob);
    double p_z = pow(new_sense_prob, z) / Z;
    sense_probs[z] = p_z;
  }
  // Add the null word to the beginning of each source segment
  for (unsigned i = 0; i < src_corpus.size(); ++i) {
    src_corpus[i].insert(src_corpus[i].begin(), 0);
  }
  assert(src_corpus.size() == tgt_corpus.size());
  // dicts contain 1 extra word, <bad>, so the values in src_corpus and tgt_corpus
  // actually run from [1, *_vocab.size()], instead of being 0-indexed.
  cerr << "Corpus size: " << document_count << " documents / " << src_corpus.size() << " sentences\n";
  cerr << src_vocab.size() << " / " << tgt_vocab.size() << " word types\n";
  cerr << "Number of topics: " << num_topics << "\n";

  vector<vector<unsigned short>> topic_assignments;
  vector<vector<unsigned short>> sense_assignments;
  vector<vector<unsigned short>> alignments;
  vector<vector<set<unsigned short>>> rev_alignments;
  if (have_initial_alignments) {
    cerr << "Loading alignments from " << initial_alignment_file << endl;
    alignments = load_alignment_file(initial_alignment_file, src_corpus, tgt_corpus);
  }
  if (fix_alignments) {
    cerr << "Alignments will be fixed" << endl;
  }

  crp<unsigned> base_ttable(0.0, 0.001);
  vector<crp<unsigned>> underlying_ttable(src_vocab.size() + 1, crp<unsigned>(0.0, 0.001));
  vector<vector<crp<unsigned>>> sense_ttables(num_senses, vector<crp<unsigned>>(src_vocab.size() + 1, crp<unsigned>(0.0, 0.001)));

  vector<crp<unsigned>> sense_table(src_vocab.size() + 1, crp<unsigned>(0.0, 0.001));
  vector<vector<crp<unsigned>>> topic_sense_table(src_vocab.size() + 1, vector<crp<unsigned>>(num_topics, crp<unsigned>(0.0, 0.001)));

  crp<unsigned> base_discourse(0.0, 0.1);
  vector<crp<unsigned>> document_discourses(doc_dict.max() + 1, crp<unsigned>(0.0, 0.1));

  tied_parameter_resampler<crp<unsigned>>       base_ttable_params(1,1,1,1,0.1,uniform_target_word);
  tied_parameter_resampler<crp<unsigned>> underlying_ttable_params(1,1,1,1,0.1,uniform_target_word);
  tied_parameter_resampler<crp<unsigned>>    sense_ttable_params(1,1,1,0.1,uniform_target_word);

  tied_parameter_resampler<crp<unsigned>> sense_table_params(1,1,1,1,0.1,uniform_sense);
  tied_parameter_resampler<crp<unsigned>> topic_sense_table_params(1,1,1,1,0.1,uniform_sense);

  tied_parameter_resampler<crp<unsigned>> base_discourse_params(1,1,1,1,0.1,uniform_topic);
  tied_parameter_resampler<crp<unsigned>> document_discourse_params(1,1,1,1,0.1,uniform_topic);

  topic_assignments.resize(src_corpus.size());
  sense_assignments.resize(src_corpus.size());
  alignments.resize(tgt_corpus.size());
  rev_alignments.resize(tgt_corpus.size());
  for (unsigned i = 0; i < src_corpus.size(); ++i) {
    sense_assignments[i].resize(src_corpus[i].size());
    topic_assignments[i].resize(src_corpus[i].size());
  }
  for (unsigned i = 0; i < tgt_corpus.size(); ++i) {
    alignments[i].resize(tgt_corpus[i].size());
    rev_alignments[i].resize(src_corpus[i].size());
  }

  base_ttable_params.insert(&base_ttable);
  for (unsigned i = 0; i < src_vocab.size() + 1; i++) {
    underlying_ttable_params.insert(&underlying_ttable[i]);
    for (unsigned j = 0; j < num_senses; j++ ) {
      sense_ttable_params.insert(&sense_ttables[j][i]);
    }
  }

  for (unsigned i = 0; i < src_vocab.size() + 1; ++i) {
    sense_table_params.insert(&sense_table[i]);
    for (unsigned j = 0; j < num_topics; ++j) {
      topic_sense_table_params.insert(&topic_sense_table[i][j]);
    }
  }
  base_discourse_params.insert(&base_discourse);
  for (unsigned i = 0; i < doc_dict.max() + 1; i++) {
    document_discourse_params.insert(&document_discourses[i]);
  }

  unsigned longest_src_sent_length = 0;
  for (unsigned i = 0; i < src_corpus.size(); i++) {
    longest_src_sent_length = (src_corpus[i].size() > longest_src_sent_length) ? src_corpus[i].size() : longest_src_sent_length;
  } 

  vector<double> a_probs(longest_src_sent_length);
  vector<double> d_probs(num_topics);
  vector<double> z_probs(num_senses);
  vector<double> joint_probs(longest_src_sent_length * num_senses * num_topics);
  for (unsigned sample=0; sample < samples; ++sample) {
    cerr << "beginning loop with sample = " << sample << endl;
    for (unsigned i = 0; i < tgt_corpus.size(); ++i) {
      const auto& src = src_corpus[i];
      const auto& tgt = tgt_corpus[i];
      const unsigned doc_id = document_ids[i];

      for (unsigned m = 0; m < src.size(); ++m) { 
        const auto& s = src[m];
        unsigned short& z_im = sense_assignments[i][m];
        unsigned short& d_im = topic_assignments[i][m];
        if (sample == 0) {
          z_im = static_cast<unsigned>(sample_uniform01<float>(eng) * num_senses);
          assert(z_im >= 0);
          assert(z_im < num_senses);
 
          d_im = static_cast<unsigned>(sample_uniform01<float>(eng) * num_topics); 
          assert(d_im >= 0);
          assert(d_im < num_topics);
        }
        else {
          // Decrement the CRPs of the old value
          if (document_discourses[doc_id].decrement(d_im, eng)) {
            base_discourse.decrement(d_im, eng);
          }

          // When we decrement the z, we must also update the sense ttables
          if (topic_sense_table[s][d_im].decrement(z_im, eng)) {
            sense_table[s].decrement(z_im, eng);
          }

          for (unsigned short n : rev_alignments[i][m]) {
            const auto& t = tgt[n];
            if (sense_ttables[z_im][s].decrement(t, eng)) {
              if (underlying_ttable[s].decrement(t, eng)) {
                base_ttable.decrement(t, eng);
              }
            } 
          }

          // Find the probability of each topic
          for (unsigned k = 0; k < num_topics; ++k) {
            d_probs[k] = document_discourses[doc_id].prob(k, base_discourse.prob(k, uniform_topic));
          }

          multinomial_distribution<double> d_mult(d_probs);
          d_im = d_mult(eng);

          for (unsigned k = 0; k < num_senses; ++k) {
            z_probs[k] = topic_sense_table[s][d_im].prob(k, uniform_sense);
          }

          multinomial_distribution<double> z_mult(z_probs);
          z_im = z_mult(eng);
        }

        assert(d_im >= 0);
        assert(d_im < num_topics);

        assert(z_im >= 0);
        assert(z_im < num_senses);

        // Increment the CRPs with the new value
        if (document_discourses[doc_id].increment(d_im, base_discourse.prob(d_im, uniform_topic), eng)) {
          base_discourse.increment(d_im, uniform_topic, eng);
        }

        // Make sure to increment the affect entries in the sense_ttables!
        if (topic_sense_table[s][d_im].increment(z_im, uniform_sense, eng)) {
          sense_table[s].increment(z_im, sense_probs[z_im], eng);
        }
        for (unsigned short n : rev_alignments[i][m]) {
          const auto& t = tgt[n];
          if (sense_ttables[z_im][s].increment(t, underlying_ttable[s].prob(t, base_ttable.prob(t, uniform_target_word)), eng)) {
            if (underlying_ttable[s].increment(t, base_ttable.prob(t, uniform_target_word), eng)) {
              base_ttable.increment(t, uniform_target_word, eng);
            }
          }
        }
      }

      for (unsigned n = 0; n < tgt.size(); ++n) {
        unsigned short& a = alignments[i][n];
        unsigned short z = sense_assignments[i][a];
        const unsigned t = tgt[n];

        if (sample == 0) {
          // random sample during the first iteration
          if (!fix_alignments) {
            a = static_cast<unsigned>(sample_uniform01<float>(eng) * src.size());
          }
          assert(a >= 0);
          assert(a < src.size());
        }
        else { 
          rev_alignments[i][a].erase(n);
          if (sense_ttables[z][src[a]].decrement(t, eng)) {
            if (underlying_ttable[src[a]].decrement(t, eng)) {
              base_ttable.decrement(t, eng);
            }
          }

          // Find the probability of each alignment link
          if (!fix_alignments) {
            a_probs.resize(src.size());
            for (unsigned k = 0; k < src.size(); ++k) {
              a_probs[k] = sense_ttables[z][src[k]].prob(t, underlying_ttable[src[k]].prob(t, base_ttable.prob(t, uniform_target_word)));
              if (k == 0)
                  a_probs[k] *= diag_alignment_prior.null_prob(n, tgt.size(), src.size() - 1);
           }

            multinomial_distribution<double> mult(a_probs);
            a = mult(eng);
          }
        }

        // Verify that the draw produced valid results 
        assert(a >= 0);
        assert(a < src.size());
        rev_alignments[i][a].insert(n);
        z = sense_assignments[i][a];

        // Increment the CRPs with the new value
        if (sense_ttables[z][src[a]].increment(t, underlying_ttable[src[a]].prob(t, base_ttable.prob(t, uniform_target_word)), eng)) {
          if (underlying_ttable[src[a]].increment(t, base_ttable.prob(t, uniform_target_word), eng)) {
            base_ttable.increment(t, uniform_target_word, eng);
          }
        }
      }
    }
    output_alignments(tgt_corpus, alignments);

    if (sample % 10 == 9) {
      cerr << " [LLH=" << log_likelihood(base_ttable_params,
                                         underlying_ttable_params,
                                         sense_ttable_params,
                                         topic_sense_table_params,
                                         base_discourse_params,
                                         document_discourse_params,
                                         diag_alignment_prior,
                                         src_corpus,
                                         base_ttable,
                                         underlying_ttable,
                                         sense_ttables,
                                         topic_sense_table,
                                         base_discourse,
                                         document_discourses,
                                         alignments,
                                         sense_assignments,
                                         topic_assignments) << "]" << endl;
      if (sample % 30u == 29) {
        underlying_ttable_params.resample_hyperparameters(eng);
        sense_ttable_params.resample_hyperparameters(eng);
        sense_table_params.resample_hyperparameters(eng);
        topic_sense_table_params.resample_hyperparameters(eng);
        document_discourse_params.resample_hyperparameters(eng);
        diag_alignment_prior.resample_hyperparameters(alignments, src_corpus, eng);
      }
    }
    else {
      cerr << '.' << flush;
    }
  
    if (sample % 100u == 99) {
      output_latent_variables(underlying_ttable, sense_ttables, base_discourse, document_discourses, src_dict, tgt_dict, doc_dict, num_topics);
    }
  }

  if (true) {
    output_latent_variables(underlying_ttable, sense_ttables, base_discourse, document_discourses, src_dict, tgt_dict, doc_dict, num_topics);
  }

  return 0;
}

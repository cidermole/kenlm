#include "lm/model.hh"
#include "util/fake_ofstream.hh"
#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/usage.hh"

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <sys/stat.h>
#include <thread>

namespace {

template <class Model, class Width>
class Corpus {
public:
  Corpus(int fd_in, Width kEOS): m_kEOS(kEOS) {
    // Read entire corpus and index by sentence
    Width buf[4096];
    Width last = m_kEOS;
    size_t got;

    size_t fwords = fileSize(fd_in) / sizeof(Width);
    // nice-to: reserve some space in advance
    m_words.reserve(fwords);
    // guess about 10 words per sentence
    m_index.reserve(fwords / 10);

    // first sentence index
    m_index.push_back(0);

    while((got = util::ReadOrEOF(fd_in, buf, sizeof(buf)))) {
      UTIL_THROW_IF2(got % sizeof(Width), "File read size " << got << " not a multiple of vocab id size " << sizeof(Width));

      Width *const end = buf + got / sizeof(Width);

      for(Width *i = buf; i != end; i++) {
        m_words.push_back(*i);
        if(*i == m_kEOS)
          m_index.push_back(m_words.size());
        last = *i;
      }
    }

    UTIL_THROW_IF2(last != m_kEOS, "Input binary file must end with word ID of </s>.");
    //m_index.pop_back(); // NOTE: nothing starts here - points past the end. But we keep the sentinel.
    UTIL_THROW_IF2(m_index.size() <= 1, "Empty file!");
  }

  /** Reads sentence isent into target, returning the number of words including </s>. */
  size_t GetSentence(size_t isent, Width *target, Width *targetEnd) const {
    UTIL_THROW_IF2(isent >= m_index.size() - 1, "Out of bounds!");
    size_t nwords = m_index[isent+1] - m_index[isent];
    //UTIL_THROW_IF2(nwords > maxWords, "Sentence too large!");
    size_t i;
    for(i = m_index[isent]; i < m_index[isent+1] && target != targetEnd; i++)
      *target++ = m_words[i];
    UTIL_THROW_IF2(i < m_index[isent+1], "Sentence too large!");
    return nwords;
  }

  size_t nsents() {
    return m_index.size() - 1; // exclude trailing sentinel
  }

private:
  std::vector<size_t> m_index; //< index of sentence beginnings in m_words. Has trailing sentinel past the last sentence.
  std::vector<Width> m_words; //< raw corpus: a sequence of word-IDs
  Width m_kEOS; //< </s> end-of-sentence marker ID

  /** Attempt to guess file size, return 0 otherwise. */
  size_t fileSize(int fd_in) {
    struct stat s;
    if(fstat(fd_in, &s))
      return 0;
    return s.st_size;
  }
};


template <class Model, class Width> void ConvertToBytes(const Model &model, int fd_in) {
  util::FilePiece in(fd_in);
  util::FakeOFStream out(1);
  Width width;
  StringPiece word;
  const Width end_sentence = (Width)model.GetVocabulary().EndSentence();
  while (true) {
    while (in.ReadWordSameLine(word)) {
      width = (Width)model.GetVocabulary().Index(word);
      out.write(&width, sizeof(Width));
    }
    if (!in.ReadLineOrEOF(word)) break;
    out.write(&end_sentence, sizeof(Width));
  }
}

using namespace lm;
using namespace lm::ngram;
using namespace lm::ngram::detail;

template <class Model, class Width> void _QueryFromBytes(const Model &model, int fd_in) {
  Width kEOS = model.GetVocabulary().EndSentence();
  Width buf[4096000];
  float sum = 0.0;
  //State state = model.BeginSentenceState(), new_state;
  
  Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width> sentence(model);
  
  //sentence.Init();
  
  Width *t = sentence.GetBuf();
  const Width *const tend = sentence.GetBufEnd();
  
  std::size_t got;
  const Width *end;
  if((got = util::ReadOrEOF(fd_in, buf, sizeof(buf)))) {
    UTIL_THROW_IF2(got % sizeof(Width), "File size not a multiple of vocab id size " << sizeof(Width));
    got /= sizeof(Width);
    end = buf + got;
  }
  std::size_t more = util::ReadOrEOF(fd_in, buf, sizeof(buf));
  assert(more == 0);

  const Width *i;
  for (i = buf; i != end; i++) {
    /*
    sum += model.FullScore(state, *i, new_state).prob;
    state = (*i == kEOS) ? model.BeginSentenceState() : new_state;
    */
    
    // the same with class Sentence:
    
    *t++ = *i;
    assert(t != tend);
    
    if(*i == kEOS) {
      //sentence.
      sentence.Init();
      while(sentence.RunState());
      // done here, can submit new work (in next while iteration)
      sum += sentence.GetSum();
      
      t = sentence.GetBuf();
    }
  }
  
  std::cout << "Sum is " << sum << std::endl;
}


template <class Model, class Width> void QueryFromBytes(const Model &model, int fd_in, size_t nthreads) {
  lm::ngram::State state[3];
  const lm::ngram::State *const begin_state = &model.BeginSentenceState();
  const lm::ngram::State *next_state = begin_state;
  Width kEOS = model.GetVocabulary().EndSentence();
  Width buf[4096];
  float sum = 0.0;
  uint64_t completed = 0;

  if(nthreads != 1)
    std::cerr << "WARNING: ignoring nthreads (not implemented here)." << std::endl;

  double loaded = util::CPUTime();
  std::cout << "After loading: ";
  util::PrintUsage(std::cout);

  while (std::size_t got = util::ReadOrEOF(fd_in, buf, sizeof(buf))) {
    UTIL_THROW_IF2(got % sizeof(Width), "File size not a multiple of vocab id size " << sizeof(Width));
    got /= sizeof(Width);
    completed += got;
    // Do even stuff first.
    const Width *even_end = buf + (got & ~1);
    // Alternating states
    const Width *i;
    for (i = buf; i != even_end;) {
      sum += model.FullScore(*next_state, *i, state[1]).prob;
      next_state = (*i++ == kEOS) ? begin_state : &state[1];
      sum += model.FullScore(*next_state, *i, state[0]).prob;
      next_state = (*i++ == kEOS) ? begin_state : &state[0];
    }
    // Odd corner case.
    if (got & 1) {
      sum += model.FullScore(*next_state, *i, state[2]).prob;
      next_state = (*i++ == kEOS) ? begin_state : &state[2];
    }
  }
  std::cerr << "Probability sum is " << sum << std::endl;

  std::cout << "CPU_excluding_load: " << (util::CPUTime() - loaded) << " CPU_per_query: " << ((util::CPUTime() - loaded) / static_cast<double>(completed)) << std::endl;
}


int nprefetch = 1;

/*
template<class Model, class Width>
struct QfbhcParams {
  const Model& model;
  int ithread;
  const Corpus<Model, Width>& corpus;
  size_t isent_begin;
  size_t isent_end;
};
*/

// model, ithread, corpus, isent_begin, isent_end, partialSumOut

template<class Model, class Width> void QueryFromBytes_Hash_Cache(const Model &model, int ithread, const Corpus<Model, Width>& corpus, size_t isent_begin, size_t isent_end, float &partialSumOut) {
  //const int nprefetch = 5;
  //Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width> sentence(model);
  int isent = 0;
  bool prefetching = true;
  int n = 0;
  
  uint64_t completed = 0;

  std::cerr << "ithread = " << ithread << std::endl;
  std::cerr << "nprefetch = " << nprefetch << std::endl;
  
  double loaded = util::CPUTime();
  std::cout << "After loading: ";
  util::PrintUsage(std::cout);
  
  Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width> *sentences[nprefetch];
  
  for(int i = 0; i < nprefetch; i++)
    sentences[i] = new Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width>(model);
  
  Width buf[4096];
  const Width *i = buf;
  float sum = 0.0;

  //while(got) {
  for(size_t i = isent_begin; i < isent_end; i++) {
    // get sentence from buffer
    sentences[isent]->SetSize(corpus.GetSentence(i, sentences[isent]->GetBuf(), sentences[isent]->GetBufEnd()));

    completed += sentences[isent]->Size();

    sentences[isent]->Init();

    if(prefetching) {
      if(isent < nprefetch - 1) {
        isent++;
        continue;
      } else {
        isent = 0;
        prefetching = false;
      }
    }
    
    while(sentences[isent]->RunState()) {
      n++;
      if(++isent == nprefetch)
        isent = 0;
    }
    n++;
    
    // done here, can submit new work (in next while iteration)
    float f = sentences[isent]->GetSum();
    //std::cout << " isent " << isent << " partial sum " << f << std::endl;
    sum += f;
    
    sentences[isent]->FeedInit();
  }

  const size_t nsents = isent_end - isent_begin;

  // mark sentence as finished
  delete sentences[isent];
  sentences[isent] = NULL;
  if(++isent == nprefetch)
    isent = 0;
  if(isent >= nsents) // amendment for very short input (nsents < nprefetch).
    isent = 0;
  
  // when exiting the above loop, there is no more input data, 
  // and we have finished sentences[isent].
  // all other sentences still need advancing to the end.
  for(int i = 0; i < nprefetch - 1 && i < nsents; i++) {
    while(sentences[isent]->RunState())
      n++;
    n++;
    
    // done here
    float f = sentences[isent]->GetSum();
    //std::cout << " isent " << isent << " partial sum " << f << std::endl;
    sum += f;
    
    // mark sentence as finished
    delete sentences[isent];
    sentences[isent] = NULL;
    
    if(++isent == nprefetch)
      isent = 0;
  }
  /*
  for(int i = 0; i < nprefetch; i++)
    delete sentences[i];
  */
  
  std::cerr << "n is " << n << std::endl;
  std::cerr << "Probability sum is " << sum << std::endl;

  partialSumOut = sum;

  std::cout << "CPU_excluding_load: " << (util::CPUTime() - loaded) << " CPU_per_query: " << ((util::CPUTime() - loaded) / static_cast<double>(completed)) << std::endl;
}

                                                       // static_cast<const void *>(&model), i, &corpus, 0, corpus.nsents(), &partialSum[i], 0
template<class Model, class Width> void QueryFromBytes_Hash_CacheX(const void *model, int ithread, const void* corpus, size_t isent_begin, size_t isent_end, float *partialSumOut, int unused) {
  QueryFromBytes_Hash_Cache(*static_cast<const Model *>(model), ithread, *static_cast<const Corpus<Model, Width> *>(corpus), isent_begin, isent_end, *partialSumOut);
}

  /*
template<class Model, class Width>
//void call(const void* model, int ithread, const void* corpus, float *partialSumOut) {
void call(const void *model, int ithread, const void* corpus, size_t isent_begin, size_t isent_end, float *partialSumOut, int unused) {
  const Model *p = static_cast<const Model *>(model);
  std::cout << "call(" << p << ")" << std::endl;
}
   */

template<class Model, class Width> void QueryFromBytes_Hash(const Model &model, int fd_in, size_t nthreads) {
  Corpus<Model, Width> corpus(fd_in, model.GetVocabulary().EndSentence());

  if(nthreads != 1)
    std::cerr << "WARNING: ignoring nthreads (not implemented here)." << std::endl;

  float partialSum[nthreads], totalSum = 0.0;
  //std::thread *workers = new std::thread[nthreads];
  std::thread workers[nthreads];

  // start individual worker threads
  // TODO: sentence boundaries
  for(size_t i = 0; i < nthreads; i++)
    workers[i] = std::thread(QueryFromBytes_Hash_CacheX<Model, Width>, static_cast<const void *>(&model), i, static_cast<const void *>(&corpus), 0, corpus.nsents(), &partialSum[i], 0);
    //workers[i] = std::thread(QueryFromBytes_Hash_Cache<Model, Width>, static_cast<const void *>(&model), i, static_cast<const void *>(&corpus), 0, corpus.nsents(), &partialSum[i], 0);
    //workers[i] = std::move(std::thread(call));
    //workers[i] = std::thread(call<Model, Width>, static_cast<const void *>(&model), 0, static_cast<const void *>(&corpus), &partialSum[i]);
    //workers[i] = std::thread(call<Model, Width>, static_cast<const void *>(&model), i, static_cast<const void *>(&corpus), 0, corpus.nsents(), &partialSum[i], 0);

  //QueryFromBytes_Hash_Cache<Model, Width>(&model, 0, &corpus, 0, corpus.nsents(), &partialSum[0]);

  // wait for results
  for(size_t i = 0; i < nthreads; i++)
    workers[i].join();

  // collect individual workers' results
  for(size_t i = 0; i < nthreads; i++)
    totalSum += partialSum[i];

  //delete[] workers;

  std::cerr << "totalSum = " << totalSum << std::endl;
}


  // specialize...
template<> void QueryFromBytes<lm::ngram::ProbingModel, uint8_t>(const lm::ngram::ProbingModel &model, int fd_in, size_t nthreads) {
  QueryFromBytes_Hash<lm::ngram::ProbingModel, uint8_t>(model, fd_in, nthreads);
}
template<> void QueryFromBytes<lm::ngram::ProbingModel, uint16_t>(const lm::ngram::ProbingModel &model, int fd_in, size_t nthreads) {
  QueryFromBytes_Hash<lm::ngram::ProbingModel, uint16_t>(model, fd_in, nthreads);
}
template<> void QueryFromBytes<lm::ngram::ProbingModel, uint32_t>(const lm::ngram::ProbingModel &model, int fd_in, size_t nthreads) {
  QueryFromBytes_Hash<lm::ngram::ProbingModel, uint32_t>(model, fd_in, nthreads);
}
template<> void QueryFromBytes<lm::ngram::ProbingModel, uint64_t>(const lm::ngram::ProbingModel &model, int fd_in, size_t nthreads) {
  QueryFromBytes_Hash<lm::ngram::ProbingModel, uint64_t>(model, fd_in, nthreads);
}

//LM_NAME_MODEL(ProbingModel, detail::GenericModel<detail::HashedSearch<BackoffValue> LM_COMMA() ProbingVocabulary>);


template <class Model, class Width> void DispatchFunction(const Model &model, bool query, size_t nthreads) {
  if (query) {
    QueryFromBytes<Model, Width>(model, 0, nthreads);
  } else {
    ConvertToBytes<Model, Width>(model, 0);
  }
}

template <class Model> void DispatchWidth(const char *file, bool query, size_t nthreads) {
  lm::ngram::Config config;
  config.load_method = util::READ;
  std::cerr << "Using load_method = READ." << std::endl;
  Model model(file, config);
  lm::WordIndex bound = model.GetVocabulary().Bound();
  if (bound <= 256) {
    DispatchFunction<Model, uint8_t>(model, query, nthreads);
  } else if (bound <= 65536) {
    DispatchFunction<Model, uint16_t>(model, query, nthreads);
  } else if (bound <= (1ULL << 32)) {
    DispatchFunction<Model, uint32_t>(model, query, nthreads);
  } else {
    DispatchFunction<Model, uint64_t>(model, query, nthreads);
  }
}

void Dispatch(const char *file, bool query, size_t nthreads) {
  using namespace lm::ngram;
  lm::ngram::ModelType model_type;
  if (lm::ngram::RecognizeBinary(file, model_type)) {
    switch(model_type) {
      case PROBING:
        DispatchWidth<lm::ngram::ProbingModel>(file, query, nthreads);
        break;
      case REST_PROBING:
        DispatchWidth<lm::ngram::RestProbingModel>(file, query, nthreads);
        break;
      case TRIE:
        DispatchWidth<lm::ngram::TrieModel>(file, query, nthreads);
        break;
      case QUANT_TRIE:
        DispatchWidth<lm::ngram::QuantTrieModel>(file, query, nthreads);
        break;
      case ARRAY_TRIE:
        DispatchWidth<lm::ngram::ArrayTrieModel>(file, query, nthreads);
        break;
      case QUANT_ARRAY_TRIE:
        DispatchWidth<lm::ngram::QuantArrayTrieModel>(file, query, nthreads);
        break;
      default:
        UTIL_THROW(util::Exception, "Unrecognized kenlm model type " << model_type);
    }
  } else {
    UTIL_THROW(util::Exception, "Binarize before running benchmarks.");
  }
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 3 || (strcmp(argv[1], "vocab") && strcmp(argv[1], "query"))) {
    std::cerr
      << "Benchmark program for KenLM.  Intended usage:\n"
      << "#Convert text to vocabulary ids offline.  These ids are tied to a model.\n"
      << argv[0] << " vocab $model [nprefetch] <$text >$text.vocab\n"
      << "#Ensure files are in RAM.\n"
      << "cat $text.vocab $model >/dev/null\n"
      << "#Timed query against the model, including loading.\n"
      << "time " << argv[0] << " query $model <$text.vocab\n";
    return 1;
  }
  if(argc > 3)
    nprefetch = atoi(argv[3]);
  Dispatch(argv[2], !strcmp(argv[1], "query"), 1);
  util::PrintUsage(std::cerr);
  return 0;
}

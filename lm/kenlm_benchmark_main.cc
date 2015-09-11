#include "lm/model.hh"
#include "util/fake_ofstream.hh"
#include "util/file.hh"
#include "util/file_piece.hh"
#include "util/usage.hh"

#include <stdint.h>

namespace {

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


template <class Model, class Width> void QueryFromBytes(const Model &model, int fd_in) {
  lm::ngram::State state[3];
  const lm::ngram::State *const begin_state = &model.BeginSentenceState();
  const lm::ngram::State *next_state = begin_state;
  Width kEOS = model.GetVocabulary().EndSentence();
  Width buf[4096];
  float sum = 0.0;
  while (true) {
    std::size_t got = util::ReadOrEOF(fd_in, buf, sizeof(buf));
    if (!got) break;
    UTIL_THROW_IF2(got % sizeof(Width), "File size not a multiple of vocab id size " << sizeof(Width));
    got /= sizeof(Width);
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
  std::cout << "Sum is " << sum << std::endl;
}


template<class Model, class Width> void QueryFromBytes_Hash(const Model &model, int fd_in) {
  const int nprefetch = 2;
  //Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width> sentence(model);
  int isent = 0;
  bool prefetching = true;
  int n = 0;
  
  Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width> *sentences[nprefetch];
  
  for(int i = 0; i < nprefetch; i++)
    sentences[i] = new Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width>(model);
  
  Width buf[4096];
  const Width *i = buf, *ibak;
  float sum = 0.0;
  std::size_t got = util::ReadOrEOF(fd_in, buf, sizeof(buf));
  UTIL_THROW_IF2(got % sizeof(Width), "File read size " << got << " not a multiple of vocab id size " << sizeof(Width));
  got /= sizeof(Width);
  ibak = buf;
  while(got) {
    std::cout << "Feeding from " << (i - buf) << " to isent " << isent << std::endl;
    ibak = i;

    if(sentences[isent]->FeedBuffer(i, buf + got)) {
      std::cout << "Reading more..." << std::endl;
      got = util::ReadOrEOF(fd_in, buf, sizeof(buf));
      UTIL_THROW_IF2(got % sizeof(Width), "File read size " << got << " not a multiple of vocab id size " << sizeof(Width));
      got /= sizeof(Width);
      std::cout << "Read "<< got << "." << std::endl;
      i = buf;
      // feed more data
      continue;
    }
    std::cout << "Fed " << (i - ibak) << " to isent " << isent << std::endl;

    sentences[isent]->Init();

    if(prefetching) {
      // TODO: prefetch ONLY here.
      if(isent < nprefetch - 1) {
        isent++;
        continue;
      } else {
        isent = 0;
        prefetching = false;
      }
    }
    
    // TODO: prefetch and run here.
    while(sentences[isent]->RunState()) {
      n++;
      if(++isent == nprefetch)
        isent = 0;
    }
    n++;
    
    // done here, can submit new work (in next while iteration)
    float f = sentences[isent]->GetSum();
    std::cout << " isent " << isent << " partial sum " << f << std::endl;
    sum += f;
    
    sentences[isent]->FeedInit();
  }

  // mark sentence as finished
  delete sentences[isent];
  sentences[isent] = NULL;
  if(++isent == nprefetch)
    isent = 0;
  
  // when exiting the above loop, there is no more input data, 
  // and we have finished sentences[isent].
  // all other sentences still need advancing to the end.
  for(int i = 0; i < nprefetch - 1; i++) {
    while(sentences[isent]->RunState())
      n++;
    n++;
    
    // done here
    float f = sentences[isent]->GetSum();
    std::cout << " isent " << isent << " partial sum " << f << std::endl;
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
  
  std::cout << "n is " << n << std::endl;
  std::cout << "Sum is " << sum << std::endl;
}

// specialize...
template<> void QueryFromBytes<lm::ngram::ProbingModel, uint8_t>(const lm::ngram::ProbingModel &model, int fd_in) {
  QueryFromBytes_Hash<lm::ngram::ProbingModel, uint8_t>(model, fd_in);
}
template<> void QueryFromBytes<lm::ngram::ProbingModel, uint16_t>(const lm::ngram::ProbingModel &model, int fd_in) {
  QueryFromBytes_Hash<lm::ngram::ProbingModel, uint16_t>(model, fd_in);
}
template<> void QueryFromBytes<lm::ngram::ProbingModel, uint32_t>(const lm::ngram::ProbingModel &model, int fd_in) {
  QueryFromBytes_Hash<lm::ngram::ProbingModel, uint32_t>(model, fd_in);
}
template<> void QueryFromBytes<lm::ngram::ProbingModel, uint64_t>(const lm::ngram::ProbingModel &model, int fd_in) {
  QueryFromBytes_Hash<lm::ngram::ProbingModel, uint64_t>(model, fd_in);
}

//LM_NAME_MODEL(ProbingModel, detail::GenericModel<detail::HashedSearch<BackoffValue> LM_COMMA() ProbingVocabulary>);


template <class Model, class Width> void DispatchFunction(const Model &model, bool query) {
  if (query) {
    QueryFromBytes<Model, Width>(model, 0);
  } else {
    ConvertToBytes<Model, Width>(model, 0);
  }
}

template <class Model> void DispatchWidth(const char *file, bool query) {
  lm::ngram::Config config;
  config.load_method = util::READ;
  std::cerr << "Using load_method = READ." << std::endl;
  Model model(file, config);
  lm::WordIndex bound = model.GetVocabulary().Bound();
  if (bound <= 256) {
    DispatchFunction<Model, uint8_t>(model, query);
  } else if (bound <= 65536) {
    DispatchFunction<Model, uint16_t>(model, query);
  } else if (bound <= (1ULL << 32)) {
    DispatchFunction<Model, uint32_t>(model, query);
  } else {
    DispatchFunction<Model, uint64_t>(model, query);
  }
}

void Dispatch(const char *file, bool query) {
  using namespace lm::ngram;
  lm::ngram::ModelType model_type;
  if (lm::ngram::RecognizeBinary(file, model_type)) {
    switch(model_type) {
      case PROBING:
        DispatchWidth<lm::ngram::ProbingModel>(file, query);
        break;
      case REST_PROBING:
        DispatchWidth<lm::ngram::RestProbingModel>(file, query);
        break;
      case TRIE:
        DispatchWidth<lm::ngram::TrieModel>(file, query);
        break;
      case QUANT_TRIE:
        DispatchWidth<lm::ngram::QuantTrieModel>(file, query);
        break;
      case ARRAY_TRIE:
        DispatchWidth<lm::ngram::ArrayTrieModel>(file, query);
        break;
      case QUANT_ARRAY_TRIE:
        DispatchWidth<lm::ngram::QuantArrayTrieModel>(file, query);
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
  if (argc != 3 || (strcmp(argv[1], "vocab") && strcmp(argv[1], "query"))) {
    std::cerr
      << "Benchmark program for KenLM.  Intended usage:\n"
      << "#Convert text to vocabulary ids offline.  These ids are tied to a model.\n"
      << argv[0] << " vocab $model <$text >$text.vocab\n"
      << "#Ensure files are in RAM.\n"
      << "cat $text.vocab $model >/dev/null\n"
      << "#Timed query against the model, including loading.\n"
      << "time " << argv[0] << " query $model <$text.vocab\n";
    return 1;
  }
  Dispatch(argv[2], !strcmp(argv[1], "query"));
  util::PrintUsage(std::cerr);
  return 0;
}

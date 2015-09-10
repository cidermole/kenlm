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

template <class Model, class Width> void __QueryFromBytes(const Model &model, int fd_in) {
  Width kEOS = model.GetVocabulary().EndSentence();
  Width buf[4096];
  float sum = 0.0;
  State state = model.BeginSentenceState(), new_state;
  
  Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width> sentence(model);
  
  sentence.Init();
  
  while (std::size_t got = util::ReadOrEOF(fd_in, buf, sizeof(buf))) {
    UTIL_THROW_IF2(got % sizeof(Width), "File size not a multiple of vocab id size " << sizeof(Width));
    got /= sizeof(Width);
    const Width *end = buf + got;
    // Alternating states
    
    const Width *i;
    for (i = buf; i != end; i++) {
      /*
      sum += model.FullScore(state, *i, new_state).prob;
      state = (*i == kEOS) ? model.BeginSentenceState() : new_state;
      */
      
      // TODO: work in progress...
      
      if(*i == kEOS) {
        //sentence.
        sentence.Init();
      }
    }
  }
  std::cerr << "Probability sum is " << sum << std::endl;

  std::cout << "CPU_excluding_load: " << (util::CPUTime() - loaded) << " CPU_per_query: " << ((util::CPUTime() - loaded) / static_cast<double>(completed)) << std::endl;
}

template <class Model, class Width> void QueryFromBytes(const Model &model, int fd_in) {
  const int nprefetch = 5;
  //Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width> sentence(model);
  int isent = 0;
  bool prefetching = true;
  
  Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width> *sentences[nprefetch];
  
  for(int i = 0; i < nprefetch; i++)
    sentences[i] = new Sentence<typename Model::SearchType, typename Model::VocabularyType, Model, Width>(model);
  
  Width kEOS = model.GetVocabulary().EndSentence();
  Width buf[4096];
  float sum = 0.0;
  Width *t = sentences[isent]->GetBuf();
  const Width *tend = sentences[isent]->GetBufEnd();
  while(std::size_t got = util::ReadOrEOF(fd_in, buf, sizeof(buf))) {
    UTIL_THROW_IF2(got % sizeof(Width), "File size not a multiple of vocab id size " << sizeof(Width));
    got /= sizeof(Width);
    
    const Width *i;
    const Width *end = buf + got;
    for(i = buf; i != end && t != tend && *i != kEOS; i++, t++)
      *t = *i;
    assert(t != tend); // assume that each sentence fits into < 4096 chars.
    if(*i != kEOS)
      continue;
    *t = kEOS; // TODO: if very last sentence does not have end symbol, then that will not be fed...
    
    sentences[isent]->Init();

    if(prefetching) {
      // TODO: prefetch ONLY here.
      if(isent < nprefetch - 1)
        isent++;
      else {
        isent = 0;
        prefetching = false;
      }
    } else {
      // TODO: prefetch and run here.
      while(sentences[isent]->RunState()) {
        if(++isent == nprefetch)
          isent = 0;
      }
      
      // done here, can submit new work (in next while iteration)
      sum += sentences[isent]->GetSum();
    }
  }
  
  for(int i = 0; i < nprefetch; i++)
    delete sentences[i];
  
  std::cout << "Sum is " << sum << std::endl;
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

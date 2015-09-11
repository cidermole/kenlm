#ifndef LM_MODEL_H
#define LM_MODEL_H

#include "lm/bhiksha.hh"
#include "lm/binary_format.hh"
#include "lm/config.hh"
#include "lm/facade.hh"
#include "lm/quantize.hh"
#include "lm/search_hashed.hh"
#include "lm/search_trie.hh"
#include "lm/state.hh"
#include "lm/value.hh"
#include "lm/vocab.hh"
#include "lm/weights.hh"

#include "util/murmur_hash.hh"

#include <algorithm>
#include <vector>
#include <cstring>

namespace util { class FilePiece; }

namespace lm {
namespace ngram {  
namespace detail {

void CopyRemainingHistory(const WordIndex *from, State &out_state);

// Should return the same results as SRI.
// ModelFacade typedefs Vocabulary so we use VocabularyT to avoid naming conflicts.
template <class Search, class VocabularyT> class GenericModel : public base::ModelFacade<GenericModel<Search, VocabularyT>, State, VocabularyT> {
  private:
    typedef base::ModelFacade<GenericModel<Search, VocabularyT>, State, VocabularyT> P;
  public:
    typedef Search SearchType;
    typedef VocabularyT VocabularyType;
    
    // This is the model type returned by RecognizeBinary.
    static const ModelType kModelType;

    static const unsigned int kVersion = Search::kVersion;

    /* Get the size of memory that will be mapped given ngram counts.  This
     * does not include small non-mapped control structures, such as this class
     * itself.
     */
    static uint64_t Size(const std::vector<uint64_t> &counts, const Config &config = Config());

    /* Load the model from a file.  It may be an ARPA or binary file.  Binary
     * files must have the format expected by this class or you'll get an
     * exception.  So TrieModel can only load ARPA or binary created by
     * TrieModel.  To classify binary files, call RecognizeBinary in
     * lm/binary_format.hh.
     */
    explicit GenericModel(const char *file, const Config &config = Config());

    /* Score p(new_word | in_state) and incorporate new_word into out_state.
     * Note that in_state and out_state must be different references:
     * &in_state != &out_state.
     */
    FullScoreReturn FullScore(const State &in_state, const WordIndex new_word, State &out_state) const;

    /* Slower call without in_state.  Try to remember state, but sometimes it
     * would cost too much memory or your decoder isn't setup properly.
     * To use this function, make an array of WordIndex containing the context
     * vocabulary ids in reverse order.  Then, pass the bounds of the array:
     * [context_rbegin, context_rend).  The new_word is not part of the context
     * array unless you intend to repeat words.
     */
    FullScoreReturn FullScoreForgotState(const WordIndex *context_rbegin, const WordIndex *context_rend, const WordIndex new_word, State &out_state) const;

    /* Get the state for a context.  Don't use this if you can avoid it.  Use
     * BeginSentenceState or NullContextState and extend from those.  If
     * you're only going to use this state to call FullScore once, use
     * FullScoreForgotState.
     * To use this function, make an array of WordIndex containing the context
     * vocabulary ids in reverse order.  Then, pass the bounds of the array:
     * [context_rbegin, context_rend).
     */
    void GetState(const WordIndex *context_rbegin, const WordIndex *context_rend, State &out_state) const;

    /* More efficient version of FullScore where a partial n-gram has already
     * been scored.
     * NOTE: THE RETURNED .rest AND .prob ARE RELATIVE TO THE .rest RETURNED BEFORE.
     */
    FullScoreReturn ExtendLeft(
        // Additional context in reverse order.  This will update add_rend to
        const WordIndex *add_rbegin, const WordIndex *add_rend,
        // Backoff weights to use.
        const float *backoff_in,
        // extend_left returned by a previous query.
        uint64_t extend_pointer,
        // Length of n-gram that the pointer corresponds to.
        unsigned char extend_length,
        // Where to write additional backoffs for [extend_length + 1, min(Order() - 1, return.ngram_length)]
        float *backoff_out,
        // Amount of additional content that should be considered by the next call.
        unsigned char &next_use) const;

    /* Return probabilities minus rest costs for an array of pointers.  The
     * first length should be the length of the n-gram to which pointers_begin
     * points.
     */
    float UnRest(const uint64_t *pointers_begin, const uint64_t *pointers_end, unsigned char first_length) const {
      // Compiler should optimize this if away.
      return Search::kDifferentRest ? InternalUnRest(pointers_begin, pointers_end, first_length) : 0.0;
    }
    
    const Search &GetSearch() const { return search_; }
    const VocabularyT &GetVocab() const { return vocab_; }

  private:
    FullScoreReturn ScoreExceptBackoff(const WordIndex *const context_rbegin, const WordIndex *const context_rend, const WordIndex new_word, State &out_state) const;
    
    FullScoreReturn __ScoreExceptBackoff(const WordIndex *const context_rbegin, const WordIndex *const context_rend, const WordIndex new_word, State &out_state) const;

    // Score bigrams and above.  Do not include backoff.
    void ResumeScore(const WordIndex *context_rbegin, const WordIndex *const context_rend, unsigned char starting_order_minus_2, typename Search::Node &node, float *backoff_out, unsigned char &next_use, FullScoreReturn &ret) const;

    // Appears after Size in the cc file.
    void SetupMemory(void *start, const std::vector<uint64_t> &counts, const Config &config);

    void InitializeFromARPA(int fd, const char *file, const Config &config);

    float InternalUnRest(const uint64_t *pointers_begin, const uint64_t *pointers_end, unsigned char first_length) const;

    BinaryFormat backing_;

    VocabularyT vocab_;

    Search search_;
};


/**
 * Keeping state for pipelined single hashtable queries: what was in
 *  - ScoreExceptBackoff()
 *  - ResumeScore()
 * before.
 * State-machine setting off queries for a succession of n-gram orders.
 */
template <class Search, class VocabularyT>
class Lookup {
public:
  Lookup(const VocabularyT &vocab, const Search &search, unsigned char order): vocab_(vocab), search_(search), order_(order) {}
  
  void Init(const WordIndex *const context_rbegin, const WordIndex *const context_rend, const WordIndex new_word) {
    order_minus_2 = 0;
    
    this->context_rbegin = context_rbegin;
    this->context_rend = context_rend;
    this->new_word = new_word;
    this->node = typename Search::Node();
    
    assert(new_word < vocab_.Bound());
    // ret.ngram_length contains the last known non-blank ngram length.
    ret.ngram_length = 1;

    typename Search::UnigramPointer uni(search_.LookupUnigram(new_word, node, ret.independent_left, ret.extend_left));
    out_state.backoff[0] = uni.Backoff();
    ret.prob = uni.Prob();
    ret.rest = uni.Rest();

    // This is the length of the context that should be used for continuation to the right.
    out_state.length = HasExtension(out_state.backoff[0]) ? 1 : 0;
    // We'll write the word anyway since it will probably be used and does no harm being there.
    out_state.words[0] = new_word;
    
    hist_iter = context_rbegin;
    backoff_out = out_state.backoff + 1;
    
    // prefetch first address, if necessary
    if (context_rbegin == context_rend)
      return;
    // TODO: could prefetch Unigrams (but then, move LookupUnigram to RunState()...)
    
    // prefetch
    it = search_.LookupMiddleIterator(order_minus_2, *hist_iter, node);
    __builtin_prefetch(it);
  }
  
  /** Returns true if still needs to run. */
  bool RunState() {
    if (hist_iter == context_rend) return Final();
    if (ret.independent_left) return Final();
    if (order_minus_2 == Order() - 2) {
      Longest();
      return Final();
    }

    typename Search::MiddlePointer pointer(search_.LookupMiddleFromIterator(order_minus_2, node, ret.independent_left, ret.extend_left, it));
    if (!pointer.Found()) return Final();
    *backoff_out = pointer.Backoff();
    ret.prob = pointer.Prob();
    ret.rest = pointer.Rest();
    ret.ngram_length = order_minus_2 + 2;
    if (HasExtension(*backoff_out)) {
      out_state.length = ret.ngram_length;
    }
    
    ++order_minus_2, ++hist_iter, ++backoff_out;
    
    // prefetch
    if (hist_iter != context_rend && order_minus_2 != Order() - 2) {
      it = search_.LookupMiddleIterator(order_minus_2, *hist_iter, node);
      __builtin_prefetch(it);
    }
    
    return true;
  }
  
  State &GetOutState() { return out_state; }
  
  // TODO: contrary to the name, this score does not yet include backoffs.
  FullScoreReturn &GetRet() { return ret; }
  
private:
  unsigned char Order() { return order_; }
  
  void Longest() {
    ret.independent_left = true;
    typename Search::LongestPointer longest(search_.LookupLongest(*hist_iter, node));
    if (longest.Found()) {
      ret.prob = longest.Prob();
      ret.rest = ret.prob;
      // There is no blank in longest_.
      ret.ngram_length = Order();
    }
  }
    
  bool Final() {
    CopyRemainingHistory(context_rbegin, out_state);
    return false;
  }
  
  unsigned char order_minus_2;
  FullScoreReturn ret;
  typename Search::Node node;
  typename Search::Middle::ConstIterator it;

  const WordIndex *hist_iter;
  float *backoff_out;
  
  const WordIndex *context_rbegin;
  const WordIndex *context_rend;
  WordIndex new_word;
  State out_state;
  
  // from GenericModel
  const VocabularyT &vocab_;
  const Search &search_;
  unsigned char order_;
};



template<class Search, class VocabularyT, class Model, class Width>
class Sentence {
public:
  // TODO: model and vocab/search somewhat redundant
  Sentence(const Model &model, const VocabularyT &vocab, const Search &search, unsigned char order): ibuf(buf), model(model), sum(0), lookup(vocab, search, order), kEOS(model.GetVocabulary().EndSentence()) {}
  
  Sentence(const Model &model): ibuf(buf), model(model), sum(0), lookup(model.GetVocab(), model.GetSearch(), model.Order()), kEOS(model.GetVocabulary().EndSentence()) {}
  
  Width *GetBuf() { return buf; }
  Width *GetBufEnd() { return buf + sizeof(buf)/sizeof(Width); }
  
  void FeedInit() {
    ibuf = buf;
  }
  
  /** Returns true if still needs more data. omnomnom! */
  bool FeedBuffer(const Width *&buf, const Width *buf_end) {
    const Width* const end = GetBufEnd();
    while(buf != buf_end && ibuf != end) {
      *ibuf = *buf++;
      if(*ibuf++ == kEOS)
        return false;
    }
    assert(ibuf != end); // assume that each sentence fits into < 4096 chars.
    return true;
  }
  
  float GetSum() { return sum; }
  
  /** Must call this after FeedBuffer() and before any RunState() calls. */
  void Init() {
    this->ibuf = buf;
    this->sum = 0.0;
    this->state = model.BeginSentenceState(); // copy!!!
    
    // TODO: prefetch??
    lookup.Init(state.words, state.words + state.length, *ibuf);
  }
  
  /** Returns true if still needs to run. */
  bool RunState() {
    if(lookup.RunState())
      // more calls for the same word (different n-gram orders)
      return true;
    
    // have result of lookup for word
    sum += lookup.GetRet().prob;
    for (const float *i = state.backoff + lookup.GetRet().ngram_length - 1; i < state.backoff + state.length; ++i) {
      sum += *i;
    }
    state = lookup.GetOutState();
    
    //state = lookup.GetOutState(); // enough to do once
    //return (*i++ != kEOS);

    if(*ibuf++ == kEOS) {
      return false;
    }
    // init for next word
    lookup.Init(state.words, state.words + state.length, *ibuf);
    return true;
  }
  
  bool _RunState() {
    State new_state;
    sum += model.FullScore(state, *ibuf, new_state).prob;
    state = new_state;
    
    return (*ibuf++ != kEOS);
  }
  
private:  
  Width buf[4096];
  Width *ibuf;
  State state;
  const Model &model;
  float sum;
  Lookup<Search, VocabularyT> lookup;
  const Width kEOS;
};


} // namespace detail

// Instead of typedef, inherit.  This allows the Model etc to be forward declared.
// Oh the joys of C and C++.
#define LM_COMMA() ,
#define LM_NAME_MODEL(name, from)\
class name : public from {\
  public:\
    name(const char *file, const Config &config = Config()) : from(file, config) {}\
};

LM_NAME_MODEL(ProbingModel, detail::GenericModel<detail::HashedSearch<BackoffValue> LM_COMMA() ProbingVocabulary>);
LM_NAME_MODEL(RestProbingModel, detail::GenericModel<detail::HashedSearch<RestValue> LM_COMMA() ProbingVocabulary>);
LM_NAME_MODEL(TrieModel, detail::GenericModel<trie::TrieSearch<DontQuantize LM_COMMA() trie::DontBhiksha> LM_COMMA() SortedVocabulary>);
LM_NAME_MODEL(ArrayTrieModel, detail::GenericModel<trie::TrieSearch<DontQuantize LM_COMMA() trie::ArrayBhiksha> LM_COMMA() SortedVocabulary>);
LM_NAME_MODEL(QuantTrieModel, detail::GenericModel<trie::TrieSearch<SeparatelyQuantize LM_COMMA() trie::DontBhiksha> LM_COMMA() SortedVocabulary>);
LM_NAME_MODEL(QuantArrayTrieModel, detail::GenericModel<trie::TrieSearch<SeparatelyQuantize LM_COMMA() trie::ArrayBhiksha> LM_COMMA() SortedVocabulary>);

// Default implementation.  No real reason for it to be the default.
typedef ::lm::ngram::ProbingVocabulary Vocabulary;
typedef ProbingModel Model;

/* Autorecognize the file type, load, and return the virtual base class.  Don't
 * use the virtual base class if you can avoid it.  Instead, use the above
 * classes as template arguments to your own virtual feature function.*/
base::Model *LoadVirtual(const char *file_name, const Config &config = Config(), ModelType if_arpa = PROBING);

} // namespace ngram
} // namespace lm

#endif // LM_MODEL_H

#include "util/probing_hash_table.hh"
#include "util/scoped.hh"
#include "util/usage.hh"

#include <boost/random/mersenne_twister.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION >= 104700
#include <boost/random/uniform_int_distribution.hpp>
#define UTIL_TWISTER boost::random::mt19937
#define UTIL_INT_DIST boost::random::uniform_int_distribution
#else
#include <boost/random/uniform_int.hpp>
#define UTIL_TWISTER boost::mt19937
#define UTIL_INT_DIST boost::uniform_int
#endif

#include <iostream>
#include <queue>
#include <stdlib.h>

namespace util {
namespace {

struct Entry {
  typedef uint64_t Key;
  Key key;
  Key GetKey() const { return key; }
};

template <class Table>
struct BufferEntry {
  typedef uint64_t Key;
  typename Table::ConstIterator it;
  Key k;
  
  BufferEntry(typename Table::ConstIterator it, Key k): it(it), k(k) {}
  BufferEntry(): it(NULL), k(0) {}
};

template <class Mod> bool Test(uint64_t entries, int qsize, uint64_t lookups = 20000000, float multiplier = 1.5) {
  typedef util::ProbingHashTable<Entry, util::IdentityHash, std::equal_to<Entry::Key>, Mod> Table;
  // Always round up to power of 2 for fair comparison.
  std::size_t size = Power2Mod::RoundBuckets(Table::Size(entries, multiplier) / sizeof(Entry)) * sizeof(Entry);
  scoped_malloc backing(util::CallocOrThrow(size));
  Table table(backing.get(), size);
  UTIL_TWISTER gen;
  UTIL_INT_DIST<uint64_t> dist(std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max());
  double start = UserTime();
  for (uint64_t i = 0; i < entries; ++i) {
    Entry entry;
    entry.key = dist(gen);
    table.Insert(entry);
  }
  double inserted = UserTime();
  bool meaningless = true;
  // gen addrs
  uint64_t *keys = new uint64_t[lookups];
  uint64_t *p = keys;
  for (uint64_t i = 0; i < lookups; ++i)
    keys[i] = dist(gen);

  // benchmark find
  double before_find = UserTime();
  
  std::queue<BufferEntry<Table> > q;
  //const int qsize = 10;
  
  for (uint64_t i = 0; i < lookups; ++i) {
    typename Table::ConstIterator it;
    BufferEntry<Table> be;
    //meaningless ^= table.Find(*p++, it);
    
    if(q.size() == qsize) {
      be = q.front();
      meaningless ^= table.FindFromIterator(be.k, be.it);
      q.pop();
    }
    it = table.Ideal(*p); // compute address in hash
    q.push(BufferEntry<Table>(it, *p++));
    __builtin_prefetch(it);
  }
  
  // drain the queue
  while(q.size()) {
    typename Table::ConstIterator it;
    BufferEntry<Table> be;
    be = q.front();
    meaningless ^= table.FindFromIterator(be.k, be.it);
    q.pop();
  }
  
  
  /*
  for (uint64_t i = 0; i < lookups; ++i) {
    typename Table::ConstIterator it;
    //meaningless ^= table.Find(*p++, it);
    it = table.Ideal(*p);
    meaningless ^= table.FindFromIterator(*p++, it);
  }
  */
  std::cout << entries << ' ' << size << ' ' << (inserted - start) / static_cast<double>(entries) << ' ' << (UserTime() - before_find) / static_cast<double>(lookups) << '\n';

  delete[] keys;
  return meaningless;
}

} // namespace
} // namespace util

int main(int argc, char** argv) {
  int qsize = atoi(argv[1]);
  
  bool meaningless = false;
  std::cout << "#Integer division\n";
  for (uint64_t i = 1; i <= 100000000ULL; i *= 10) {
    meaningless ^= util::Test<util::DivMod>(i, qsize);
  }
  std::cout << "#Masking\n";
  for (uint64_t i = 1; i <= 100000000ULL; i *= 10) {
    meaningless ^= util::Test<util::Power2Mod>(i, qsize);
  }
  std::cerr << "Meaningless: " << meaningless << '\n';
}

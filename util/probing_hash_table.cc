#include "probing_hash_table.hh"
#include "libdivide.h"

namespace util {

Divider::Divider(std::size_t divisor)
  : fast_buckets_(new libdivide::divider<std::size_t, -1>(divisor)), divisor_(divisor)
{
  divisor_ = divisor;
}

Divider::Divider(const Divider &other)
  : fast_buckets_(new libdivide::divider<std::size_t, -1>(other.divisor_)), divisor_(other.divisor_)
{
}

Divider &Divider::operator=(const Divider &other) {
  fast_buckets_.reset(new libdivide::divider<std::size_t, -1>(other.divisor_));
  divisor_ = other.divisor_;
  return *this;
}

uint64_t Divider::divide(uint64_t num) const {
  //return num / (*fast_buckets_);
  return fast_buckets_->perform_divide(num);
  //return num / divisor_;
}

Divider::~Divider() {
}

}

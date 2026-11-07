#include <utility>

#include "Tools/Exception/julia_error/julia_error.hpp"

using namespace spu::tools;

#define DEFAULT_MESSAGE "Julia error."

julia_error::julia_error() noexcept
  : exception(DEFAULT_MESSAGE)
{
}

julia_error::julia_error(std::string&& message) noexcept
  : exception(message.empty() ? DEFAULT_MESSAGE : message)
{
}

julia_error::julia_error(std::string&& filename, int&& line_num, std::string&& funcname, std::string&& message) noexcept
  : exception(std::move(filename),
              std::move(line_num),
              std::move(funcname),
              message.empty() ? DEFAULT_MESSAGE : message)
{
}

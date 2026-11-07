/*!
 * \file
 * \brief Class tools::julia_error.
 */
#ifndef JULIA_ERROR_HPP_
#define JULIA_ERROR_HPP_

#include <string>

#include "Tools/Exception/exception.hpp"

namespace spu
{
namespace tools
{
class SPU_EXPORT_EXCEPTION julia_error : public exception
{
  public:
    julia_error() noexcept;
    explicit julia_error(std::string&& message) noexcept;
    julia_error(std::string&& filename,
                int&& line_num,
                std::string&& funcname = "",
                std::string&& message = "") noexcept;

    virtual ~julia_error() = default;
};
}
}

#endif /* JULIA_ERROR_HPP_ */

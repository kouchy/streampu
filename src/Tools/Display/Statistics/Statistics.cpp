#include <algorithm>
#include <iomanip>
#include <ios>
#include <map>
#include <sstream>
#include <type_traits>

#include "Runtime/Pipeline/Pipeline.hpp"
#include "Runtime/Sequence/Sequence.hpp"
#include "Tools/Display/Statistics/Statistics.hpp"
#include "Tools/Display/rang_format/rang_format.h"
#include "Tools/Exception/exception.hpp"

using namespace spu;
using namespace spu::tools;

void
Statistics::separation1(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    if (display_thr)
        stream << "# " << rang::style::bold << "---------------------------------------------------------------||------------------------------||--------------------------------||--------------------------------" << rang::style::reset << std::endl;
    else
        stream << "# " << rang::style::bold << "---------------------------------------------------------------||------------------------------||--------------------------------" << rang::style::reset << std::endl;
    // clang-format on
}

void
Statistics::separation2(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    if (display_thr)
        stream << "# " << rang::style::bold << "-------------------|-------------------|-----|-------|---------||----------|----------|--------||----------|----------|----------||----------|----------|----------" << rang::style::reset << std::endl;
    else
        stream << "# " << rang::style::bold << "-------------------|-------------------|-----|-------|---------||----------|----------|--------||----------|----------|----------" << rang::style::reset << std::endl;
    // clang-format on
}

void
Statistics::show_header(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    Statistics::separation1(display_thr, stream);
    if (display_thr)
    {
//      stream << "# " << rang::style::bold << "---------------------------------------------------------------||------------------------------||--------------------------------||--------------------------------" << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "                 Statistics for the given task                 ||       Basic statistics       ||       Measured throughput      ||        Measured latency        " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "              ('*' = any, '-' = same as previous)              ||          on the task         ||   considering the last socket  ||                                " << rang::style::reset << std::endl;
//      stream << "# " << rang::style::bold << "---------------------------------------------------------------||------------------------------||--------------------------------||--------------------------------" << rang::style::reset << std::endl;
    }
    else
    {
//      stream << "# " << rang::style::bold << "---------------------------------------------------------------||------------------------------||--------------------------------" << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "                 Statistics for the given task                 ||       Basic statistics       ||        Measured latency        " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "              ('*' = any, '-' = same as previous)              ||          on the task         ||                                " << rang::style::reset << std::endl;
//      stream << "# " << rang::style::bold << "---------------------------------------------------------------||------------------------------||--------------------------------" << rang::style::reset << std::endl;
    }
    Statistics::separation1(display_thr, stream);
    Statistics::separation2(display_thr, stream);
    if (display_thr)
    {
//      stream << "# " << rang::style::bold << "-------------------|-------------------|-----|-------|---------||----------|----------|--------||----------|----------|----------||----------|----------|----------" << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "       MODULE NAME |         TASK NAME | REP | ORDER |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM ||  AVERAGE |  MINIMUM |  MAXIMUM " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "                   |                   |     |       |         ||          |      (s) |    (%) ||   (Mb/s) |   (Mb/s) |   (Mb/s) ||     (us) |     (us) |     (us) " << rang::style::reset << std::endl;
//      stream << "# " << rang::style::bold << "-------------------|-------------------|-----|-------|---------||----------|----------|--------||----------|----------|----------||----------|----------|----------" << rang::style::reset << std::endl;
    }
    else
    {
//      stream << "# " << rang::style::bold << "-------------------|-------------------|-----|-------|---------||----------|----------|--------||----------|----------|----------" << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "       MODULE NAME |         TASK NAME | REP | ORDER |   TIMER ||    CALLS |     TIME |   PERC ||  AVERAGE |  MINIMUM |  MAXIMUM " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "                   |                   |     |       |         ||          |      (s) |    (%) ||     (us) |     (us) |     (us) " << rang::style::reset << std::endl;
//      stream << "# " << rang::style::bold << "-------------------|-------------------|-----|-------|---------||----------|----------|--------||----------|----------|----------" << rang::style::reset << std::endl;
    }
    Statistics::separation2(display_thr, stream);
    // clang-format on
}

void
Statistics::show_task(const float total_sec,
                      const std::string& module_name,
                      const std::string& task_name,
                      const bool task_replicability,
                      const int task_order,
                      const size_t task_n_elmts,
                      const uint32_t task_n_calls,
                      const std::chrono::nanoseconds task_tot_duration,
                      const std::chrono::nanoseconds task_min_duration,
                      const std::chrono::nanoseconds task_max_duration,
                      const bool display_thr,
                      std::ostream& stream)
{
    // clang-format off
    if (task_n_calls == 0)
        return;

    auto tot_dur = ((float)task_tot_duration.count()) * 0.000000001f;
    auto percent = (tot_dur / total_sec) * 100.f;
    auto avg_thr = (float)(task_n_calls * task_n_elmts) / ((float)task_tot_duration.count() * 0.001f);
    auto min_thr = (float)(1.f          * task_n_elmts) / ((float)task_max_duration.count() * 0.001f);
    auto max_thr = (float)(1.f          * task_n_elmts) / ((float)task_min_duration.count() * 0.001f);
    auto avg_lat = (float)(task_tot_duration.count() * 0.001f) / task_n_calls;
    auto min_lat = (float)(task_min_duration.count() * 0.001f);
    auto max_lat = (float)(task_max_duration.count() * 0.001f);

#ifdef _WIN32
    auto P = 1;
#else
    auto P = 2;
#endif

    unsigned l1 = 99999999;
    float    l2 = 99999.99f;

    const std::string& trimmed_module_name = module_name.substr(0, 18);
    const bool module_name_is_trimmed = module_name.length()>18;

    const std::string& trimmed_task_name = task_name.substr(0, 17);
    const bool task_name_is_trimmed = task_name.length()>17;

    std::stringstream ssmodule, ssprocess, ssrep, ssorder, sssp, ssn_calls, sstot_dur, sspercent;
    std::stringstream ssavg_thr, ssmin_thr, ssmax_thr;
    std::stringstream ssavg_lat, ssmin_lat, ssmax_lat;

    ssmodule  << std::setprecision(                        2) <<                                        std::fixed  << std::setw(18) << trimmed_module_name;
    ssprocess << std::setprecision(                        2) <<                                        std::fixed  << std::setw(17) << trimmed_task_name;
    ssrep     << std::setprecision(                        2) <<                                        std::fixed  << std::setw( 3) << (task_replicability ? "yes" : "no");
    if (task_order >= 0)
        ssorder << std::setprecision(                      2) <<                                        std::fixed  << std::setw( 5) << task_order;
    else
        ssorder << std::setprecision(                      2) <<                                        std::fixed  << std::setw( 5) << "*";
    sssp      << std::setprecision(                        2) <<                                        std::fixed  << std::setw( 7) << "*";
    ssn_calls << std::setprecision(task_n_calls > l1 ? P : 2) << (task_n_calls > l1 ? std::scientific : std::fixed) << std::setw( 8) << task_n_calls;
    sstot_dur << std::setprecision(tot_dur      > l1 ? P : 2) << (tot_dur      > l1 ? std::scientific : std::fixed) << std::setw( 8) << tot_dur;
    sspercent << std::setprecision(                        2) <<                                        std::fixed  << std::setw( 6) << percent;
    ssavg_thr << std::setprecision(((avg_thr > l1) || (avg_thr < 1)) ? P : 2) << (((avg_thr > l2) || (avg_thr < 1)) ? std::scientific : std::fixed) << std::setw( 8) << avg_thr;
    ssmin_thr << std::setprecision(((min_thr > l1) || (min_thr < 1)) ? P : 2) << (((min_thr > l2) || (min_thr < 1)) ? std::scientific : std::fixed) << std::setw( 8) << min_thr;
    ssmax_thr << std::setprecision(((max_thr > l1) || (max_thr < 1)) ? P : 2) << (((max_thr > l2) || (max_thr < 1)) ? std::scientific : std::fixed) << std::setw( 8) << max_thr;
    ssavg_lat << std::setprecision(((avg_lat > l1) || (avg_lat < 1)) ? P : 2) << (((avg_lat > l2) || (avg_lat < 1)) ? std::scientific : std::fixed) << std::setw( 8) << avg_lat;
    ssmin_lat << std::setprecision(((min_lat > l1) || (min_lat < 1)) ? P : 2) << (((min_lat > l2) || (min_lat < 1)) ? std::scientific : std::fixed) << std::setw( 8) << min_lat;
    ssmax_lat << std::setprecision(((max_lat > l1) || (max_lat < 1)) ? P : 2) << (((max_lat > l2) || (max_lat < 1)) ? std::scientific : std::fixed) << std::setw( 8) << max_lat;

    stream << "# ";
    stream << ssmodule .str() << rang::style::bold << (module_name_is_trimmed ? "*" : " ") << "| " << rang::style::reset
           << ssprocess.str() << rang::style::bold << (task_name_is_trimmed   ? "*" : " ") << "| " << rang::style::reset
           << ssrep    .str() << rang::style::bold << " | "  << rang::style::reset
           << ssorder  .str() << rang::style::bold << " | "  << rang::style::reset
           << sssp     .str() << rang::style::bold << " || " << rang::style::reset
           << ssn_calls.str() << rang::style::bold << " | "  << rang::style::reset
           << sstot_dur.str() << rang::style::bold << " | "  << rang::style::reset;

         if (percent > 50.0f) stream << rang::fg::red    << sspercent.str() << rang::style::reset;
    else if (percent > 25.0f) stream << rang::fg::yellow << sspercent.str() << rang::style::reset;
    else if (percent > 12.5f) stream << rang::fg::green  << sspercent.str() << rang::style::reset;
    else if (percent <  5.0f) stream << rang::fg::gray   << sspercent.str() << rang::style::reset;
    else                      stream <<                     sspercent.str();

    if (display_thr)
        stream <<                    rang::style::bold << " || " << rang::style::reset
               << ssavg_thr.str() << rang::style::bold << " | "  << rang::style::reset
               << ssmin_thr.str() << rang::style::bold << " | "  << rang::style::reset
               << ssmax_thr.str() << rang::style::bold << " || " << rang::style::reset
               << ssavg_lat.str() << rang::style::bold << " | "  << rang::style::reset
               << ssmin_lat.str() << rang::style::bold << " | "  << rang::style::reset
               << ssmax_lat.str() << ""
               << std::endl;
    else
        stream <<                    rang::style::bold << " || " << rang::style::reset
               << ssavg_lat.str() << rang::style::bold << " | "  << rang::style::reset
               << ssmin_lat.str() << rang::style::bold << " | "  << rang::style::reset
               << ssmax_lat.str() << ""
               << std::endl;
    // clang-format on
}

void
Statistics::show_timer(const float total_sec,
                       const uint32_t task_n_calls,
                       const size_t timer_n_elmts,
                       const std::string& timer_name,
                       const uint32_t timer_n_calls,
                       const std::chrono::nanoseconds timer_tot_duration,
                       const std::chrono::nanoseconds timer_min_duration,
                       const std::chrono::nanoseconds timer_max_duration,
                       std::ostream& stream)
{
    // clang-format off
    if (task_n_calls == 0 || timer_n_calls == 0)
        return;

    auto rn_elmts = (timer_n_elmts * task_n_calls) / timer_n_calls;
    auto rtot_dur = ((float)timer_tot_duration.count()) * 0.000000001f;
    auto rpercent = (rtot_dur / total_sec) * 100.f;
    auto ravg_thr = (float)(timer_n_calls * rn_elmts) / ((float)timer_tot_duration.count() * 0.001f);
    auto rmin_thr = (float)(1.f           * rn_elmts) / ((float)timer_max_duration.count() * 0.001f);
    auto rmax_thr = (float)(1.f           * rn_elmts) / ((float)timer_min_duration.count() * 0.001f);
    auto ravg_lat = (float)(timer_tot_duration.count() * 0.001f) / timer_n_calls;
    auto rmin_lat = (float)(timer_min_duration.count() * 0.001f);
    auto rmax_lat = (float)(timer_max_duration.count() * 0.001f);

#ifdef _WIN32
    auto P = 1;
#else
    auto P = 2;
#endif

    unsigned l1 = 99999999;
    float    l2 = 99999.99f;

    std::stringstream spaces, ssprocess, ssrep, ssorder, sssp, ssrn_calls, ssrtot_dur, ssrpercent;
    std::stringstream ssravg_thr, ssrmin_thr, ssrmax_thr;
    std::stringstream ssravg_lat, ssrmin_lat, ssrmax_lat;

    spaces     <<                                                                                          std::fixed  << std::setw(18) << "-";
    ssprocess  << std::setprecision(                         2) <<                                         std::fixed  << std::setw(17) << "-";
    ssrep      << std::setprecision(                         2) <<                                         std::fixed  << std::setw( 3) << "-";
    ssorder    << std::setprecision(                         2) <<                                         std::fixed  << std::setw( 5) << "-";
    sssp       << std::setprecision(                         2) <<                                         std::fixed  << std::setw( 7) << timer_name;
    ssrn_calls << std::setprecision(timer_n_calls > l1 ? P : 2) << (timer_n_calls > l1 ? std::scientific : std::fixed) << std::setw( 8) << timer_n_calls;
    ssrtot_dur << std::setprecision(rtot_dur      > l1 ? P : 2) << (rtot_dur      > l1 ? std::scientific : std::fixed) << std::setw( 8) << rtot_dur;
    ssrpercent << std::setprecision(                         2) <<                                         std::fixed  << std::setw( 6) << rpercent;
    ssravg_thr << std::setprecision(((ravg_thr > l1) || (ravg_thr < 1)) ? P : 2) << (((ravg_thr > l2) || (ravg_thr < 1)) ? std::scientific : std::fixed) << std::setw( 8) << ravg_thr;
    ssrmin_thr << std::setprecision(((rmin_thr > l1) || (rmin_thr < 1)) ? P : 2) << (((rmin_thr > l2) || (rmin_thr < 1)) ? std::scientific : std::fixed) << std::setw( 8) << rmin_thr;
    ssrmax_thr << std::setprecision(((rmax_thr > l1) || (rmax_thr < 1)) ? P : 2) << (((rmax_thr > l2) || (rmax_thr < 1)) ? std::scientific : std::fixed) << std::setw( 8) << rmax_thr;
    ssravg_lat << std::setprecision(((ravg_lat > l1) || (ravg_lat < 1)) ? P : 2) << (((ravg_lat > l2) || (ravg_lat < 1)) ? std::scientific : std::fixed) << std::setw( 8) << ravg_lat;
    ssrmin_lat << std::setprecision(((rmin_lat > l1) || (rmin_lat < 1)) ? P : 2) << (((rmin_lat > l2) || (rmin_lat < 1)) ? std::scientific : std::fixed) << std::setw( 8) << rmin_lat;
    ssrmax_lat << std::setprecision(((rmax_lat > l1) || (rmax_lat < 1)) ? P : 2) << (((rmax_lat > l2) || (rmax_lat < 1)) ? std::scientific : std::fixed) << std::setw( 8) << rmax_lat;

    stream << "# ";
    stream << spaces.str()                                                  << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssprocess .str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssrep     .str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssorder   .str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << sssp      .str() << rang::style::reset << rang::style::bold << " || " << rang::style::reset
           << rang::style::italic << ssrn_calls.str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssrtot_dur.str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssrpercent.str() << rang::style::reset << rang::style::bold << " || " << rang::style::reset
           << rang::style::italic << ssravg_thr.str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssrmin_thr.str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssrmax_thr.str() << rang::style::reset << rang::style::bold << " || " << rang::style::reset
           << rang::style::italic << ssravg_lat.str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssrmin_lat.str() << rang::style::reset << rang::style::bold << " | "  << rang::style::reset
           << rang::style::italic << ssrmax_lat.str() << rang::style::reset << ""
           << std::endl;
    // clang-format on
}

template<class MODULE_OR_TASK>
void
Statistics::show(std::vector<MODULE_OR_TASK*> modules_or_tasks,
                 const bool ordered,
                 const bool display_thr,
                 std::ostream& stream)
{
    std::stringstream message;
    message << "The 'Statistics::show' method expect a 'std::vector' of 'module::Module' or 'runtime::Task'.";
    throw tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
}

namespace spu
{
namespace tools
{

template<>
void
Statistics::show<module::Module>(std::vector<module::Module*> modules,
                                 const bool ordered,
                                 const bool display_thr,
                                 std::ostream& stream)
{
    Statistics::show_modules<module::Module>(modules, ordered, display_thr, stream);
}

template<>
void
Statistics::show<const module::Module>(std::vector<const module::Module*> modules,
                                       const bool ordered,
                                       const bool display_thr,
                                       std::ostream& stream)
{
    Statistics::show_modules<const module::Module>(modules, ordered, display_thr, stream);
}

template<>
void
Statistics::show<runtime::Task>(std::vector<runtime::Task*> tasks,
                                const bool ordered,
                                const bool display_thr,
                                std::ostream& stream)
{
    Statistics::show_tasks<runtime::Task>(tasks, ordered, display_thr, stream);
}

template<>
void
Statistics::show<const runtime::Task>(std::vector<const runtime::Task*> tasks,
                                      const bool ordered,
                                      const bool display_thr,
                                      std::ostream& stream)
{
    Statistics::show_tasks<const runtime::Task>(tasks, ordered, display_thr, stream);
}

}
}

template<class MODULE>
void
Statistics::show_modules(std::vector<MODULE*> modules, const bool ordered, const bool display_thr, std::ostream& stream)
{
    std::vector<const runtime::Task*> tasks;
    for (auto& m : modules)
        if (m != nullptr)
            for (auto& t : m->tasks)
                if (t->get_n_calls()) tasks.push_back(t.get());

    Statistics::show_tasks(tasks, ordered, display_thr, stream);
}

template<class TASK>
void
Statistics::show_tasks(std::vector<TASK*> tasks, const bool ordered, const bool display_thr, std::ostream& stream)
{
    for (size_t t = 0; t < tasks.size(); t++)
        if (tasks[t] == nullptr) tasks.erase(tasks.begin() + t);

    std::map<const spu::runtime::Task*, size_t> tasks_order;
    for (size_t t = 0; t < tasks.size(); t++)
        tasks_order[(const spu::runtime::Task*)tasks[t]] = t;

    if (ordered)
    {
        std::sort(tasks.begin(),
                  tasks.end(),
                  [](const runtime::Task* t1, const runtime::Task* t2)
                  { return t1->get_duration_total() > t2->get_duration_total(); });
    }

    auto ttask_tot_duration = std::chrono::nanoseconds(0);
    auto ttask_min_duration = std::chrono::nanoseconds(0);
    auto ttask_max_duration = std::chrono::nanoseconds(0);

    for (auto* t : tasks)
        ttask_tot_duration += t->get_duration_total();
    auto total_sec = ((float)ttask_tot_duration.count()) * 0.000000001f;

    if (ttask_tot_duration.count())
    {
        Statistics::show_header(display_thr, stream);

        size_t ttask_n_elmts = 0;
        uint32_t ttask_n_calls = 0;

        auto is_first = true;
        for (auto* t : tasks)
        {
            auto task_n_elmts = t->sockets.end()[-2]->get_n_elmts();
            auto task_n_calls = t->get_n_calls();

            if (is_first)
            {
                if (task_n_calls)
                {
                    ttask_n_elmts = task_n_elmts;
                    ttask_n_calls = task_n_calls;
                    is_first = false;
                }
            }
            else
            {
                ttask_n_elmts = task_n_elmts ? std::min(ttask_n_elmts, task_n_elmts) : ttask_n_elmts;
                ttask_n_calls = task_n_calls ? std::min(ttask_n_calls, task_n_calls) : ttask_n_calls;
            }
        }

        bool all_replicable = true;
        for (auto* t : tasks)
        {
            auto module_name = t->get_module().get_custom_name().empty() ? t->get_module().get_short_name()
                                                                         : t->get_module().get_custom_name();
            auto task_n_elmts = t->sockets.end()[-2]->get_n_elmts();
            auto task_name = t->get_name();
            bool task_replicability = t->is_replicable();
            if (!task_replicability) all_replicable = false;
            auto task_n_calls = t->get_n_calls();
            auto task_tot_duration = t->get_duration_total();
            auto task_min_duration = t->get_duration_min();
            auto task_max_duration = t->get_duration_max();

            ttask_min_duration += (task_min_duration * task_n_calls) / ttask_n_calls;
            ttask_max_duration += (task_max_duration * task_n_calls) / ttask_n_calls;

            Statistics::show_task(total_sec,
                                  module_name,
                                  task_name,
                                  task_replicability,
                                  tasks_order[t],
                                  task_n_elmts,
                                  task_n_calls,
                                  task_tot_duration,
                                  task_min_duration,
                                  task_max_duration,
                                  display_thr,
                                  stream);

            auto task_total_sec = ((float)task_tot_duration.count()) * 0.000000001f;

            auto timers_name = t->get_timers_name();
            auto timers_n_elmts = task_n_elmts;
            auto timers_n_calls = t->get_timers_n_calls();
            auto timers_tot_duration = t->get_timers_total();
            auto timers_min_duration = t->get_timers_min();
            auto timers_max_duration = t->get_timers_max();

            for (size_t i = 0; i < timers_name.size(); i++)
            {
                Statistics::show_timer(task_total_sec,
                                       task_n_calls,
                                       timers_n_elmts,
                                       timers_name[i],
                                       timers_n_calls[i],
                                       timers_tot_duration[i],
                                       timers_min_duration[i],
                                       timers_max_duration[i],
                                       stream);
            }
        }
        Statistics::separation2(display_thr, stream);

        Statistics::show_task(total_sec,
                              "TOTAL",
                              "*",
                              all_replicable,
                              -1,
                              ttask_n_elmts,
                              ttask_n_calls,
                              ttask_tot_duration,
                              ttask_min_duration,
                              ttask_max_duration,
                              display_thr,
                              stream);
    }
    else
    {
        stream << rang::tag::comment << rang::tag::info
               << "Statistics are unavailable. Did you enable the statistics in the tasks?" << std::endl;
    }
}

template<class MODULE_OR_TASK>
void
Statistics::show(std::vector<std::vector<MODULE_OR_TASK*>> modules_or_tasks,
                 const bool ordered,
                 const bool display_thr,
                 std::ostream& stream)
{
    std::stringstream message;
    message << "The 'Statistics::show' method expect a 'std::vector' of 'std::vector' of 'module::Module' or "
            << "'runtime::Task'.";
    throw tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
}

namespace spu
{
namespace tools
{

template<>
void
Statistics::show<module::Module>(std::vector<std::vector<module::Module*>> modules,
                                 const bool ordered,
                                 const bool display_thr,
                                 std::ostream& stream)
{
    Statistics::show_modules<module::Module>(modules, ordered, display_thr, stream);
}

template<>
void
Statistics::show<const module::Module>(std::vector<std::vector<const module::Module*>> modules,
                                       const bool ordered,
                                       const bool display_thr,
                                       std::ostream& stream)
{
    Statistics::show_modules<const module::Module>(modules, ordered, display_thr, stream);
}

template<>
void
Statistics::show<runtime::Task>(std::vector<std::vector<runtime::Task*>> tasks,
                                const bool ordered,
                                const bool display_thr,
                                std::ostream& stream)
{
    Statistics::show_tasks<runtime::Task>(tasks, ordered, display_thr, stream);
}

template<>
void
Statistics::show<const runtime::Task>(std::vector<std::vector<const runtime::Task*>> tasks,
                                      const bool ordered,
                                      const bool display_thr,
                                      std::ostream& stream)
{
    Statistics::show_tasks<const runtime::Task>(tasks, ordered, display_thr, stream);
}

}
}

template<class MODULE>
void
Statistics::show_modules(std::vector<std::vector<MODULE*>> modules,
                         const bool ordered,
                         const bool display_thr,
                         std::ostream& stream)
{
    std::vector<std::vector<const runtime::Task*>> tasks;
    for (auto& vm : modules)
        if (vm.size() > 0 && vm[0] != nullptr)
        {
            auto& tasks0 = vm[0]->tasks;
            for (size_t t = 0; t < tasks0.size(); t++)
            {
                std::vector<const runtime::Task*> tsk;
                for (auto& m : vm)
                    tsk.push_back(m->tasks[t].get());
                tasks.push_back(tsk);
            }
        }

    Statistics::show_tasks(tasks, ordered, display_thr, stream);
}

template<class TASK>
void
Statistics::show_tasks(std::vector<std::vector<TASK*>> tasks,
                       const bool ordered,
                       const bool display_thr,
                       std::ostream& stream)
{
    using namespace std::chrono;

    if (tasks.size())
        for (size_t t = 0; t < tasks[0].size(); t++)
            if (tasks[0][t] == nullptr)
                for (size_t i = 0; i < tasks.size(); i++)
                    tasks[i].erase(tasks[i].begin() + t);

    std::map<const spu::runtime::Task*, size_t> tasks_order;
    if (tasks.size())
        for (size_t t = 0; t < tasks.size(); t++)
            tasks_order[(const spu::runtime::Task*)tasks[t][0]] = t;

    if (ordered)
    {
        std::sort(tasks.begin(),
                  tasks.end(),
                  [](const std::vector<TASK*>& t1, const std::vector<TASK*>& t2)
                  {
                      auto total1 = nanoseconds(0);
                      auto total2 = nanoseconds(0);
                      for (auto* t : t1)
                          total1 += t->get_duration_total();
                      for (auto* t : t2)
                          total2 += t->get_duration_total();
                      return total1 > total2;
                  });
    }

    auto ttask_tot_duration = nanoseconds(0);
    auto ttask_min_duration = nanoseconds(0);
    auto ttask_max_duration = nanoseconds(0);

    for (auto& vt : tasks)
        for (auto* t : vt)
            ttask_tot_duration += t->get_duration_total();
    auto total_sec = ((float)ttask_tot_duration.count()) * 0.000000001f;

    if (ttask_tot_duration.count())
    {
        Statistics::show_header(display_thr, stream);

        size_t ttask_n_elmts = 0;
        auto ttask_n_calls = 0;

        auto is_first = true;
        for (auto& vt : tasks)
        {
            auto task_n_elmts = vt[0]->sockets.size() >= 2 ? vt[0]->sockets.end()[-2]->get_n_elmts() : 0;
            auto task_n_calls = 0;

            for (auto* t : vt)
                task_n_calls += t->get_n_calls();

            if (is_first)
            {
                if (task_n_calls)
                {
                    ttask_n_elmts = task_n_elmts;
                    ttask_n_calls = task_n_calls;
                    is_first = false;
                }
            }
            else
            {
                ttask_n_elmts = task_n_elmts ? std::min(ttask_n_elmts, task_n_elmts) : ttask_n_elmts;
                ttask_n_calls = task_n_calls ? std::min(ttask_n_calls, task_n_calls) : ttask_n_calls;
            }
        }

        bool all_replicable = true;
        for (auto& vt : tasks)
        {
            auto module_name = vt[0]->get_module().get_custom_name().empty() ? vt[0]->get_module().get_short_name()
                                                                             : vt[0]->get_module().get_custom_name();
            auto task_n_elmts = vt[0]->sockets.size() >= 2 ? vt[0]->sockets.end()[-2]->get_n_elmts() : 0;
            auto task_name = vt[0]->get_name();
            bool task_replicability = vt[0]->is_replicable();
            if (!task_replicability) all_replicable = false;
            auto task_n_calls = 0;
            auto task_tot_duration = nanoseconds(0);
            auto task_min_duration = ttask_tot_duration;
            auto task_max_duration = nanoseconds(0);

            for (auto* t : vt)
            {
                task_n_calls += t->get_n_calls();
                task_tot_duration += t->get_duration_total();
                task_min_duration = std::min(task_min_duration, t->get_duration_min());
                task_max_duration = std::max(task_max_duration, t->get_duration_max());
            }

            ttask_min_duration += (task_min_duration * task_n_calls) / ttask_n_calls;
            ttask_max_duration += (task_max_duration * task_n_calls) / ttask_n_calls;

            Statistics::show_task(total_sec,
                                  module_name,
                                  task_name,
                                  task_replicability,
                                  tasks_order[vt[0]],
                                  task_n_elmts,
                                  task_n_calls,
                                  task_tot_duration,
                                  task_min_duration,
                                  task_max_duration,
                                  display_thr,
                                  stream);

            auto task_total_sec = ((float)task_tot_duration.count()) * 0.000000001f;

            auto timers_name = vt[0]->get_timers_name();
            auto timers_n_elmts = task_n_elmts;
            auto timers_n_calls = std::vector<uint32_t>(timers_name.size(), 0);
            auto timers_tot_duration = std::vector<nanoseconds>(timers_name.size(), nanoseconds(0));
            auto timers_min_duration = std::vector<nanoseconds>(timers_name.size(), ttask_tot_duration);
            auto timers_max_duration = std::vector<nanoseconds>(timers_name.size(), nanoseconds(0));

            for (size_t tn = 0; tn < vt[0]->get_timers_name().size(); tn++)
            {
                for (auto* t : vt)
                {
                    timers_n_calls[tn] += t->get_timers_n_calls()[tn];
                    timers_tot_duration[tn] += t->get_timers_total()[tn];
                    timers_min_duration[tn] = std::min(task_min_duration, t->get_timers_min()[tn]);
                    timers_max_duration[tn] = std::max(task_max_duration, t->get_timers_max()[tn]);
                }

                Statistics::show_timer(task_total_sec,
                                       task_n_calls,
                                       timers_n_elmts,
                                       timers_name[tn],
                                       timers_n_calls[tn],
                                       timers_tot_duration[tn],
                                       timers_min_duration[tn],
                                       timers_max_duration[tn],
                                       stream);
            }
        }
        Statistics::separation2(display_thr, stream);

        Statistics::show_task(total_sec,
                              "TOTAL",
                              "*",
                              all_replicable,
                              -1,
                              ttask_n_elmts,
                              ttask_n_calls,
                              ttask_tot_duration,
                              ttask_min_duration,
                              ttask_max_duration,
                              display_thr,
                              stream);
    }
    else
    {
        stream << rang::tag::comment << rang::tag::info
               << "Statistics are unavailable. Did you enable the statistics in the tasks?" << std::endl;
    }
}

namespace
{
void
rt_separation1(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    if (display_thr)
        stream << "# " << rang::style::bold << "-----------------------------------------------------||---------------------||--------------------------------------------------------------------------------------------------||------------------------------------------------------" << rang::style::reset << std::endl;
    else
        stream << "# " << rang::style::bold << "-----------------------------------------------------||---------------------||--------------------------------------------------------------------------------------------------" << rang::style::reset << std::endl;
    // clang-format on
}

void
rt_separation2(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    if (display_thr)
        stream << "# " << rang::style::bold << "-------------------|-------------------|-----|-------||----------|----------||----------|----------|----------|----------|----------|----------|----------|----------|----------||----------|----------|----------|----------|----------" << rang::style::reset << std::endl;
    else
        stream << "# " << rang::style::bold << "-------------------|-------------------|-----|-------||----------|----------||----------|----------|----------|----------|----------|----------|----------|----------|----------" << rang::style::reset << std::endl;
    // clang-format on
}

void
rt_show_columns_header(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    rt_separation2(display_thr, stream);
    if (display_thr)
    {
        stream << "# " << rang::style::bold << "       MODULE NAME |         TASK NAME | REP | ORDER ||    CALLS |     TIME ||  AVERAGE |  STD DEV |  MINIMUM |      P50 |      P90 |      P99 |    P99.9 |   P99.99 |  MAXIMUM ||  AVERAGE |   1% LOW |  0.1% LW | 0.01% LW |  MINIMUM " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "                   |                   |     |       ||          |      (s) ||     (us) |     (us) |     (us) |     (us) |     (us) |     (us) |     (us) |     (us) |     (us) ||   (Mb/s) |   (Mb/s) |   (Mb/s) |   (Mb/s) |   (Mb/s) " << rang::style::reset << std::endl;
    }
    else
    {
        stream << "# " << rang::style::bold << "       MODULE NAME |         TASK NAME | REP | ORDER ||    CALLS |     TIME ||  AVERAGE |  STD DEV |  MINIMUM |      P50 |      P90 |      P99 |    P99.9 |   P99.99 |  MAXIMUM " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "                   |                   |     |       ||          |      (s) ||     (us) |     (us) |     (us) |     (us) |     (us) |     (us) |     (us) |     (us) |     (us) " << rang::style::reset << std::endl;
    }
    rt_separation2(display_thr, stream);
    // clang-format on
}

void
rt_show_sequence_header(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    rt_separation1(display_thr, stream);
    if (display_thr)
    {
        stream << "# " << rang::style::bold << "            Real-Time (RT) statistics for            ||   Basic statistics  ||                            Measured frame latency & jitter distribution                          ||             Measured throughput & lows               " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "                tasks & the sequence                 ||   on the sequence   ||                                          (HDR histogram)                                         ||            considering the last socket               " << rang::style::reset << std::endl;
    }
    else
    {
        stream << "# " << rang::style::bold << "            Real-Time (RT) statistics for            ||   Basic statistics  ||                            Measured frame latency & jitter distribution                          " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "                tasks & the sequence                 ||   on the sequence   ||                                          (HDR histogram)                                         " << rang::style::reset << std::endl;
    }
    rt_separation1(display_thr, stream);
    rt_show_columns_header(display_thr, stream);
    // clang-format on
}

void
rt_show_pipeline_stages_header(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    rt_separation1(display_thr, stream);
    if (display_thr)
    {
        stream << "# " << rang::style::bold << "            Real-Time (RT) statistics for            ||   Basic statistics  ||                            Measured stage latency & jitter distribution                          ||             Measured throughput & lows               " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "         pipeline stages (excluding wait time)       ||     on the stage    ||                                          (HDR histogram)                                         ||            considering the last socket               " << rang::style::reset << std::endl;
    }
    else
    {
        stream << "# " << rang::style::bold << "            Real-Time (RT) statistics for            ||   Basic statistics  ||                            Measured stage latency & jitter distribution                          " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "         pipeline stages (excluding wait time)       ||     on the stage    ||                                          (HDR histogram)                                         " << rang::style::reset << std::endl;
    }
    rt_separation1(display_thr, stream);
    rt_show_columns_header(display_thr, stream);
    // clang-format on
}

void
rt_show_pipeline_e2e_header(const bool display_thr, std::ostream& stream)
{
    // clang-format off
    rt_separation1(display_thr, stream);
    if (display_thr)
    {
        stream << "# " << rang::style::bold << "            Real-Time (RT) statistics for            ||   Basic statistics  ||                         Measured end-to-end latency & jitter distribution                        ||             Measured throughput & lows               " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "       full pipeline (including wait & FIFO time)    ||   on the pipeline   ||                                          (HDR histogram)                                         ||            considering the last socket               " << rang::style::reset << std::endl;
    }
    else
    {
        stream << "# " << rang::style::bold << "            Real-Time (RT) statistics for            ||   Basic statistics  ||                         Measured end-to-end latency & jitter distribution                        " << rang::style::reset << std::endl;
        stream << "# " << rang::style::bold << "       full pipeline (including wait & FIFO time)    ||   on the pipeline   ||                                          (HDR histogram)                                         " << rang::style::reset << std::endl;
    }
    rt_separation1(display_thr, stream);
    rt_separation2(display_thr, stream);
    // clang-format on
}

std::string
rt_format_num(const double val)
{
#ifdef _WIN32
    const int P = 1;
#else
    const int P = 2;
#endif
    const double l1 = 99999999.0;
    const double l2 = 99999.99;
    std::stringstream ss;
    if (val <= 0.0)
    {
        ss << std::fixed << std::setprecision(2) << std::setw(8) << 0.0;
    }
    else
    {
        ss << std::setprecision(((val > l1) || (val < 1.0)) ? P : 2)
           << (((val > l2) || (val < 1.0)) ? std::scientific : std::fixed) << std::setw(8) << val;
    }
    return ss.str();
}

void
rt_show_empty_task_row(const std::string& module_name,
                       const std::string& task_name,
                       const bool task_replicability,
                       const int task_order,
                       const bool display_thr,
                       std::ostream& stream)
{
    const std::string trimmed_module_name = module_name.substr(0, 18);
    const bool module_name_is_trimmed = module_name.length() > 18;
    const std::string trimmed_task_name = task_name.substr(0, 17);
    const bool task_name_is_trimmed = task_name.length() > 17;

    std::stringstream ssmodule, ssprocess, ssrep, ssorder, ssdash;
    ssmodule << std::setw(18) << trimmed_module_name;
    ssprocess << std::setw(17) << trimmed_task_name;
    ssrep << std::setw(3) << (task_replicability ? "yes" : "no");
    if (task_order >= 0)
        ssorder << std::setw(5) << task_order;
    else
        ssorder << std::setw(5) << "*";
    ssdash << std::setw(8) << "-";

    stream << "# " << ssmodule.str() << rang::style::bold << (module_name_is_trimmed ? "*" : " ") << "| "
           << rang::style::reset << ssprocess.str() << rang::style::bold << (task_name_is_trimmed ? "*" : " ") << "| "
           << rang::style::reset << ssrep.str() << rang::style::bold << " | " << rang::style::reset << ssorder.str()
           << rang::style::bold << " || " << rang::style::reset << ssdash.str() << rang::style::bold << " | "
           << rang::style::reset << ssdash.str() << rang::style::bold << " || " << rang::style::reset << ssdash.str()
           << rang::style::bold << " | " << rang::style::reset << ssdash.str() << rang::style::bold << " | "
           << rang::style::reset << ssdash.str() << rang::style::bold << " | " << rang::style::reset << ssdash.str()
           << rang::style::bold << " | " << rang::style::reset << ssdash.str() << rang::style::bold << " | "
           << rang::style::reset << ssdash.str() << rang::style::bold << " | " << rang::style::reset << ssdash.str()
           << rang::style::bold << " | " << rang::style::reset << ssdash.str() << rang::style::bold << " | "
           << rang::style::reset << ssdash.str();

    if (display_thr)
    {
        stream << rang::style::bold << " || " << rang::style::reset << ssdash.str() << rang::style::bold << " | "
               << rang::style::reset << ssdash.str() << rang::style::bold << " | " << rang::style::reset << ssdash.str()
               << rang::style::bold << " | " << rang::style::reset << ssdash.str() << rang::style::bold << " | "
               << rang::style::reset << ssdash.str();
    }
    stream << std::endl;
}

void
rt_show_stats_row(const std::string& label_left,
                  const std::string& label_task,
                  const bool all_replicable,
                  const tools::Rt_stats& stats,
                  const double ns_per_tick,
                  const size_t n_elmts,
                  const double lat_divisor,
                  const bool show_thr_values,
                  const bool display_thr,
                  const bool italic,
                  std::ostream& stream,
                  const tools::Rt_stats* thr_stats_override = nullptr,
                  const size_t n_threads = 1)
{
    if (stats.n_calls == 0) return;

    const tools::Rt_stats& t_stats =
      (thr_stats_override && thr_stats_override->n_calls > 0) ? *thr_stats_override : stats;
    const double n_thr_d = (double)std::max<size_t>(1, n_threads);
    const double us_per_tick = (ns_per_tick * 1e-3) / lat_divisor;
    const double tot_sec = (((double)t_stats.tot_ticks * ns_per_tick) * 1e-9) / n_thr_d;

    const double avg_lat = stats.get_avg_ticks() * us_per_tick;
    const double std_lat = stats.get_std_ticks() * us_per_tick;
    const double min_lat = (double)stats.min_ticks * us_per_tick;
    const double p50_lat = stats.get_percentile_ticks(50.0) * us_per_tick;
    const double p90_lat = stats.get_percentile_ticks(90.0) * us_per_tick;
    const double p99_lat = stats.get_percentile_ticks(99.0) * us_per_tick;
    const double p999_lat = stats.get_percentile_ticks(99.9) * us_per_tick;
    const double p9999_lat = stats.get_percentile_ticks(99.99) * us_per_tick;
    const double max_lat = (double)stats.max_ticks * us_per_tick;

    const std::string trimmed_left = label_left.substr(0, 18);
    const bool left_is_trimmed = label_left.length() > 18;
    const std::string trimmed_task = label_task.substr(0, 17);
    const bool task_is_trimmed = label_task.length() > 17;

    std::stringstream ssleft, sstask, ssrep, ssorder, sscalls;
    ssleft << std::setw(18) << trimmed_left;
    sstask << std::setw(17) << trimmed_task;
    ssrep << std::setw(3) << (italic ? "-" : (all_replicable ? "yes" : "no"));
    ssorder << std::setw(5) << (italic ? "-" : "*");
    if (italic)
    {
        sscalls << std::setw(8) << "-";
    }
    else if (stats.n_calls > 99999999ULL)
    {
#ifdef _WIN32
        sscalls << std::scientific << std::setprecision(1) << std::setw(8) << (double)stats.n_calls;
#else
        sscalls << std::scientific << std::setprecision(2) << std::setw(8) << (double)stats.n_calls;
#endif
    }
    else
    {
        sscalls << std::fixed << std::setw(8) << stats.n_calls;
    }

    const std::string sstot = italic ? std::string("       -") : rt_format_num(tot_sec);

    auto wrap_style = [italic](const std::string& s) -> std::string
    {
        if (!italic) return s;
        std::stringstream ss;
        ss << rang::style::italic << s << rang::style::reset;
        return ss.str();
    };

    stream << "# " << wrap_style(ssleft.str()) << rang::style::bold << (left_is_trimmed ? "*" : " ") << "| "
           << rang::style::reset << wrap_style(sstask.str()) << rang::style::bold << (task_is_trimmed ? "*" : " ")
           << "| " << rang::style::reset << wrap_style(ssrep.str()) << rang::style::bold << " | " << rang::style::reset
           << wrap_style(ssorder.str()) << rang::style::bold << " || " << rang::style::reset
           << wrap_style(sscalls.str()) << rang::style::bold << " | " << rang::style::reset << wrap_style(sstot)
           << rang::style::bold << " || " << rang::style::reset << wrap_style(rt_format_num(avg_lat))
           << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(std_lat))
           << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(min_lat))
           << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(p50_lat))
           << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(p90_lat))
           << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(p99_lat))
           << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(p999_lat))
           << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(p9999_lat))
           << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(max_lat));

    if (display_thr)
    {
        if (show_thr_values && n_elmts > 0)
        {
            const double t_avg = t_stats.get_avg_ticks() * us_per_tick;
            const double t_p99 = t_stats.get_percentile_ticks(99.0) * us_per_tick;
            const double t_p999 = t_stats.get_percentile_ticks(99.9) * us_per_tick;
            const double t_p9999 = t_stats.get_percentile_ticks(99.99) * us_per_tick;
            const double t_max = (double)t_stats.max_ticks * us_per_tick;
            const double elmts_scaled = (double)n_elmts * n_thr_d;

            const double avg_thr = (t_avg > 0.0) ? (elmts_scaled / t_avg) : 0.0;
            const double low1_thr = (t_p99 > 0.0) ? (elmts_scaled / t_p99) : 0.0;
            const double low01_thr = (t_p999 > 0.0) ? (elmts_scaled / t_p999) : 0.0;
            const double low001_thr = (t_p9999 > 0.0) ? (elmts_scaled / t_p9999) : 0.0;
            const double min_thr = (t_max > 0.0) ? (elmts_scaled / t_max) : 0.0;

            stream << rang::style::bold << " || " << rang::style::reset << wrap_style(rt_format_num(avg_thr))
                   << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(low1_thr))
                   << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(low01_thr))
                   << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(low001_thr))
                   << rang::style::bold << " | " << rang::style::reset << wrap_style(rt_format_num(min_thr));
        }
        else
        {
            const std::string ssdash = wrap_style("       -");
            stream << rang::style::bold << " || " << rang::style::reset << ssdash << rang::style::bold << " | "
                   << rang::style::reset << ssdash << rang::style::bold << " | " << rang::style::reset << ssdash
                   << rang::style::bold << " | " << rang::style::reset << ssdash << rang::style::bold << " | "
                   << rang::style::reset << ssdash;
        }
    }
    stream << std::endl;
}

size_t
rt_find_stage_n_elmts(const std::vector<std::vector<runtime::Task*>>& tasks_per_types)
{
    size_t n_elmts = 0;
    for (size_t i = tasks_per_types.size(); i > 0; i--)
    {
        const auto& vt = tasks_per_types[i - 1];
        if (!vt.empty() && vt[0] != nullptr && vt[0]->sockets.size() >= 2)
        {
            const size_t e = vt[0]->sockets.end()[-2]->get_n_elmts();
            if (e > 0)
            {
                n_elmts = e;
                break;
            }
        }
    }
    return n_elmts;
}
} // anonymous namespace

void
Statistics::show_rt(const runtime::Sequence& sequence, const bool display_thr, std::ostream& stream)
{
    tools::Rt_stats merged;
    for (const auto& s : sequence.get_rt_stats_per_thread())
        merged.merge(s);

    if (merged.n_calls == 0)
    {
        stream << rang::tag::comment << rang::tag::info
               << "Real-Time statistics are unavailable. Did you call 'sequence.set_rt_stats(true)' before 'exec()'?"
               << std::endl;
        return;
    }

    rt_show_sequence_header(display_thr, stream);

    const auto tasks_per_types = sequence.get_tasks_per_types();
    bool all_replicable = true;
    for (size_t i = 0; i < tasks_per_types.size(); i++)
    {
        const auto& vt = tasks_per_types[i];
        if (vt.empty() || vt[0] == nullptr) continue;
        const auto* t = vt[0];
        const auto module_name = t->get_module().get_custom_name().empty() ? t->get_module().get_short_name()
                                                                           : t->get_module().get_custom_name();
        const bool rep = t->is_replicable();
        if (!rep) all_replicable = false;
        rt_show_empty_task_row(module_name, t->get_name(), rep, (int)i, display_thr, stream);
    }

    rt_separation2(display_thr, stream);

    const size_t n_tasks = tasks_per_types.size();
    const size_t n_elmts = rt_find_stage_n_elmts(tasks_per_types);
    const double ns_per_tick = sequence.get_rt_ns_per_tick();

    rt_show_stats_row("SEQUENCE",
                      "*" + (n_tasks > 0 ? (" (" + std::to_string(n_tasks) + " tasks)") : std::string("")),
                      all_replicable,
                      merged,
                      ns_per_tick,
                      n_elmts,
                      1.0,
                      true,
                      display_thr,
                      false,
                      stream,
                      nullptr,
                      sequence.get_n_threads());

    if (n_tasks > 1)
    {
        rt_show_stats_row("  / " + std::to_string(n_tasks) + " tasks",
                          "avg / task",
                          all_replicable,
                          merged,
                          ns_per_tick,
                          n_elmts,
                          (double)n_tasks,
                          false,
                          display_thr,
                          true,
                          stream,
                          nullptr,
                          sequence.get_n_threads());
    }
}

void
Statistics::show_rt(const runtime::Pipeline& pipeline, const bool display_thr, std::ostream& stream)
{
    const auto stages = pipeline.get_stages();
    if (stages.empty()) return;

    bool any_calls = false;
    for (const auto* stage : stages)
        for (const auto& s : stage->get_rt_stats_per_thread())
            if (s.n_calls > 0) any_calls = true;

    if (!any_calls)
    {
        stream << rang::tag::comment << rang::tag::info
               << "Real-Time statistics are unavailable. Did you call 'pipeline.set_rt_stats(true)' before 'exec()'?"
               << std::endl;
        return;
    }

    rt_show_pipeline_stages_header(display_thr, stream);

    size_t global_order = 0;
    bool all_pipeline_replicable = true;

    for (size_t sta = 0; sta < stages.size(); sta++)
    {
        const auto* stage = stages[sta];
        const auto tasks_per_types = stage->get_tasks_per_types();
        bool stage_replicable = true;

        for (size_t i = 0; i < tasks_per_types.size(); i++)
        {
            const auto& vt = tasks_per_types[i];
            if (vt.empty() || vt[0] == nullptr) continue;
            const auto* t = vt[0];
            const auto module_name = t->get_module().get_custom_name().empty() ? t->get_module().get_short_name()
                                                                               : t->get_module().get_custom_name();
            const bool rep = t->is_replicable();
            if (!rep)
            {
                stage_replicable = false;
                all_pipeline_replicable = false;
            }
            rt_show_empty_task_row(module_name, t->get_name(), rep, (int)global_order++, display_thr, stream);
        }

        tools::Rt_stats stage_merged;
        for (const auto& s : stage->get_rt_stats_per_thread())
            stage_merged.merge(s);

        rt_separation2(display_thr, stream);

        const size_t n_tasks = tasks_per_types.size();
        const size_t n_elmts = rt_find_stage_n_elmts(tasks_per_types);
        const double ns_per_tick = stage->get_rt_ns_per_tick();

        rt_show_stats_row("STAGE " + std::to_string(sta) + " (" + std::to_string(stage->get_n_threads()) + "t)",
                          "*" + (n_tasks > 0 ? (" (" + std::to_string(n_tasks) + " tasks)") : std::string("")),
                          stage_replicable,
                          stage_merged,
                          ns_per_tick,
                          n_elmts,
                          1.0,
                          true,
                          display_thr,
                          false,
                          stream,
                          nullptr,
                          stage->get_n_threads());

        if (n_tasks > 1)
        {
            rt_show_stats_row("  / " + std::to_string(n_tasks) + " tasks",
                              "avg / task",
                              stage_replicable,
                              stage_merged,
                              ns_per_tick,
                              n_elmts,
                              (double)n_tasks,
                              false,
                              display_thr,
                              true,
                              stream,
                              nullptr,
                              stage->get_n_threads());
        }

        if (sta + 1 < stages.size())
        {
            rt_separation2(display_thr, stream);
            rt_separation2(display_thr, stream);
        }
    }

    tools::Rt_stats e2e_merged;
    for (const auto& s : stages.back()->get_pipeline_e2e_rt_stats_per_thread())
        e2e_merged.merge(s);

    tools::Rt_stats inter_merged;
    for (const auto& s : stages.back()->get_pipeline_inter_rt_stats_per_thread())
        inter_merged.merge(s);

    const auto last_tasks = stages.back()->get_tasks_per_types();
    const size_t e2e_n_elmts = rt_find_stage_n_elmts(last_tasks);
    const double ns_per_tick = stages.back()->get_rt_ns_per_tick();
    const size_t last_n_threads = stages.back()->get_n_threads();

    rt_show_pipeline_e2e_header(display_thr, stream);

    rt_show_stats_row("PIPELINE E2E",
                      "end-to-end",
                      all_pipeline_replicable,
                      e2e_merged,
                      ns_per_tick,
                      e2e_n_elmts,
                      1.0,
                      true,
                      display_thr,
                      false,
                      stream,
                      &inter_merged,
                      last_n_threads);
}

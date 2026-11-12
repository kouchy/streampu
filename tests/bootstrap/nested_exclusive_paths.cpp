#include <atomic>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <streampu.hpp>
using namespace spu;
using namespace spu::runtime;

// Exclusive paths that contain exclusive paths (a Switcher inside a path of another Switcher):
//
//                    +-- path 0: Switcher A (path 0: +1 | path 1: +2), then +1 ----+
//   initialize --> commute -- path 1: +1 -------------------------------------------> select --> finalize
//                    +-- path 2: Switcher B (path 0: +3 | path 1: +4) --------------+
//
// The path 0 is not the last path of the outer Switcher, and the path 2 ends with the 'select' of B directly bound
// to the outer 'select'.
int
main(int argc, char** argv)
{
    tools::Signal_handler::init();

    option longopts[] = { { "n-threads", required_argument, NULL, 't' },
                          { "n-inter-frames", required_argument, NULL, 'f' },
                          { "sleep-time", required_argument, NULL, 's' },
                          { "data-length", required_argument, NULL, 'd' },
                          { "n-exec", required_argument, NULL, 'e' },
                          { "path", required_argument, NULL, 'a' },
                          { "inner-path", required_argument, NULL, 'i' },
                          { "dot-filepath", required_argument, NULL, 'o' },
                          { "copy-mode", no_argument, NULL, 'c' },
                          { "print-stats", no_argument, NULL, 'p' },
                          { "step-by-step", no_argument, NULL, 'b' },
                          { "debug", no_argument, NULL, 'g' },
                          { "cyclic-path", no_argument, NULL, 'y' },
                          { "task-autoalloc", no_argument, NULL, 'k' },
                          { "help", no_argument, NULL, 'h' },
                          { NULL, 0, NULL, 0 } };

    size_t n_threads = std::thread::hardware_concurrency();
    size_t n_inter_frames = 1;
    size_t sleep_time_us = 5;
    size_t data_length = 2048;
    size_t n_exec = 100000;
    size_t path = 0;
    size_t inner_path = 0;
    std::string dot_filepath;
    bool no_copy_mode = true;
    bool print_stats = false;
    bool step_by_step = false;
    bool debug = false;
    bool cyclic_path = false;
    bool task_autoalloc = false;

    while (1)
    {
        const int opt = getopt_long(argc, argv, "t:f:s:d:e:a:i:o:cpbgykh", longopts, 0);
        if (opt == -1) break;
        switch (opt)
        {
            case 't':
                n_threads = atoi(optarg);
                break;
            case 'f':
                n_inter_frames = atoi(optarg);
                break;
            case 's':
                sleep_time_us = atoi(optarg);
                break;
            case 'd':
                data_length = atoi(optarg);
                break;
            case 'e':
                n_exec = atoi(optarg);
                break;
            case 'a':
                path = atoi(optarg);
                break;
            case 'i':
                inner_path = atoi(optarg);
                break;
            case 'o':
                dot_filepath = std::string(optarg);
                break;
            case 'c':
                no_copy_mode = false;
                break;
            case 'p':
                print_stats = true;
                break;
            case 'b':
                step_by_step = true;
                break;
            case 'g':
                debug = true;
                break;
            case 'y':
                cyclic_path = true;
                break;
            case 'k':
                task_autoalloc = true;
                break;
            case 'h':
                std::cout << "usage: " << argv[0] << " [options]" << std::endl;
                std::cout << std::endl;
                std::cout << "  -t, --n-threads       "
                          << "Number of threads to run in parallel                                  "
                          << "[" << n_threads << "]" << std::endl;
                std::cout << "  -f, --n-inter-frames  "
                          << "Number of frames to process in one task                               "
                          << "[" << n_inter_frames << "]" << std::endl;
                std::cout << "  -s, --sleep-time      "
                          << "Sleep time duration in one task (microseconds)                        "
                          << "[" << sleep_time_us << "]" << std::endl;
                std::cout << "  -d, --data-length     "
                          << "Size of data to process in one task (in bytes)                        "
                          << "[" << data_length << "]" << std::endl;
                std::cout << "  -e, --n-exec          "
                          << "Number of sequence executions                                         "
                          << "[" << n_exec << "]" << std::endl;
                std::cout << "  -a, --path            "
                          << "Path to take in the outer switch (0, 1 or 2)                          "
                          << "[" << path << "]" << std::endl;
                std::cout << "  -i, --inner-path      "
                          << "Path to take in the inner switches (0 or 1)                           "
                          << "[" << inner_path << "]" << std::endl;
                std::cout << "  -o, --dot-filepath    "
                          << "Path to dot output file                                               "
                          << "[" << (dot_filepath.empty() ? "empty" : "\"" + dot_filepath + "\"") << "]" << std::endl;
                std::cout << "  -c, --copy-mode       "
                          << "Enable to copy data in sequence (performance will be reduced)         "
                          << "[" << (no_copy_mode ? "false" : "true") << "]" << std::endl;
                std::cout << "  -b, --step-by-step    "
                          << "Enable step-by-step sequence execution (performance will be reduced)  "
                          << "[" << (step_by_step ? "true" : "false") << "]" << std::endl;
                std::cout << "  -p, --print-stats     "
                          << "Enable to print per task statistics (performance will be reduced)     "
                          << "[" << (print_stats ? "true" : "false") << "]" << std::endl;
                std::cout << "  -g, --debug           "
                          << "Enable task debug mode (print socket data)                            "
                          << "[" << (debug ? "true" : "false") << "]" << std::endl;
                std::cout << "  -y, --cyclic-path     "
                          << "Enable cyclic selection of the outer path (`--path` is ignored)       "
                          << "[" << (cyclic_path ? "true" : "false") << "]" << std::endl;
                std::cout << "  -k, --task-autoalloc "
                          << "Enable task SOUT autoalloc mode                                      "
                          << "[" << (task_autoalloc ? "true" : "false") << "]" << std::endl;
                std::cout << "  -h, --help            "
                          << "This help                                                             "
                          << "[false]" << std::endl;
                exit(0);
                break;
            default:
                break;
        }
    }

    std::cout << "###########################################" << std::endl;
    std::cout << "# Micro-benchmark: Nested exclusive paths #" << std::endl;
    std::cout << "###########################################" << std::endl;
    std::cout << "#" << std::endl;

    std::cout << "# Command line arguments:" << std::endl;
    std::cout << "#   - n_threads      = " << n_threads << std::endl;
    std::cout << "#   - n_inter_frames = " << n_inter_frames << std::endl;
    std::cout << "#   - sleep_time_us  = " << sleep_time_us << std::endl;
    std::cout << "#   - data_length    = " << data_length << std::endl;
    std::cout << "#   - n_exec         = " << n_exec << std::endl;
    std::cout << "#   - path           = " << path << (cyclic_path ? " (unused)" : "") << std::endl;
    std::cout << "#   - inner_path     = " << inner_path << std::endl;
    std::cout << "#   - dot_filepath   = " << (dot_filepath.empty() ? "[empty]" : dot_filepath.c_str()) << std::endl;
    std::cout << "#   - no_copy_mode   = " << (no_copy_mode ? "true" : "false") << std::endl;
    std::cout << "#   - print_stats    = " << (print_stats ? "true" : "false") << std::endl;
    std::cout << "#   - step_by_step   = " << (step_by_step ? "true" : "false") << std::endl;
    std::cout << "#   - debug          = " << (debug ? "true" : "false") << std::endl;
    std::cout << "#   - cyclic_path    = " << (cyclic_path ? "true" : "false") << std::endl;
    std::cout << "#   - task_autoalloc = " << (task_autoalloc ? "true" : "false") << std::endl;
    std::cout << "#" << std::endl;

    tools::Buffer_allocator::set_task_autoalloc(task_autoalloc);

    if (path >= 3)
    {
        std::stringstream message;
        message << "'path' has to be smaller than 3";
        throw tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
    }
    if (inner_path >= 2)
    {
        std::stringstream message;
        message << "'inner_path' has to be smaller than 2";
        throw tools::invalid_argument(__FILE__, __LINE__, __func__, message.str());
    }

    std::unique_ptr<module::Controller> controller_ptr;
    if (cyclic_path)
        controller_ptr = std::unique_ptr<module::Controller>(new module::Controller_cyclic(3, 0));
    else
        controller_ptr = std::unique_ptr<module::Controller>(new module::Controller_static(path));
    module::Controller& controller = *controller_ptr;
    module::Controller_static controller_a(inner_path);
    module::Controller_static controller_b(inner_path);
    module::Switcher switchex(3, data_length, typeid(uint8_t));
    module::Switcher switch_a(2, data_length, typeid(uint8_t));
    module::Switcher switch_b(2, data_length, typeid(uint8_t));

    // modules creation
    module::Initializer<uint8_t> initializer(data_length);
    module::Finalizer<uint8_t> finalizer(data_length);

    std::vector<std::shared_ptr<module::Incrementer<uint8_t>>> incs(12);
    for (size_t s = 0; s < incs.size(); s++)
    {
        incs[s].reset(new module::Incrementer<uint8_t>(data_length));
        incs[s]->set_ns(sleep_time_us * 1000);
        incs[s]->set_custom_name("Inc" + std::to_string(s));
    }

    // sockets binding
    // clang-format off
    controller  (  "control"          ) = initializer[ "initialize::out"      ];
    switchex    [  "commute::in_data" ] = initializer[ "initialize::out"      ];
    switchex    [  "commute::in_ctrl" ] = controller [    "control::out"      ];
    // path 0: Switcher A, then +1
    controller_a(  "control"          ) = switchex   [    "commute::out_data0"];
    switch_a    [  "commute::in_data" ] = switchex   [    "commute::out_data0"];
    switch_a    [  "commute::in_ctrl" ] = controller_a[   "control::out"      ];
    (*incs[ 0]) [ "increment::in"     ] = switch_a   [    "commute::out_data0"];  // A, path 0: +1
    switch_a    [   "select::in_data0"] = (*incs[ 0])[  "increment::out"      ];
    (*incs[ 1]) [ "increment::in"     ] = switch_a   [    "commute::out_data1"];  // A, path 1: +2
    (*incs[ 2]) [ "increment::in"     ] = (*incs[ 1])[  "increment::out"      ];
    switch_a    [   "select::in_data1"] = (*incs[ 2])[  "increment::out"      ];
    (*incs[ 3]) [ "increment::in"     ] = switch_a   [     "select::out_data" ];
    switchex    [   "select::in_data0"] = (*incs[ 3])[  "increment::out"      ];
    // path 1: +1
    (*incs[ 4]) [ "increment::in"     ] = switchex   [    "commute::out_data1"];
    switchex    [   "select::in_data1"] = (*incs[ 4])[  "increment::out"      ];
    // path 2: Switcher B, bound directly to the outer select
    controller_b(  "control"          ) = switchex   [    "commute::out_data2"];
    switch_b    [  "commute::in_data" ] = switchex   [    "commute::out_data2"];
    switch_b    [  "commute::in_ctrl" ] = controller_b[   "control::out"      ];
    (*incs[ 5]) [ "increment::in"     ] = switch_b   [    "commute::out_data0"];  // B, path 0: +3
    (*incs[ 6]) [ "increment::in"     ] = (*incs[ 5])[  "increment::out"      ];
    (*incs[ 7]) [ "increment::in"     ] = (*incs[ 6])[  "increment::out"      ];
    switch_b    [   "select::in_data0"] = (*incs[ 7])[  "increment::out"      ];
    (*incs[ 8]) [ "increment::in"     ] = switch_b   [    "commute::out_data1"];  // B, path 1: +4
    (*incs[ 9]) [ "increment::in"     ] = (*incs[ 8])[  "increment::out"      ];
    (*incs[10]) [ "increment::in"     ] = (*incs[ 9])[  "increment::out"      ];
    (*incs[11]) [ "increment::in"     ] = (*incs[10])[  "increment::out"      ];
    switch_b    [   "select::in_data1"] = (*incs[11])[  "increment::out"      ];
    switchex    [   "select::in_data2"] = switch_b   [     "select::out_data" ];
    // end
    finalizer   [ "finalize::in"      ] = switchex   [     "select::out_data" ];
    // clang-format on

    runtime::Sequence sequence_nested_paths(initializer("initialize"), n_threads);
    sequence_nested_paths.set_n_frames(n_inter_frames);
    sequence_nested_paths.set_no_copy_mode(no_copy_mode);

    auto tid = 0;
    for (auto cur_initializer : sequence_nested_paths.get_cloned_modules<module::Initializer<uint8_t>>(initializer))
    {
        std::vector<std::vector<uint8_t>> init_data(n_inter_frames, std::vector<uint8_t>(data_length, 0));
        for (size_t f = 0; f < n_inter_frames; f++)
            std::fill(init_data[f].begin(), init_data[f].end(), tid * n_inter_frames + f);
        cur_initializer->set_init_data(init_data);
        tid++;
    }

    if (!dot_filepath.empty())
    {
        std::ofstream file(dot_filepath);
        sequence_nested_paths.export_dot(file);
    }

    // configuration of the sequence tasks
    for (auto& mod : sequence_nested_paths.get_modules<module::Module>(false))
        for (auto& tsk : mod->tasks)
        {
            tsk->reset();
            tsk->set_debug(debug);       // disable the debug mode
            tsk->set_debug_limit(16);    // display only the 16 first bits if the debug mode is enabled
            tsk->set_stats(print_stats); // enable the statistics
            tsk->set_fast(true);         // enable the fast mode (= disable the useless verifs in the tasks)
        }

    std::atomic<unsigned int> counter(0);
    auto t_start = std::chrono::steady_clock::now();
    if (!step_by_step)
    {
        // execute the sequence (multi-threaded)
        sequence_nested_paths.exec([&counter, n_exec]() { return ++counter >= n_exec; });
    }
    else
    {
        do
            for (size_t tid = 0; tid < n_threads; tid++)
                while (sequence_nested_paths.exec_step(tid))
                    ;
        while (++counter < (n_exec / n_threads));
    }
    auto duration = std::chrono::steady_clock::now() - t_start;

    auto elapsed_time = duration.count() / 1000.f / 1000.f;
    std::cout << "Sequence elapsed time: " << elapsed_time << " ms" << std::endl;

    // increments added by each outer path, for each inner path
    const size_t increments[3][2] = { { 2, 3 }, { 1, 1 }, { 3, 4 } };

    // verification of the sequence execution
    auto tests_passed = true;
    tid = 0;
    for (auto cur_finalizer : sequence_nested_paths.get_cloned_modules<module::Finalizer<uint8_t>>(finalizer))
    {
        size_t real_path = path;
        if (cyclic_path) real_path = ((*cur_finalizer)[module::fin::tsk::finalize].get_n_calls() - 1) % 3;
        for (size_t f = 0; f < n_inter_frames; f++)
        {
            const auto& final_data = cur_finalizer->get_final_data()[f];
            for (size_t d = 0; d < final_data.size(); d++)
            {
                auto expected = (int)(increments[real_path][inner_path] + (tid * n_inter_frames + f));
                expected = expected % 256;
                if (final_data[d] != expected)
                {
                    std::cout << "# expected = " << +expected << " - obtained = " << +final_data[d] << " (d = " << d
                              << ", tid = " << tid << ")" << std::endl;
                    tests_passed = false;
                    break;
                }
            }
        }
        tid++;
    }

    if (tests_passed)
        std::cout << "# " << rang::style::bold << rang::fg::green << "Tests passed!" << rang::style::reset << std::endl;
    else
        std::cout << "# " << rang::style::bold << rang::fg::red << "Tests failed :-(" << rang::style::reset
                  << std::endl;

    unsigned int test_results = !tests_passed;

    // display the statistics of the tasks (if enabled)
    if (print_stats)
    {
        std::cout << "#" << std::endl;
        tools::Stats::show(sequence_nested_paths.get_modules_per_types(), true, false);
    }

    sequence_nested_paths.set_n_frames(1);

    // clang-format off
    controller  ("control").unbind(initializer["initialize::out"]);
    switchex    ["commute::in_data" ].unbind(initializer ["initialize::out"   ]);
    switchex    ["commute::in_ctrl" ].unbind(controller  ["control::out"      ]);
    controller_a("control").unbind(switchex["commute::out_data0"]);
    switch_a    ["commute::in_data" ].unbind(switchex    ["commute::out_data0"]);
    switch_a    ["commute::in_ctrl" ].unbind(controller_a["control::out"      ]);
    (*incs[ 0]) ["increment::in"    ].unbind(switch_a    ["commute::out_data0"]);
    switch_a    ["select::in_data0" ].unbind((*incs[ 0]) ["increment::out"    ]);
    (*incs[ 1]) ["increment::in"    ].unbind(switch_a    ["commute::out_data1"]);
    (*incs[ 2]) ["increment::in"    ].unbind((*incs[ 1]) ["increment::out"    ]);
    switch_a    ["select::in_data1" ].unbind((*incs[ 2]) ["increment::out"    ]);
    (*incs[ 3]) ["increment::in"    ].unbind(switch_a    ["select::out_data"  ]);
    switchex    ["select::in_data0" ].unbind((*incs[ 3]) ["increment::out"    ]);
    (*incs[ 4]) ["increment::in"    ].unbind(switchex    ["commute::out_data1"]);
    switchex    ["select::in_data1" ].unbind((*incs[ 4]) ["increment::out"    ]);
    controller_b("control").unbind(switchex["commute::out_data2"]);
    switch_b    ["commute::in_data" ].unbind(switchex    ["commute::out_data2"]);
    switch_b    ["commute::in_ctrl" ].unbind(controller_b["control::out"      ]);
    (*incs[ 5]) ["increment::in"    ].unbind(switch_b    ["commute::out_data0"]);
    (*incs[ 6]) ["increment::in"    ].unbind((*incs[ 5]) ["increment::out"    ]);
    (*incs[ 7]) ["increment::in"    ].unbind((*incs[ 6]) ["increment::out"    ]);
    switch_b    ["select::in_data0" ].unbind((*incs[ 7]) ["increment::out"    ]);
    (*incs[ 8]) ["increment::in"    ].unbind(switch_b    ["commute::out_data1"]);
    (*incs[ 9]) ["increment::in"    ].unbind((*incs[ 8]) ["increment::out"    ]);
    (*incs[10]) ["increment::in"    ].unbind((*incs[ 9]) ["increment::out"    ]);
    (*incs[11]) ["increment::in"    ].unbind((*incs[10]) ["increment::out"    ]);
    switch_b    ["select::in_data1" ].unbind((*incs[11]) ["increment::out"    ]);
    switchex    ["select::in_data2" ].unbind(switch_b    ["select::out_data"  ]);
    finalizer   ["finalize::in"     ].unbind(switchex    ["select::out_data"  ]);
    // clang-format on

    return test_results;
}

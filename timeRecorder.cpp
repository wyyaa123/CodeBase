#include <chrono>
#include <cstdio>
// #include <unistd.h> // Linux
#include <Windows.h>

class TimeRecorder {
public:
    TimeRecorder() { start = std::chrono::high_resolution_clock::now(); }
    double get_time() { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count(); }
    void reset() {start = std::chrono::high_resolution_clock::now();}
private:
    // std::chrono::_V2::system_clock::time_point start;
    std::chrono::high_resolution_clock::time_point start;
};


int main(int argc, char** argv) {

    static TimeRecorder timer = TimeRecorder();
    for (int i = 0; i != 10; ++i) {
        printf("经过了 %.6f 秒\n", timer.get_time());
        sleep(1);
    }

    return 0;
}


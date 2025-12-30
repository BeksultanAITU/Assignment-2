#include <iostream>     // Для вывода в консоль (cout)
#include <vector>       // Для динамических массивов (std::vector)
#include <random>       // Для генерации случайных чисел
#include <chrono>       // Для измерения времени (chrono)
#include <limits>       // Для INT_MAX/INT_MIN через numeric_limits
#include <algorithm>    // Для std::is_sorted, std::swap
#include <omp.h>        // Для OpenMP директив и функций


static long long us_since(const std::chrono::high_resolution_clock::time_point& start,
                          const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}


// Генерация массива случайных чисел

static std::vector<int> make_random_vector(int n, int lo = 1, int hi = 100000) {
  
    // Здесь мы создаём генератор случайных чисел:
    // mt19937 — быстрый и качественный генератор псевдослучайных чисел.
    // seed берём из текущего времени, чтобы каждый запуск давал разные числа.
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // uniform_int_distribution задаёт равномерное распределение в диапазоне [lo, hi]
    std::uniform_int_distribution<int> dist(lo, hi);

    // Создаём вектор нужного размера
    std::vector<int> a(n);

    // Заполняем вектор случайными числами
    for (int i = 0; i < n; ++i) a[i] = dist(rng);

    return a;
}

// TASK 2: Min/Max sequential vs OpenMP
static void task2_minmax_seq_and_omp() {
  
    // По условию Task 2:
    // 1) создать массив из 10000 случайных чисел
    // 2) найти min и max:
    //    - последовательно
    //    - параллельно с OpenMP
    // 3) сравнить время выполнения и сделать выводы (выводим time и краткий итог)

    const int N = 10000;
    std::vector<int> a = make_random_vector(N);

    std::cout << "=============================\n";
    std::cout << "Task 2: Array Min/Max (Sequential vs OpenMP)\n";
    std::cout << "=============================\n";
    std::cout << "Task 2.1: Generated array of size N = " << N << "\n\n";

    // 2.2 Последовательный поиск min/max
    int min_seq = std::numeric_limits<int>::max();
    int max_seq = std::numeric_limits<int>::min();

  
    // Замеряем время: берём текущий момент, выполняем вычисление, берём конец.
    auto t1 = std::chrono::high_resolution_clock::now();

  
    // Последовательный проход по массиву: один поток, обычный for.
    for (int i = 0; i < N; ++i) {
        if (a[i] < min_seq) min_seq = a[i];
        if (a[i] > max_seq) max_seq = a[i];
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    long long seq_us = us_since(t1, t2);

    std::cout << "Task 2.2: Sequential result -> min=" << min_seq
              << ", max=" << max_seq
              << ", time=" << seq_us << " us\n";

    // 2.3 Параллельный поиск min/max с OpenMP
    int min_par = std::numeric_limits<int>::max();
    int max_par = std::numeric_limits<int>::min();

    auto t3 = std::chrono::high_resolution_clock::now();

  
    // Идея параллельного поиска min/max:
    // - Делим массив на части между потоками (omp for).
    // - Каждый поток ищет локальный минимум/максимум в своей части.
    // - Затем аккуратно объединяем локальные результаты в глобальные (через critical).
    //
    // Почему не просто shared min/max?
    // Потому что одновременная запись из потоков вызовет race condition.
    #pragma omp parallel
    {
      
        // Локальные переменные у каждого потока:
        // они не конфликтуют между потоками.
        int local_min = std::numeric_limits<int>::max();
        int local_max = std::numeric_limits<int>::min();

      
        // omp for делит итерации цикла между потоками.
        // nowait — чтобы потоки не ждали друг друга перед переходом к объединению,
        // но при этом мы всё равно корректно объединим результаты в critical.
        #pragma omp for nowait
        for (int i = 0; i < N; ++i) {
            if (a[i] < local_min) local_min = a[i];
            if (a[i] > local_max) local_max = a[i];
        }

      
        // critical гарантирует, что в этот блок входит только один поток одновременно.
        // Здесь мы объединяем локальные min/max в глобальные.
        #pragma omp critical
        {
            if (local_min < min_par) min_par = local_min;
            if (local_max > max_par) max_par = local_max;
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    long long par_us = us_since(t3, t4);

    std::cout << "Task 2.3: OpenMP result     -> min=" << min_par
              << ", max=" << max_par
              << ", time=" << par_us << " us\n";

  
    // Проверяем, совпали ли результаты.
    bool ok = (min_seq == min_par) && (max_seq == max_par);

    std::cout << "Task 2.4: Validation        -> " << (ok ? "OK (results match)" : "FAIL (results differ)") << "\n";

  
    // Краткий вывод: на маленьком N ускорение может быть небольшим,
    // потому что есть накладные расходы на создание/синхронизацию потоков.

}

// TASK 3: Selection Sort sequential and OpenMP


// Последовательная сортировка выбором

static void selection_sort_seq(std::vector<int>& a) {
  
    // Сортировка выбором (Selection Sort):
    // На шаге i ищем минимальный элемент на отрезке [i..n-1],
    // затем меняем местами a[i] и найденный минимум.
    // Время: O(n^2) — поэтому на больших размерах медленно.

    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;

      
        // Внутренний цикл ищет индекс минимального элемента.
        for (int j = i + 1; j < n; ++j) {
            if (a[j] < a[min_idx]) min_idx = j;
        }

      
        // Ставим найденный минимум на позицию i.
        if (min_idx != i) std::swap(a[i], a[min_idx]);
    }
}


// Параллельная сортировка выбором (OpenMP)

static void selection_sort_omp(std::vector<int>& a) {
  
    // Важно понять: selection sort сложно распараллелить полностью,
    // потому что внешний цикл i зависит от результата предыдущих шагов:
    // после каждой итерации меняется массив.
    //
    // Но можно распараллелить ПОИСК минимума в внутреннем цикле:
    // - внешняя итерация i остаётся последовательной
    // - поиск min на [i..n-1] выполняем параллельно

    int n = (int)a.size();
    for (int i = 0; i < n - 1; ++i) {
        // Глобальный минимум для текущей позиции i
        int global_min_val = a[i];
        int global_min_idx = i;

      
        // Запускаем параллельную область:
        // каждый поток будет искать локальный минимум на своей части диапазона.
        #pragma omp parallel
        {
            int local_min_val = global_min_val;
            int local_min_idx = global_min_idx;

          
            // Делим внутренний цикл между потоками.
            #pragma omp for nowait
            for (int j = i + 1; j < n; ++j) {
                if (a[j] < local_min_val) {
                    local_min_val = a[j];
                    local_min_idx = j;
                }
            }

          
            // Объединяем локальные минимумы в глобальный минимум.
            // Делаем это в critical, чтобы избежать гонок.
            #pragma omp critical
            {
                if (local_min_val < global_min_val) {
                    global_min_val = local_min_val;
                    global_min_idx = local_min_idx;
                }
            }
        }

      
        // После того как нашли минимальный элемент, делаем swap.
        if (global_min_idx != i) std::swap(a[i], a[global_min_idx]);
    }
}


// Запуск сортировки и замер времени для заданного N

static void run_sort_benchmark(int N) {
  
    // Для честного сравнения:
    // - генерируем один исходный массив base
    // - копируем его в a_seq и a_omp
    // - сортируем разными версиями
    // - проверяем, что оба результата отсортированы и одинаковы по смыслу

    std::vector<int> base = make_random_vector(N);

    std::vector<int> a_seq = base;
    std::vector<int> a_omp = base;

    // ---- Sequential ----
    auto t1 = std::chrono::high_resolution_clock::now();
    selection_sort_seq(a_seq);
    auto t2 = std::chrono::high_resolution_clock::now();
    long long seq_us = us_since(t1, t2);

    // ---- OpenMP ----
    auto t3 = std::chrono::high_resolution_clock::now();
    selection_sort_omp(a_omp);
    auto t4 = std::chrono::high_resolution_clock::now();
    long long omp_us = us_since(t3, t4);

  
    // Проверка корректности:
    // is_sorted — проверяет, что массив отсортирован по возрастанию.
    bool ok_seq = std::is_sorted(a_seq.begin(), a_seq.end());
    bool ok_omp = std::is_sorted(a_omp.begin(), a_omp.end());

  
    // Дополнительная проверка: результаты должны совпадать.
    // (Для одинакового input сортировка должна дать один и тот же отсортированный массив)
    bool same = (a_seq == a_omp);

    std::cout << "Task 3.1: N = " << N << "\n";
    std::cout << "  Sequential Selection Sort -> time=" << seq_us << " us, sorted=" << (ok_seq ? "YES" : "NO") << "\n";
    std::cout << "  OpenMP Selection Sort     -> time=" << omp_us << " us, sorted=" << (ok_omp ? "YES" : "NO") << "\n";
    std::cout << "  Validation (same result)  -> " << (same ? "OK" : "FAIL") << "\n\n";


}

static void task3_selection_sort_seq_and_omp() {
    std::cout << "=============================\n";
    std::cout << "Task 3: Selection Sort (Sequential vs OpenMP)\n";
    std::cout << "=============================\n";
    std::cout << "Task 3.1: Benchmark for N = 1000\n";
    run_sort_benchmark(1000);

    std::cout << "Task 3.2: Benchmark for N = 10000\n";
    run_sort_benchmark(10000);

}

int main() {
  
    // Главная функция запускает задачи по очереди:
    // Task 2
    std::cout << "=========================================\n";
    std::cout << "Assignment 2\n";
    std::cout << "=========================================\n\n";

    // Запуск Task 2
    task2_minmax_seq_and_omp();

    // Запуск Task 3
    task3_selection_sort_seq_and_omp();

   
    return 0;
}
